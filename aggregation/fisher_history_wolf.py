from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aggregation.base import Aggregator
from fl.types import AggregationResult, ClientUpdate
from models.param_groups import get_expert_id_from_name
from utils.state_dict_ops import check_finite_state_dict, clone_state_dict


class FisherHistoryWolfExpertAggregator(Aggregator):
    """
    Fisher-History-WoLF 专家聚合器。

    这个聚合器只用于 expert 参数组，不用于 non_expert 参数组。

    设计目标：
        在 fisher_only 的基础上，只修正 expert-wise 客户端权重，
        不改变 expert delta 聚合形式，不引入 old global prior / alpha / 方向一致性。

    核心思想：
        1. fisher_strength 是主信号，表示 client-expert 的平均 Fisher 敏感性。
        2. active_count 只作为 activation support，低支撑降权，高支撑不额外奖励。
        3. 服务器为每个 (client_id, expert_id) 维护历史 Fisher evidence 状态 mu/P。
        4. 当前 Fisher observation 和历史预测偏离太大时，用 WoLF-IMQ 降低该观测对历史状态的影响。
        5. 最终用 filtered_mu + log(support) 得到 expert-wise 客户端权重。

    最终聚合形式保持为：
        theta_new = theta_global + sum_i weight_i,e * delta_i,e
    """

    def __init__(self, cfg: Any, param_group_name: str) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)

        # 按 (expert_id, client_id) 保存历史滤波状态。
        # history_states[expert_id][client_id] = {"mu": ..., "P": ..., "age": ...}
        self.history_states: Dict[int, Dict[int, Dict[str, float]]] = {}

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "fisher_history_wolf"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        返回客户端 Fisher strength 总和。

        说明：
            fisher_history_wolf 的真实聚合权重是每个 expert 一套权重，
            因此这个函数不参与真正的 expert 聚合流程。

            这里保留实现只是为了满足 Aggregator 抽象接口。
        """
        self._validate_client_updates(client_updates)

        weights: Dict[int, float] = {}

        for update in client_updates:
            total_strength = 0.0
            expert_payloads = _get_expert_payloads(update)

            for payload in expert_payloads.values():
                active_count = _safe_int(payload.get("active_count", 0), default=0)
                if active_count < self.min_active_count:
                    continue

                mean_A = _safe_float(payload.get("mean_A", 0.0), default=0.0)
                mean_B = _safe_float(payload.get("mean_B", 0.0), default=0.0)
                fisher_strength = _safe_float(
                    payload.get("fisher_strength", mean_A * mean_B),
                    default=0.0,
                )

                if fisher_strength > 0.0 and math.isfinite(fisher_strength):
                    total_strength += float(fisher_strength)

            weights[int(update.client_id)] = float(total_strength)

        return weights

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        执行 Fisher-History-WoLF expert-wise delta 聚合。

        聚合方式：
            对每个 expert e：
                1. 收集每个客户端的 expert K-FAC evidence。
                2. 用 fisher_strength 构造 robust normalized log Fisher observation z。
                3. 用 active_count 只构造 support confidence，不作为高 usage 奖励。
                4. 对每个 client-expert 用 WoLF-IMQ 更新历史 mu/P。
                5. logit_i,e = mu_new_i,e + log(support_i,e + eps)。
                6. 在同一个 expert 内 softmax 得到客户端权重。
                7. 用 weight_i,e 聚合该 expert 的参数 delta。
        """
        if self.param_group_name != "expert":
            raise ValueError(
                "FisherHistoryWolfExpertAggregator 只能用于 expert 参数组，"
                f"当前 param_group_name={self.param_group_name}"
            )

        self._validate_client_updates(client_updates)

        names = _resolve_param_names(
            global_state=global_state,
            param_names=param_names,
        )

        expert_param_names = _group_param_names_by_expert(names)

        if len(expert_param_names) == 0:
            raise ValueError(
                "fisher_history_wolf 没有收到任何 expert 参数名。"
                "请确认 param_names 是否来自 self.param_groups.expert，"
                "以及模型参数名是否包含 experts.。"
            )

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        expert_weight_map: Dict[int, Dict[int, float]] = {}
        expert_record_map: Dict[int, List[Dict[str, Any]]] = {}
        expert_fallback_map: Dict[int, bool] = {}

        for expert_id, expert_names in sorted(expert_param_names.items()):
            raw_records = self._build_expert_records(
                expert_id=expert_id,
                client_updates=client_updates,
                strict=strict,
            )

            if len(raw_records) < self.min_valid_clients:
                # fallback=keep_global：
                # 不更新该 expert 参数，也不更新该 expert 的历史状态，避免少量观测污染历史。
                if self.fallback != "keep_global":
                    raise ValueError(
                        f"不支持的 fisher_history_wolf fallback：{self.fallback}。"
                        "当前只支持 keep_global。"
                    )

                expert_record_map[int(expert_id)] = raw_records
                expert_weight_map[int(expert_id)] = {}
                expert_fallback_map[int(expert_id)] = True
                continue

            records = self._filter_records_for_expert(
                expert_id=int(expert_id),
                records=raw_records,
            )
            expert_record_map[int(expert_id)] = records

            weights = _softmax_records(records)
            expert_weight_map[int(expert_id)] = weights
            expert_fallback_map[int(expert_id)] = False

            self._apply_expert_weighted_delta(
                global_state=global_state,
                new_state_dict=new_state_dict,
                client_updates=client_updates,
                expert_names=expert_names,
                weights=weights,
                strict=strict,
            )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=names,
        )

        avg_weights = _average_expert_weights(
            client_updates=client_updates,
            expert_weight_map=expert_weight_map,
            expert_fallback_map=expert_fallback_map,
        )

        diagnostics = self._build_fisher_wolf_diagnostics(
            client_updates=client_updates,
            param_names=names,
            expert_weight_map=expert_weight_map,
            expert_record_map=expert_record_map,
            expert_fallback_map=expert_fallback_map,
            avg_weights=avg_weights,
        )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=avg_weights,
            diagnostics=diagnostics,
        )

    @property
    def expert_fisher_cfg(self) -> Any:
        """读取 expert_fisher 配置块。"""
        return _cfg_get(self.cfg, "expert_fisher", {})

    @property
    def wolf_cfg(self) -> Any:
        """读取 fisher_history_wolf 配置块。"""
        return _cfg_get(self.cfg, "fisher_history_wolf", {})

    @property
    def min_active_count(self) -> int:
        """expert 有效参与聚合所需的最小 routed token 数。"""
        return int(_cfg_get(self.expert_fisher_cfg, "min_active_count", 1))

    @property
    def min_valid_clients(self) -> int:
        """每个 expert 至少需要多少个有效客户端，否则 keep_global。"""
        return int(_cfg_get(self.expert_fisher_cfg, "min_valid_clients", 2))

    @property
    def fallback(self) -> str:
        """有效客户端不足时的 fallback 策略。"""
        return str(_cfg_get(self.expert_fisher_cfg, "fallback", "keep_global")).lower()

    @property
    def eps(self) -> float:
        """数值稳定项。"""
        return float(
            _cfg_get(
                self.wolf_cfg,
                "eps",
                _cfg_get(self.expert_fisher_cfg, "eps", 1.0e-8),
            )
        )

    @property
    def init_P(self) -> float:
        """新 client-expert 历史状态的初始不确定性。"""
        return float(_cfg_get(self.wolf_cfg, "init_P", 1.0))

    @property
    def process_noise_Q(self) -> float:
        """历史 Fisher evidence 状态的过程噪声。"""
        return float(_cfg_get(self.wolf_cfg, "process_noise_Q", 0.05))

    @property
    def observation_R(self) -> float:
        """
        normalized z 的基础观测噪声。

        因为 z 是同 expert 内 robust normalized log Fisher，
        所以 R=1 表示约 1 个 MAD 单位的正常观测波动。
        """
        return float(_cfg_get(self.wolf_cfg, "observation_R", 1.0))

    @property
    def robust_c(self) -> float:
        """WoLF-IMQ 的软阈值。"""
        return float(_cfg_get(self.wolf_cfg, "robust_c", 2.0))

    @property
    def diagnostics_enabled(self) -> bool:
        """是否生成 fisher_history_wolf 诊断字段。"""
        return bool(_cfg_get(self.wolf_cfg, "diagnostics_enabled", True))

    @property
    def diagnostics_include_records(self) -> bool:
        """是否在 summary.json 中保存完整 records 和 expert_weights。"""
        return bool(_cfg_get(self.wolf_cfg, "diagnostics_include_records", False))

    def _build_expert_records(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
        strict: bool,
    ) -> List[Dict[str, Any]]:
        """
        为单个 expert 收集所有有效客户端的原始 Fisher record。

        注意：
            fisher_history_wolf 的有效性主要看 fisher_strength，
            payload["score"] 只保留做诊断，不作为最终权重主信号。
        """
        records: List[Dict[str, Any]] = []

        for update in client_updates:
            payload = _get_single_expert_payload(
                update=update,
                expert_id=expert_id,
                strict=strict,
            )

            if payload is None:
                continue

            active_count = _safe_int(payload.get("active_count", 0), default=0)
            mean_A = _safe_float(payload.get("mean_A", 0.0), default=0.0)
            mean_B = _safe_float(payload.get("mean_B", 0.0), default=0.0)
            fisher_strength = _safe_float(
                payload.get("fisher_strength", mean_A * mean_B),
                default=0.0,
            )
            score = _extract_score(payload)

            is_valid = (
                active_count >= self.min_active_count
                and fisher_strength > 0.0
                and math.isfinite(fisher_strength)
            )

            if not is_valid:
                continue

            records.append(
                {
                    "client_id": int(update.client_id),
                    "active_count": int(active_count),
                    "mean_A": float(mean_A),
                    "mean_B": float(mean_B),
                    "fisher_strength": float(fisher_strength),
                    "score": float(score),
                }
            )

        return records

    def _filter_records_for_expert(
        self,
        expert_id: int,
        records: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对单个 expert 的 records 计算：
            z: robust normalized log Fisher observation
            support: active_count 低支撑降权
            rho: WoLF-IMQ 可靠性
            K: Kalman-style gain
            mu_new/P_new: 更新后的历史 Fisher evidence 状态
            logit: 最终 softmax logit
        """
        if len(records) == 0:
            return []

        eps = self.eps

        h_values = [
            math.log(max(float(record["fisher_strength"]), 0.0) + eps)
            for record in records
        ]
        h_median = _median(h_values)
        h_mad = _median([abs(value - h_median) for value in h_values])

        active_values = [
            float(record["active_count"])
            for record in records
            if float(record["active_count"]) > 0.0
        ]
        active_median = _median(active_values) if len(active_values) > 0 else 1.0
        active_median = max(active_median, eps)

        expert_states = self.history_states.setdefault(int(expert_id), {})
        filtered_records: List[Dict[str, Any]] = []

        for record, h_value in zip(records, h_values):
            client_id = int(record["client_id"])

            # 如果本轮同 expert 内 Fisher 几乎没有差异，则不制造虚假的极端 z。
            if h_mad <= eps:
                z = 0.0
            else:
                z = (h_value - h_median) / (h_mad + eps)

            active_count = float(record["active_count"])
            support = min(1.0, active_count / (active_median + eps))
            support = max(0.0, float(support))

            old_state = expert_states.get(client_id)

            if old_state is None:
                # 冷启动：第一次有效观测不降权，避免误伤前几轮快速学习。
                cold_start = True
                mu_pred = float(z)
                P_pred = float(self.init_P)
                residual = 0.0
                d2 = 0.0
                rho = 1.0
                R_eff = float(self.observation_R)
                kalman_gain = 1.0
                mu_new = float(z)
                P_new = float(self.init_P)
                age = 1.0
            else:
                cold_start = False
                mu_old = _safe_float(old_state.get("mu", 0.0), default=0.0)
                P_old = max(
                    _safe_float(old_state.get("P", self.init_P), default=self.init_P),
                    eps,
                )
                age_old = _safe_float(old_state.get("age", 0.0), default=0.0)

                mu_pred = float(mu_old)
                P_pred = float(P_old + self.process_noise_Q)
                residual = float(z - mu_pred)

                denom = P_pred + self.observation_R + eps
                d2 = float((residual * residual) / denom)

                robust_c = max(self.robust_c, eps)
                rho = float((1.0 + d2 / (robust_c * robust_c)) ** -0.5)

                # WoLF 的作用等价于把观测精度乘 rho^2，
                # 这里写成有效观测噪声 R_eff = R / rho^2。
                R_eff = float(self.observation_R / (rho * rho + eps))
                kalman_gain = float(P_pred / (P_pred + R_eff + eps))

                mu_new = float(mu_pred + kalman_gain * residual)
                P_new = float(max((1.0 - kalman_gain) * P_pred, eps))
                age = float(age_old + 1.0)

            # 这两个派生量只用于诊断：
            # abs_residual 避免正负 residual 抵消；
            # mu_update_abs 直接反映历史状态本轮被改动了多少。
            abs_residual = abs(float(residual))
            mu_update_abs = abs(float(mu_new - mu_pred))

            expert_states[client_id] = {
                "mu": float(mu_new),
                "P": float(P_new),
                "age": float(age),
            }

            # 四种状态只用于诊断，不写手工 gate。
            current_good = bool(z >= 0.0)
            history_good = bool(mu_pred >= 0.0)
            state_label = _state_label(
                current_good=current_good,
                history_good=history_good,
            )

            logit = float(mu_new + math.log(support + eps))

            enriched = dict(record)
            enriched.update(
                {
                    "h_log_fisher": float(h_value),
                    "z": float(z),
                    "support": float(support),
                    "mu_pred": float(mu_pred),
                    "P_pred": float(P_pred),
                    "residual": float(residual),
                    "abs_residual": float(abs_residual),
                    "d2": float(d2),
                    "rho": float(rho),
                    "R_eff": float(R_eff),
                    "kalman_gain": float(kalman_gain),
                    "mu_new": float(mu_new),
                    "mu_update_abs": float(mu_update_abs),
                    "P_new": float(P_new),
                    "age": float(age),
                    "cold_start": bool(cold_start),
                    "current_good": current_good,
                    "history_good": history_good,
                    "state_label": state_label,
                    "logit": float(logit),
                }
            )
            filtered_records.append(enriched)

        return filtered_records

    def _apply_expert_weighted_delta(
        self,
        global_state: Mapping[str, torch.Tensor],
        new_state_dict: Dict[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        expert_names: Sequence[str],
        weights: Mapping[int, float],
        strict: bool,
    ) -> None:
        """
        对单个 expert 的参数执行加权 delta 聚合。

        公式：
            theta_new = theta_global + sum_i weight_i,e * delta_i,e
        """
        update_by_client_id = {
            int(update.client_id): update
            for update in client_updates
        }

        for name in expert_names:
            if name not in global_state:
                if strict:
                    raise KeyError(f"global_state 缺少参数：{name}")
                continue

            global_tensor = global_state[name]

            # 非浮点 tensor 不参与 delta 聚合，保留 base_state 中的值。
            if not torch.is_floating_point(global_tensor):
                continue

            total_delta = torch.zeros_like(global_tensor)

            for client_id, weight in weights.items():
                if client_id not in update_by_client_id:
                    if strict:
                        raise KeyError(f"client_updates 中缺少客户端 {client_id}")
                    continue

                update = update_by_client_id[client_id]

                if name not in update.model_delta:
                    if strict:
                        raise KeyError(
                            f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                        )
                    continue

                delta_tensor = update.model_delta[name].to(global_tensor.device)
                total_delta = total_delta + float(weight) * delta_tensor

            new_state_dict[name] = global_tensor + total_delta

    def _build_fisher_wolf_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        param_names: Sequence[str],
        expert_weight_map: Mapping[int, Mapping[int, float]],
        expert_record_map: Mapping[int, Sequence[Mapping[str, Any]]],
        expert_fallback_map: Mapping[int, bool],
        avg_weights: Mapping[int, float],
    ) -> Dict[str, Any]:
        """
        构建 fisher_history_wolf 聚合诊断信息。

        诊断目标：
            1. 看 WoLF 是否在压异常 Fisher 尖峰。
            2. 看权重是否仍保留 Fisher 区分度。
            3. 看 active_count 是否又重新支配权重。
            4. 看四种 current/history 状态比例。
        """
        num_experts = int(len(expert_record_map))
        num_fallback_experts = int(
            sum(1 for value in expert_fallback_map.values() if value)
        )

        if not self.diagnostics_enabled:
            return {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "fisher_wolf_diag_enabled": False,
                "num_clients": int(len(client_updates)),
                "param_count": int(len(param_names)),
                "num_experts": num_experts,
                "num_fallback_experts": num_fallback_experts,
                "fallback_ratio": _safe_divide(
                    num_fallback_experts,
                    max(num_experts, 1),
                ),
            }

        include_records = self.diagnostics_include_records
        expert_diagnostics: Dict[int, Dict[str, Any]] = {}

        for expert_id in sorted(expert_record_map.keys()):
            records = list(expert_record_map[expert_id])
            weights = dict(expert_weight_map.get(expert_id, {}))
            fallback = bool(expert_fallback_map.get(expert_id, False))

            active_counts = [float(record.get("active_count", 0.0)) for record in records]
            fisher_strengths = [
                float(record.get("fisher_strength", 0.0))
                for record in records
            ]
            scores = [float(record.get("score", 0.0)) for record in records]
            z_values = [float(record.get("z", 0.0)) for record in records]
            supports = [float(record.get("support", 0.0)) for record in records]
            rho_values = [float(record.get("rho", 0.0)) for record in records]
            gain_values = [float(record.get("kalman_gain", 0.0)) for record in records]
            mu_new_values = [float(record.get("mu_new", 0.0)) for record in records]
            residual_values = [float(record.get("residual", 0.0)) for record in records]
            abs_residual_values = [
                float(record.get("abs_residual", abs(float(record.get("residual", 0.0)))))
                for record in records
            ]
            mu_update_abs_values = [
                float(record.get("mu_update_abs", 0.0))
                for record in records
            ]
            cold_start_values = [
                1.0 if bool(record.get("cold_start", False)) else 0.0
                for record in records
            ]
            ages = [float(record.get("age", 0.0)) for record in records]

            record_weights = [
                float(weights.get(int(record.get("client_id", -1)), 0.0))
                for record in records
            ]

            top_client = None
            if len(weights) > 0:
                top_client = max(weights.items(), key=lambda item: item[1])[0]

            status_counts = self._count_expert_payload_status(
                expert_id=expert_id,
                client_updates=client_updates,
            )

            state_counts = _count_state_labels(records)
            valid_count = max(len(records), 1)

            weight_entropy = _weight_entropy(weights)
            weight_entropy_norm = _weight_entropy_norm(weights)
            effective_clients = _effective_clients(weights)
            top1_weight, top2_weight, top1_gap = _top_weight_stats(weights)

            expert_diag: Dict[str, Any] = {
                "fallback": fallback,
                "fallback_reason": (
                    "valid_clients_lt_min_valid_clients" if fallback else None
                ),
                "valid_clients": int(len(records)),
                "invalid_clients": int(status_counts["invalid_clients"]),
                "missing_payload_clients": int(status_counts["missing_payload_clients"]),
                "zero_fisher_clients": int(status_counts["zero_fisher_clients"]),
                "zero_active_clients": int(status_counts["zero_active_clients"]),
                "nan_fisher_clients": int(status_counts["nan_fisher_clients"]),
                "min_valid_clients": int(self.min_valid_clients),
                "min_active_count": int(self.min_active_count),
                "top_client": int(top_client) if top_client is not None else None,
                "weight_entropy": float(weight_entropy),
                "weight_entropy_norm": float(weight_entropy_norm),
                "effective_clients": float(effective_clients),
                "weight_min": min(weights.values()) if len(weights) > 0 else 0.0,
                "weight_max": max(weights.values()) if len(weights) > 0 else 0.0,
                "top1_weight": float(top1_weight),
                "top2_weight": float(top2_weight),
                "top1_gap": float(top1_gap),
                "score_stats": _stat_dict(scores),
                "active_count_stats": _stat_dict(active_counts),
                "fisher_strength_stats": _stat_dict(fisher_strengths),
                "z_stats": _stat_dict(z_values),
                "support_stats": _stat_dict(supports),
                "rho_stats": _stat_dict(rho_values),
                "rho_p10": _percentile_clean(rho_values, 10.0),
                "kalman_gain_stats": _stat_dict(gain_values),
                "mu_new_stats": _stat_dict(mu_new_values),
                "residual_stats": _stat_dict(residual_values),
                "abs_residual_stats": _stat_dict(abs_residual_values),
                "abs_residual_p90": _percentile_clean(abs_residual_values, 90.0),
                "mu_update_abs_stats": _stat_dict(mu_update_abs_values),
                "cold_start_frac": _mean_clean(cold_start_values),
                "age_stats": _stat_dict(ages),
                "score_cv": _coefficient_of_variation(scores),
                "active_count_cv": _coefficient_of_variation(active_counts),
                "fisher_strength_cv": _coefficient_of_variation(fisher_strengths),
                "weight_active_corr": _pearson_corr(record_weights, active_counts),
                "weight_fisher_corr": _pearson_corr(record_weights, fisher_strengths),
                "weight_support_corr": _pearson_corr(record_weights, supports),
                "state_hh_frac": float(state_counts["hh"] / valid_count),
                "state_hl_frac": float(state_counts["hl"] / valid_count),
                "state_lh_frac": float(state_counts["lh"] / valid_count),
                "state_ll_frac": float(state_counts["ll"] / valid_count),
            }

            if include_records:
                expert_diag["weights"] = {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                expert_diag["records"] = [
                    {
                        "client_id": int(record.get("client_id", -1)),
                        "active_count": int(record.get("active_count", 0)),
                        "mean_A": float(record.get("mean_A", 0.0)),
                        "mean_B": float(record.get("mean_B", 0.0)),
                        "fisher_strength": float(record.get("fisher_strength", 0.0)),
                        "score": float(record.get("score", 0.0)),
                        "z": float(record.get("z", 0.0)),
                        "support": float(record.get("support", 0.0)),
                        "rho": float(record.get("rho", 0.0)),
                        "kalman_gain": float(record.get("kalman_gain", 0.0)),
                        "mu_pred": float(record.get("mu_pred", 0.0)),
                        "mu_new": float(record.get("mu_new", 0.0)),
                        "abs_residual": float(record.get("abs_residual", 0.0)),
                        "mu_update_abs": float(record.get("mu_update_abs", 0.0)),
                        "P_new": float(record.get("P_new", 0.0)),
                        "cold_start": bool(record.get("cold_start", False)),
                        "state_label": str(record.get("state_label", "unknown")),
                        "logit": float(record.get("logit", 0.0)),
                        "weight": float(
                            weights.get(int(record.get("client_id", -1)), 0.0)
                        ),
                    }
                    for record in records
                ]

            expert_diagnostics[int(expert_id)] = expert_diag

        all_expert_diags = list(expert_diagnostics.values())
        non_fallback_diags = [
            diag
            for diag in all_expert_diags
            if not bool(diag.get("fallback", False))
        ]

        diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "fisher_wolf_diag_enabled": True,
            "diagnostics_include_records": bool(include_records),
            "num_clients": int(len(client_updates)),
            "param_count": int(len(param_names)),
            "num_experts": num_experts,
            "num_fallback_experts": num_fallback_experts,
            "fallback_ratio": _safe_divide(num_fallback_experts, max(num_experts, 1)),
            "fallback_experts": [
                int(expert_id)
                for expert_id, fallback in sorted(expert_fallback_map.items())
                if fallback
            ],
            "mean_valid_clients": _mean_clean(
                [diag.get("valid_clients", 0.0) for diag in all_expert_diags]
            ),
            "mean_weight_entropy_norm": _mean_clean(
                [diag.get("weight_entropy_norm", 0.0) for diag in non_fallback_diags]
            ),
            "mean_effective_clients": _mean_clean(
                [diag.get("effective_clients", 0.0) for diag in non_fallback_diags]
            ),
            "mean_weight_max": _mean_clean(
                [diag.get("weight_max", 0.0) for diag in non_fallback_diags]
            ),
            "mean_score_cv": _mean_clean(
                [diag.get("score_cv", 0.0) for diag in all_expert_diags]
            ),
            "mean_active_count_cv": _mean_clean(
                [diag.get("active_count_cv", 0.0) for diag in all_expert_diags]
            ),
            "mean_fisher_strength_cv": _mean_clean(
                [diag.get("fisher_strength_cv", 0.0) for diag in all_expert_diags]
            ),
            "mean_weight_active_corr": _mean_clean(
                [diag.get("weight_active_corr", 0.0) for diag in non_fallback_diags]
            ),
            "mean_weight_fisher_corr": _mean_clean(
                [diag.get("weight_fisher_corr", 0.0) for diag in non_fallback_diags]
            ),
            "mean_weight_support_corr": _mean_clean(
                [diag.get("weight_support_corr", 0.0) for diag in non_fallback_diags]
            ),
            "mean_rho": _mean_clean(
                [diag.get("rho_stats", {}).get("mean", 0.0) for diag in all_expert_diags]
            ),
            "mean_rho_p10": _mean_clean(
                [diag.get("rho_p10", 0.0) for diag in all_expert_diags]
            ),
            "mean_abs_residual": _mean_clean(
                [
                    diag.get("abs_residual_stats", {}).get("mean", 0.0)
                    for diag in all_expert_diags
                ]
            ),
            "mean_abs_residual_p90": _mean_clean(
                [diag.get("abs_residual_p90", 0.0) for diag in all_expert_diags]
            ),
            "mean_mu_update_abs": _mean_clean(
                [
                    diag.get("mu_update_abs_stats", {}).get("mean", 0.0)
                    for diag in all_expert_diags
                ]
            ),
            "mean_cold_start_frac": _mean_clean(
                [diag.get("cold_start_frac", 0.0) for diag in all_expert_diags]
            ),
            "mean_kalman_gain": _mean_clean(
                [
                    diag.get("kalman_gain_stats", {}).get("mean", 0.0)
                    for diag in all_expert_diags
                ]
            ),
            "mean_support": _mean_clean(
                [
                    diag.get("support_stats", {}).get("mean", 0.0)
                    for diag in all_expert_diags
                ]
            ),
            "mean_state_hh_frac": _mean_clean(
                [diag.get("state_hh_frac", 0.0) for diag in all_expert_diags]
            ),
            "mean_state_hl_frac": _mean_clean(
                [diag.get("state_hl_frac", 0.0) for diag in all_expert_diags]
            ),
            "mean_state_lh_frac": _mean_clean(
                [diag.get("state_lh_frac", 0.0) for diag in all_expert_diags]
            ),
            "mean_state_ll_frac": _mean_clean(
                [diag.get("state_ll_frac", 0.0) for diag in all_expert_diags]
            ),
            "expert_diagnostics": expert_diagnostics,
        }

        if include_records:
            diagnostics["weights"] = {
                int(client_id): float(weight)
                for client_id, weight in avg_weights.items()
            }
            diagnostics["expert_weights"] = {
                int(expert_id): {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                for expert_id, weights in expert_weight_map.items()
            }

        return diagnostics

    def _count_expert_payload_status(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[str, int]:
        """
        统计单个 expert 的原始 payload 状态。
        """
        missing_payload_clients = 0
        zero_fisher_clients = 0
        zero_active_clients = 0
        nan_fisher_clients = 0
        invalid_clients = 0

        for update in client_updates:
            payload = _get_single_expert_payload(
                update=update,
                expert_id=expert_id,
                strict=False,
            )

            if payload is None:
                missing_payload_clients += 1
                invalid_clients += 1
                continue

            active_count = _safe_int(payload.get("active_count", 0), default=0)
            mean_A = _safe_float(payload.get("mean_A", 0.0), default=0.0)
            mean_B = _safe_float(payload.get("mean_B", 0.0), default=0.0)
            fisher_strength = _safe_float(
                payload.get("fisher_strength", mean_A * mean_B),
                default=0.0,
            )

            is_nan_fisher = not math.isfinite(fisher_strength)
            is_zero_fisher = fisher_strength <= 0.0
            is_zero_active = active_count <= 0

            if is_nan_fisher:
                nan_fisher_clients += 1

            if is_zero_fisher:
                zero_fisher_clients += 1

            if is_zero_active:
                zero_active_clients += 1

            if (
                active_count < self.min_active_count
                or is_zero_fisher
                or is_nan_fisher
            ):
                invalid_clients += 1

        return {
            "missing_payload_clients": int(missing_payload_clients),
            "zero_fisher_clients": int(zero_fisher_clients),
            "zero_active_clients": int(zero_active_clients),
            "nan_fisher_clients": int(nan_fisher_clients),
            "invalid_clients": int(invalid_clients),
        }


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """
    解析当前聚合器需要处理的参数名。

    如果 param_names=None，则默认从 global_state 中筛选所有 expert 参数。
    """
    if param_names is not None:
        return list(param_names)

    return [
        name
        for name in global_state.keys()
        if get_expert_id_from_name(name) is not None
    ]


def _group_param_names_by_expert(
    param_names: Sequence[str],
) -> Dict[int, List[str]]:
    """把 expert 参数名按照 expert_id 分组。"""
    result: Dict[int, List[str]] = {}

    for name in param_names:
        expert_id = get_expert_id_from_name(name)
        if expert_id is None:
            continue

        result.setdefault(int(expert_id), []).append(name)

    return {
        int(expert_id): names
        for expert_id, names in sorted(result.items())
    }


def _get_expert_payloads(update: ClientUpdate) -> Mapping[Any, Any]:
    """
    从 ClientUpdate.extra 中读取 expert_kfac payload。

    支持两种格式：
        extra["expert_kfac"]["experts"]
        extra["expert_kfac"]
    """
    if "expert_kfac" not in update.extra:
        raise KeyError(
            f"客户端 {update.client_id} 缺少 extra['expert_kfac']。"
            "请确认 expert_fisher.enabled=true，且 client.py 已经调用 "
            "collect_expert_kfac_stats(...)。"
        )

    payload = update.extra["expert_kfac"]

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"客户端 {update.client_id} 的 extra['expert_kfac'] 类型错误，"
            f"期望 Mapping，实际是 {type(payload)}。"
        )

    experts = payload.get("experts", payload)

    if not isinstance(experts, Mapping):
        raise TypeError(
            f"客户端 {update.client_id} 的 expert_kfac['experts'] 类型错误，"
            f"期望 Mapping，实际是 {type(experts)}。"
        )

    return experts


def _get_single_expert_payload(
    update: ClientUpdate,
    expert_id: int,
    strict: bool,
) -> Optional[Mapping[str, Any]]:
    """
    读取单个客户端、单个 expert 的 K-FAC payload。

    同时兼容 int key 和 str key。
    """
    try:
        experts = _get_expert_payloads(update)
    except (KeyError, TypeError):
        if strict:
            raise
        return None

    expert_payload = None

    if expert_id in experts:
        expert_payload = experts[expert_id]
    elif str(expert_id) in experts:
        expert_payload = experts[str(expert_id)]

    if expert_payload is None:
        if strict:
            raise KeyError(
                f"客户端 {update.client_id} 的 expert_kfac 中缺少 expert {expert_id}。"
            )
        return None

    if not isinstance(expert_payload, Mapping):
        if strict:
            raise TypeError(
                f"客户端 {update.client_id} 的 expert {expert_id} payload 类型错误，"
                f"期望 Mapping，实际是 {type(expert_payload)}。"
            )
        return None

    return expert_payload


def _extract_score(payload: Mapping[str, Any]) -> float:
    """
    从 expert payload 中提取 fisher_only score。

    注意：
        这个 score 只用于诊断，不作为 fisher_history_wolf 最终权重主信号。
    """
    if "score" in payload:
        return _safe_float(payload.get("score", 0.0), default=0.0)

    active_count = _safe_float(payload.get("active_count", 0.0), default=0.0)

    if "fisher_strength" in payload:
        fisher_strength = _safe_float(
            payload.get("fisher_strength", 0.0),
            default=0.0,
        )
        return float(active_count * fisher_strength)

    mean_A = _safe_float(payload.get("mean_A", 0.0), default=0.0)
    mean_B = _safe_float(payload.get("mean_B", 0.0), default=0.0)

    return float(active_count * mean_A * mean_B)


def _softmax_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[int, float]:
    """对单个 expert 的客户端 logit 做 softmax。"""
    if len(records) == 0:
        return {}

    logits = torch.tensor(
        [float(record["logit"]) for record in records],
        dtype=torch.float64,
    )
    weights = torch.softmax(logits, dim=0).tolist()

    return {
        int(record["client_id"]): float(weight)
        for record, weight in zip(records, weights)
    }


def _average_expert_weights(
    client_updates: Sequence[ClientUpdate],
    expert_weight_map: Mapping[int, Mapping[int, float]],
    expert_fallback_map: Mapping[int, bool],
) -> Dict[int, float]:
    """
    把 expert-wise 权重压成一套 client-wise 平均权重，仅用于诊断。

    真实聚合使用的是 expert_weight_map。
    """
    client_ids = [int(update.client_id) for update in client_updates]
    avg_weights = {
        client_id: 0.0
        for client_id in client_ids
    }

    num_non_fallback_experts = 0

    for expert_id, weights in expert_weight_map.items():
        if bool(expert_fallback_map.get(expert_id, False)):
            continue

        if len(weights) == 0:
            continue

        num_non_fallback_experts += 1

        for client_id in client_ids:
            avg_weights[client_id] += float(weights.get(client_id, 0.0))

    if num_non_fallback_experts <= 0:
        return avg_weights

    for client_id in avg_weights:
        avg_weights[client_id] /= float(num_non_fallback_experts)

    return avg_weights


def _weight_entropy(weights: Mapping[int, float]) -> float:
    """计算权重熵。"""
    entropy = 0.0

    for weight in weights.values():
        weight = float(weight)
        if weight <= 0.0:
            continue
        entropy -= weight * math.log(weight + 1.0e-12)

    return float(entropy)


def _weight_entropy_norm(weights: Mapping[int, float]) -> float:
    """
    计算归一化权重熵。

    接近 1：权重接近 uniform。
    接近 0：单个客户端强烈支配。
    """
    if len(weights) <= 1:
        return 0.0

    entropy = _weight_entropy(weights)
    max_entropy = math.log(float(len(weights)) + 1.0e-12)

    return _safe_divide(entropy, max_entropy)


def _effective_clients(weights: Mapping[int, float]) -> float:
    """
    计算有效客户端数：1 / sum_i w_i^2。
    """
    if len(weights) == 0:
        return 0.0

    square_sum = 0.0

    for weight in weights.values():
        square_sum += float(weight) ** 2

    if square_sum <= 0.0:
        return 0.0

    return float(1.0 / square_sum)


def _top_weight_stats(
    weights: Mapping[int, float],
) -> Tuple[float, float, float]:
    """返回 top1_weight、top2_weight、top1_gap。"""
    if len(weights) == 0:
        return 0.0, 0.0, 0.0

    sorted_weights = sorted(
        [float(weight) for weight in weights.values()],
        reverse=True,
    )

    top1 = sorted_weights[0]
    top2 = sorted_weights[1] if len(sorted_weights) >= 2 else 0.0

    return float(top1), float(top2), float(top1 - top2)


def _coefficient_of_variation(
    values: Sequence[Any],
    eps: float = 1.0e-12,
) -> float:
    """计算变异系数 CV = std / abs(mean)。"""
    clean_values = [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]

    if len(clean_values) <= 1:
        return 0.0

    mean = sum(clean_values) / len(clean_values)
    if abs(mean) <= eps:
        return 0.0

    var = sum((value - mean) ** 2 for value in clean_values) / len(clean_values)
    std = math.sqrt(max(var, 0.0))

    return float(std / (abs(mean) + eps))


def _pearson_corr(
    xs: Sequence[Any],
    ys: Sequence[Any],
    eps: float = 1.0e-12,
) -> float:
    """计算 Pearson 相关系数。"""
    clean_pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if _is_finite_number(x) and _is_finite_number(y)
    ]

    if len(clean_pairs) <= 1:
        return 0.0

    clean_xs = [pair[0] for pair in clean_pairs]
    clean_ys = [pair[1] for pair in clean_pairs]

    mean_x = sum(clean_xs) / len(clean_xs)
    mean_y = sum(clean_ys) / len(clean_ys)

    centered_xs = [value - mean_x for value in clean_xs]
    centered_ys = [value - mean_y for value in clean_ys]

    numerator = sum(x * y for x, y in zip(centered_xs, centered_ys))
    denom_x = math.sqrt(sum(x * x for x in centered_xs))
    denom_y = math.sqrt(sum(y * y for y in centered_ys))

    denominator = denom_x * denom_y
    if denominator <= eps:
        return 0.0

    return float(numerator / (denominator + eps))


def _stat_dict(values: Sequence[Any]) -> Dict[str, float]:
    """生成一组数值的基础统计量。"""
    clean_values = [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]

    if len(clean_values) == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    mean = sum(clean_values) / len(clean_values)
    var = sum((value - mean) ** 2 for value in clean_values) / len(clean_values)
    std = math.sqrt(max(var, 0.0))

    return {
        "count": float(len(clean_values)),
        "mean": float(mean),
        "std": float(std),
        "min": float(min(clean_values)),
        "max": float(max(clean_values)),
    }


def _mean_clean(values: Sequence[Any]) -> float:
    """对有限数值求平均。"""
    clean_values = [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]

    if len(clean_values) == 0:
        return 0.0

    return float(sum(clean_values) / len(clean_values))


def _percentile_clean(
    values: Sequence[Any],
    percentile: float,
) -> float:
    """计算百分位数，使用线性插值。"""
    clean_values = sorted(
        float(value)
        for value in values
        if _is_finite_number(value)
    )

    if len(clean_values) == 0:
        return 0.0

    if len(clean_values) == 1:
        return float(clean_values[0])

    percentile = min(max(float(percentile), 0.0), 100.0)
    pos = (percentile / 100.0) * (len(clean_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))

    if low == high:
        return float(clean_values[low])

    weight = pos - low
    return float(clean_values[low] * (1.0 - weight) + clean_values[high] * weight)


def _median(values: Sequence[Any]) -> float:
    """计算中位数。"""
    clean_values = sorted(
        float(value)
        for value in values
        if _is_finite_number(value)
    )

    if len(clean_values) == 0:
        return 0.0

    mid = len(clean_values) // 2

    if len(clean_values) % 2 == 1:
        return float(clean_values[mid])

    return float((clean_values[mid - 1] + clean_values[mid]) / 2.0)


def _count_state_labels(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """统计四种 current/history 状态。"""
    counts = {
        "hh": 0,
        "hl": 0,
        "lh": 0,
        "ll": 0,
    }

    for record in records:
        label = str(record.get("state_label", ""))
        if label in counts:
            counts[label] += 1

    return counts


def _state_label(
    current_good: bool,
    history_good: bool,
) -> str:
    """
    返回四种状态标签。

    hh: 当前好，历史好
    hl: 当前好，历史不好
    lh: 当前不好，历史好
    ll: 当前不好，历史不好
    """
    if current_good and history_good:
        return "hh"

    if current_good and not history_good:
        return "hl"

    if not current_good and history_good:
        return "lh"

    return "ll"


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """安全除法。"""
    numerator = _safe_float(numerator, default=0.0)
    denominator = _safe_float(denominator, default=0.0)

    if denominator == 0.0:
        return float(default)

    return float(numerator / denominator)


def _is_finite_number(value: Any) -> bool:
    """判断 value 是否能转成有限浮点数。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(result)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转 float。

    NaN / Inf / 非数值都会返回 default。
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(result):
        return float(default)

    return result


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的配置读取。

    支持：
        cfg.get(key, default)
        getattr(cfg, key, default)
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)
