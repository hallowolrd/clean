from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aggregation.base import Aggregator
from fl.types import AggregationResult, ClientUpdate
from models.param_groups import get_expert_id_from_name
from utils.state_dict_ops import check_finite_state_dict, clone_state_dict


@dataclass
class _HistoryState:
    """
    单个 client-expert 的历史滤波状态。

    注意：
    当前项目的 expert 参数分组粒度是 expert_id，
    所以第一版状态 key 使用 (client_id, expert_id)。
    如果后续要做多层 expert 独立滤波，再扩展成
    (client_id, layer_id, expert_id)。
    """

    mu: float
    P: float
    age: int


class FisherHistoryWolfExpertAggregator(Aggregator):
    """
    Fisher evidence + History-WoLF 专家聚合器。

    这个聚合器只负责 expert 参数，不负责 backbone / router 等 non_expert 参数。

    核心流程：
    1. 客户端上传 expert K-FAC evidence：active_count / mean_A / mean_B。
    2. 服务端计算：
           score_i,e = active_count_i,e * fisher_strength_i,e
           h_i,e     = log(score_i,e + eps)
    3. 每个 client-expert 维护历史状态：mu / P / age。
    4. 前 global_warmup_rounds 轮：
           expert 最终聚合权重使用 fisher_only，即 softmax(h)。
       但后台滤波器仍然照常更新 mu / P / age。
    5. warmup 后：
           expert 最终聚合权重统一使用 filtered mu，即 softmax(mu+ / tau)。
    6. 历史不足时只影响后台更新模式：使用普通 Kalman fallback；
       不再把最终权重 fallback 回 fisher_only。

    重要约定：
    - active_count < active_count_min：该 client-expert 本轮无效，不更新状态。
    - len(valid_records) < min_valid_clients：该 expert 参数本轮 keep_global，
      但是 valid_records 中的状态仍然会更新，用于积累历史。
    - NaN / Inf score：认为 evidence 统计异常，直接跳过该 client-expert。
    - score=0：认为是合法的低 evidence，保留并得到 h=log(eps)。
    """

    def __init__(self, cfg: Any, param_group_name: str) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)
        self.history_state: Dict[Tuple[int, int], _HistoryState] = {}

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "fisher_history_wolf"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        返回每个客户端的总 Fisher score。

        说明：
        fisher_history_wolf 的真实聚合权重是 expert-wise 权重，
        这个函数不参与真正聚合，只用于满足 Aggregator 抽象接口，
        以及外部调试时快速查看客户端总体 evidence 强度。
        """
        self._validate_client_updates(client_updates)

        weights: Dict[int, float] = {}
        for update in client_updates:
            total_score = 0.0
            expert_payloads = _get_expert_payloads(update)
            for payload in expert_payloads.values():
                if not isinstance(payload, Mapping):
                    continue

                active_count = _safe_int(payload.get("active_count", 0), default=0)
                if active_count < self.active_count_min:
                    continue

                # NaN / Inf score 表示 evidence 统计异常，诊断权重里也直接跳过；
                # score=0 是合法低 evidence，保留但不会贡献总 score。
                raw_score = _extract_raw_score(payload)
                if not math.isfinite(raw_score):
                    continue

                total_score += max(float(raw_score), 0.0)

            weights[int(update.client_id)] = float(total_score)

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
        执行 Fisher evidence + History-WoLF expert-wise delta 聚合。

        参数：
            global_state:
                本轮聚合前的全局模型参数。
            client_updates:
                本轮客户端更新。每个 update.extra 必须包含 extra["expert_kfac"]。
            param_names:
                当前聚合器负责的 expert 参数名。
            base_state:
                聚合结果写入的基础 state_dict。
                极致解耦流程中通常是 non_expert 聚合后的 state_dict。
            strict:
                True 时缺少必要字段直接报错；False 时尽量跳过缺失项。
        """
        if self.param_group_name != "expert":
            raise ValueError(
                "FisherHistoryWolfExpertAggregator 只能用于 expert 参数组，"
                f"当前 param_group_name={self.param_group_name}"
            )

        self._validate_client_updates(client_updates)
        round_id = _infer_round_id(client_updates)

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
        expert_fisher_weight_map: Dict[int, Dict[int, float]] = {}
        expert_filtered_weight_map: Dict[int, Dict[int, float]] = {}
        expert_record_map: Dict[int, List[Dict[str, Any]]] = {}
        expert_diag_map: Dict[int, Dict[str, Any]] = {}
        expert_keep_global_map: Dict[int, bool] = {}

        for expert_id, expert_names in sorted(expert_param_names.items()):
            records = self._build_expert_records(
                expert_id=expert_id,
                client_updates=client_updates,
                strict=strict,
            )

            update_diag = self._update_history_states(
                expert_id=expert_id,
                records=records,
            )

            fisher_weights = _softmax_records(
                records=records,
                value_key="h",
                temperature=1.0,
            )
            filtered_weights = _softmax_records(
                records=records,
                value_key="mu_plus",
                temperature=self.expert_weight_tau,
            )

            if self._is_global_warmup(round_id):
                final_weights = fisher_weights
                weight_source = "fisher_only"
            else:
                final_weights = filtered_weights
                weight_source = "filtered_mu"

            keep_global = len(records) < self.min_valid_clients
            keep_global_reason = None
            if len(records) == 0:
                keep_global_reason = "no_valid_clients"
            elif keep_global:
                keep_global_reason = "valid_clients_lt_min_valid_clients"

            expert_weight_map[int(expert_id)] = final_weights
            expert_fisher_weight_map[int(expert_id)] = fisher_weights
            expert_filtered_weight_map[int(expert_id)] = filtered_weights
            expert_record_map[int(expert_id)] = records
            expert_keep_global_map[int(expert_id)] = bool(keep_global)

            if not keep_global:
                self._apply_expert_weighted_delta(
                    global_state=global_state,
                    new_state_dict=new_state_dict,
                    client_updates=client_updates,
                    expert_names=expert_names,
                    weights=final_weights,
                    strict=strict,
                )

            expert_diag_map[int(expert_id)] = self._build_single_expert_diagnostics(
                expert_id=expert_id,
                client_updates=client_updates,
                records=records,
                final_weights=final_weights,
                fisher_weights=fisher_weights,
                filtered_weights=filtered_weights,
                update_diag=update_diag,
                keep_global=keep_global,
                keep_global_reason=keep_global_reason,
                weight_source=weight_source,
            )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=names,
        )

        avg_weights = _average_expert_weights(
            client_updates=client_updates,
            expert_weight_map=expert_weight_map,
            expert_keep_global_map=expert_keep_global_map,
        )
        diagnostics = self._build_history_diagnostics(
            round_id=round_id,
            client_updates=client_updates,
            param_names=names,
            expert_diag_map=expert_diag_map,
            expert_weight_map=expert_weight_map,
            expert_fisher_weight_map=expert_fisher_weight_map,
            expert_filtered_weight_map=expert_filtered_weight_map,
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
    def history_filter_cfg(self) -> Any:
        """读取 history_filter 配置块。"""
        return _cfg_get(self.cfg, "history_filter", {})

    @property
    def active_count_min(self) -> int:
        """
        evidence 有效阈值。

        优先读取 history_filter.active_count_min；
        如果没配，则兼容旧配置 expert_fisher.min_active_count。
        """
        default = int(_cfg_get(self.expert_fisher_cfg, "min_active_count", 1))
        return int(_cfg_get(self.history_filter_cfg, "active_count_min", default))

    @property
    def min_valid_clients(self) -> int:
        """
        每个 expert 允许更新参数所需的最少有效客户端数。

        注意：
        这个参数只控制 expert 参数是否更新，
        不控制后台 mu / P / age 是否更新。
        """
        return int(_cfg_get(self.history_filter_cfg, "min_valid_clients", 2))

    @property
    def global_warmup_rounds(self) -> int:
        """前多少轮最终 expert 权重使用 fisher_only。"""
        return int(_cfg_get(self.history_filter_cfg, "global_warmup_rounds", 10))

    @property
    def history_warmup_age(self) -> int:
        """client-expert 至少被有效观测多少次，才算成熟历史。"""
        return int(_cfg_get(self.history_filter_cfg, "history_warmup_age", 2))

    @property
    def min_history_clients(self) -> int:
        """使用完整 History-WoLF 更新所需的成熟历史客户端数。"""
        return int(_cfg_get(self.history_filter_cfg, "min_history_clients", 3))

    @property
    def observation_R(self) -> float:
        """观测噪声倍率。"""
        return float(_cfg_get(self.history_filter_cfg, "observation_R", 1.0))

    @property
    def mad_floor(self) -> float:
        """log-score 空间 MAD 下限。"""
        return float(_cfg_get(self.history_filter_cfg, "mad_floor", 0.1))

    @property
    def init_P(self) -> float:
        """冷启动时 P = init_P * R。"""
        return float(_cfg_get(self.history_filter_cfg, "init_P", 1.0))

    @property
    def process_noise_Q(self) -> float:
        """预测阶段过程噪声倍率：P- = P_old + Q * R。"""
        return float(_cfg_get(self.history_filter_cfg, "process_noise_Q", 0.05))

    @property
    def robust_c(self) -> float:
        """WoLF-IMQ 软阈值 c。"""
        return float(_cfg_get(self.history_filter_cfg, "robust_c", 2.0))

    @property
    def expert_weight_tau(self) -> float:
        """post-warmup 阶段 softmax(mu / tau) 的温度。"""
        tau = float(_cfg_get(self.history_filter_cfg, "expert_weight_tau", 1.0))
        return max(tau, self.eps)

    @property
    def w2_floor(self) -> float:
        """WoLF 权重 W2 的下限，防止 R_eff 爆炸到无穷。"""
        return float(_cfg_get(self.history_filter_cfg, "w2_floor", 1.0e-6))

    @property
    def P_floor(self) -> float:
        """P 的下限，防止滤波器完全冻结。"""
        return float(_cfg_get(self.history_filter_cfg, "P_floor", 1.0e-8))

    @property
    def eps(self) -> float:
        """数值稳定项。"""
        return float(_cfg_get(self.history_filter_cfg, "eps", 1.0e-12))

    @property
    def diagnostics_enabled(self) -> bool:
        """是否生成详细 history_filter 诊断字段。"""
        return bool(_cfg_get(self.history_filter_cfg, "diagnostics_enabled", True))

    @property
    def diagnostics_include_records(self) -> bool:
        """是否在 diagnostics 中保存完整 client-expert records。"""
        return bool(
            _cfg_get(
                self.history_filter_cfg,
                "diagnostics_include_records",
                False,
            )
        )

    def _is_global_warmup(self, round_id: int) -> bool:
        """判断当前轮最终权重是否仍处于 fisher_only warmup 阶段。"""
        return int(round_id) <= int(self.global_warmup_rounds)

    def _build_expert_records(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
        strict: bool,
    ) -> List[Dict[str, Any]]:
        """
        为单个 expert 收集本轮有效 client-expert records。

        有效性由 active_count_min、raw_score 是否有限、h 是否有限决定。
        NaN / Inf score 会直接跳过；score=0 会保留。
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
            if active_count < self.active_count_min:
                continue

            mean_A = max(_safe_float(payload.get("mean_A", 0.0), default=0.0), 0.0)
            mean_B = max(_safe_float(payload.get("mean_B", 0.0), default=0.0), 0.0)
            fisher_strength = _extract_nonnegative_fisher_strength(
                payload=payload,
                mean_A=mean_A,
                mean_B=mean_B,
            )

            # NaN / Inf score 通常表示 evidence 统计异常，直接跳过；
            # score=0 是合法的低 evidence，保留并得到 h=log(eps)。
            raw_score = _extract_raw_score(payload)
            if not math.isfinite(raw_score):
                continue

            score = max(float(raw_score), 0.0)
            h = math.log(score + self.eps)

            if not math.isfinite(h):
                continue

            client_id = int(update.client_id)
            old_state = self.history_state.get((client_id, int(expert_id)))
            old_mu = float(old_state.mu) if old_state is not None else 0.0
            old_P = float(old_state.P) if old_state is not None else self.P_floor
            old_age = int(old_state.age) if old_state is not None else 0

            records.append(
                {
                    "client_id": client_id,
                    "active_count": int(active_count),
                    "mean_A": float(mean_A),
                    "mean_B": float(mean_B),
                    "fisher_strength": float(fisher_strength),
                    "score": float(score),
                    "h": float(h),
                    "old_mu": float(old_mu),
                    "old_P": float(old_P),
                    "old_age": int(old_age),
                }
            )

        return records

    def _update_history_states(
        self,
        expert_id: int,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        更新单个 expert 下所有有效 client-expert 的 mu / P / age。

        返回 update_diag 仅用于诊断，不参与参数聚合。
        """
        if len(records) == 0:
            return {
                "update_mode": "no_valid_clients",
                "history_sufficient": False,
                "delta": 0.0,
                "mad_h": 0.0,
                "mad_r": 0.0,
                "R": 0.0,
                "num_history_clients": 0,
                "num_cold_start_clients": 0,
            }

        history_records = [
            record
            for record in records
            if int(record.get("old_age", 0)) >= self.history_warmup_age
        ]
        history_sufficient = len(history_records) >= self.min_history_clients

        mad_h = 0.0
        mad_r = 0.0
        delta = 0.0

        if history_sufficient:
            drift_values = [
                float(record["h"]) - float(record["old_mu"])
                for record in history_records
            ]
            delta = _median_even_mean(drift_values)

            residual_values = [
                float(record["h"]) - (float(record["old_mu"]) + delta)
                for record in history_records
            ]
            med_r = _median_even_mean(residual_values)
            mad_r = _median_abs_deviation(residual_values, center=med_r)
            scale = max(float(mad_r), self.mad_floor)
            R = self.observation_R * (scale**2)
            update_mode = "history_wolf"
        else:
            h_values = [float(record["h"]) for record in records]
            if len(h_values) >= 2:
                med_h = _median_even_mean(h_values)
                mad_h = _median_abs_deviation(h_values, center=med_h)
                scale = max(float(mad_h), self.mad_floor)
            else:
                scale = self.mad_floor
            R = self.observation_R * (scale**2)
            update_mode = "kalman_fallback"

        R = max(float(R), self.eps)

        K_values: List[float] = []
        W2_values: List[float] = []
        R_eff_values: List[float] = []
        residual_values_all: List[float] = []
        mu_update_abs_values: List[float] = []
        P_values: List[float] = []
        mu_values: List[float] = []
        cold_start_count = 0

        for record in records:
            client_id = int(record["client_id"])
            h = float(record["h"])
            old_mu = float(record["old_mu"])
            old_P = max(float(record["old_P"]), self.P_floor)
            old_age = int(record["old_age"])

            if old_age <= 0:
                # 冷启动：第一条有效观测直接作为初始 mu。
                mu_minus = h
                P_minus = max(self.init_P * R, self.P_floor)
                residual = 0.0
                W2 = 1.0
                R_eff = R
                K = 1.0
                mu_plus = h
                P_plus = P_minus
                cold_start_count += 1
            else:
                if history_sufficient:
                    mu_minus = old_mu + delta
                else:
                    mu_minus = old_mu

                P_minus = max(old_P + self.process_noise_Q * R, self.P_floor)
                residual = h - mu_minus

                if history_sufficient:
                    d2 = (residual**2) / (P_minus + R + self.eps)
                    c2 = max(self.robust_c**2, self.eps)
                    W2 = c2 / (c2 + d2 + self.eps)
                    W2 = min(1.0, max(float(W2), self.w2_floor))
                else:
                    W2 = 1.0

                R_eff = R / (W2 + self.eps)
                K = P_minus / (P_minus + R_eff + self.eps)
                K = min(1.0, max(0.0, float(K)))
                mu_plus = mu_minus + K * residual
                P_plus = (1.0 - K) * P_minus
                P_plus = max(float(P_plus), self.P_floor)

            new_age = old_age + 1
            self.history_state[(client_id, int(expert_id))] = _HistoryState(
                mu=float(mu_plus),
                P=float(P_plus),
                age=int(new_age),
            )

            record["mu_minus"] = float(mu_minus)
            record["P_minus"] = float(P_minus)
            record["residual"] = float(residual)
            record["W2"] = float(W2)
            record["R_eff"] = float(R_eff)
            record["K"] = float(K)
            record["mu_plus"] = float(mu_plus)
            record["P_plus"] = float(P_plus)
            record["age_plus"] = int(new_age)
            record["mu_update_abs"] = float(abs(mu_plus - old_mu))

            K_values.append(float(K))
            W2_values.append(float(W2))
            R_eff_values.append(float(R_eff))
            residual_values_all.append(float(residual))
            mu_update_abs_values.append(float(abs(mu_plus - old_mu)))
            P_values.append(float(P_plus))
            mu_values.append(float(mu_plus))

        return {
            "update_mode": update_mode,
            "history_sufficient": bool(history_sufficient),
            "delta": float(delta),
            "mad_h": float(mad_h),
            "mad_r": float(mad_r),
            "R": float(R),
            "num_history_clients": int(len(history_records)),
            "num_cold_start_clients": int(cold_start_count),
            "K_stats": _stat_dict(K_values),
            "W2_stats": _stat_dict(W2_values),
            "R_eff_stats": _stat_dict(R_eff_values),
            "residual_abs_stats": _stat_dict([abs(v) for v in residual_values_all]),
            "mu_update_abs_stats": _stat_dict(mu_update_abs_values),
            "P_stats": _stat_dict(P_values),
            "mu_stats": _stat_dict(mu_values),
        }

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
        update_by_client_id = {int(update.client_id): update for update in client_updates}

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

    def _build_single_expert_diagnostics(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
        records: Sequence[Mapping[str, Any]],
        final_weights: Mapping[int, float],
        fisher_weights: Mapping[int, float],
        filtered_weights: Mapping[int, float],
        update_diag: Mapping[str, Any],
        keep_global: bool,
        keep_global_reason: Optional[str],
        weight_source: str,
    ) -> Dict[str, Any]:
        """构建单个 expert 的诊断信息。"""
        status_counts = self._count_expert_payload_status(
            expert_id=expert_id,
            client_updates=client_updates,
        )

        scores = [float(record.get("score", 0.0)) for record in records]
        hs = [float(record.get("h", 0.0)) for record in records]
        mus = [float(record.get("mu_plus", 0.0)) for record in records]
        active_counts = [float(record.get("active_count", 0.0)) for record in records]
        fisher_strengths = [
            float(record.get("fisher_strength", 0.0)) for record in records
        ]
        final_weight_values = [
            float(final_weights.get(int(record.get("client_id", -1)), 0.0))
            for record in records
        ]

        top_client = None
        if len(final_weights) > 0:
            top_client = max(final_weights.items(), key=lambda item: item[1])[0]

        top1_weight, top2_weight, top1_gap = _top_weight_stats(final_weights)
        fisher_filtered_l1 = _weight_l1(fisher_weights, filtered_weights)

        diag: Dict[str, Any] = {
            "expert_id": int(expert_id),
            "keep_global": bool(keep_global),
            "fallback": bool(keep_global),
            "fallback_reason": keep_global_reason,
            "update_mode": str(update_diag.get("update_mode", "unknown")),
            "weight_source": str(weight_source),
            "valid_clients": int(len(records)),
            "history_clients": int(update_diag.get("num_history_clients", 0)),
            "cold_start_clients": int(update_diag.get("num_cold_start_clients", 0)),
            "min_valid_clients": int(self.min_valid_clients),
            "active_count_min": int(self.active_count_min),
            "history_warmup_age": int(self.history_warmup_age),
            "min_history_clients": int(self.min_history_clients),
            "missing_payload_clients": int(status_counts["missing_payload_clients"]),
            "invalid_clients": int(status_counts["invalid_clients"]),
            "zero_active_clients": int(status_counts["zero_active_clients"]),
            "zero_score_clients": int(status_counts["zero_score_clients"]),
            "nan_score_clients": int(status_counts["nan_score_clients"]),
            "delta": float(update_diag.get("delta", 0.0)),
            "mad_h": float(update_diag.get("mad_h", 0.0)),
            "mad_r": float(update_diag.get("mad_r", 0.0)),
            "R": float(update_diag.get("R", 0.0)),
            "K_mean": _nested_stat_mean(update_diag, "K_stats"),
            "K_min": _nested_stat_min(update_diag, "K_stats"),
            "K_max": _nested_stat_max(update_diag, "K_stats"),
            "W2_mean": _nested_stat_mean(update_diag, "W2_stats"),
            "W2_min": _nested_stat_min(update_diag, "W2_stats"),
            "W2_max": _nested_stat_max(update_diag, "W2_stats"),
            "R_eff_mean": _nested_stat_mean(update_diag, "R_eff_stats"),
            "P_mean": _nested_stat_mean(update_diag, "P_stats"),
            "P_min": _nested_stat_min(update_diag, "P_stats"),
            "P_max": _nested_stat_max(update_diag, "P_stats"),
            "mu_mean": _nested_stat_mean(update_diag, "mu_stats"),
            "abs_residual_mean": _nested_stat_mean(update_diag, "residual_abs_stats"),
            "abs_mu_update_mean": _nested_stat_mean(
                update_diag,
                "mu_update_abs_stats",
            ),
            "score_stats": _stat_dict(scores),
            "h_stats": _stat_dict(hs),
            "mu_plus_stats": _stat_dict(mus),
            "active_count_stats": _stat_dict(active_counts),
            "fisher_strength_stats": _stat_dict(fisher_strengths),
            "score_cv": _coefficient_of_variation(scores),
            "active_count_cv": _coefficient_of_variation(active_counts),
            "fisher_strength_cv": _coefficient_of_variation(fisher_strengths),
            "final_weight_entropy": _weight_entropy(final_weights),
            "final_weight_entropy_norm": _weight_entropy_norm(final_weights),
            "fisher_weight_entropy_norm": _weight_entropy_norm(fisher_weights),
            "filtered_weight_entropy_norm": _weight_entropy_norm(filtered_weights),
            "effective_clients": _effective_clients(final_weights),
            "weight_min": min(final_weights.values()) if len(final_weights) > 0 else 0.0,
            "weight_max": max(final_weights.values()) if len(final_weights) > 0 else 0.0,
            "top_client": int(top_client) if top_client is not None else None,
            "top1_weight": float(top1_weight),
            "top2_weight": float(top2_weight),
            "top1_gap": float(top1_gap),
            "fisher_filtered_l1": float(fisher_filtered_l1),
            "weight_active_corr": _pearson_corr(final_weight_values, active_counts),
            "weight_fisher_corr": _pearson_corr(
                final_weight_values,
                fisher_strengths,
            ),
        }

        if self.diagnostics_include_records:
            diag["final_weights"] = {
                int(client_id): float(weight)
                for client_id, weight in final_weights.items()
            }
            diag["fisher_weights"] = {
                int(client_id): float(weight)
                for client_id, weight in fisher_weights.items()
            }
            diag["filtered_weights"] = {
                int(client_id): float(weight)
                for client_id, weight in filtered_weights.items()
            }
            diag["records"] = [
                {
                    "client_id": int(record.get("client_id", -1)),
                    "active_count": int(record.get("active_count", 0)),
                    "mean_A": float(record.get("mean_A", 0.0)),
                    "mean_B": float(record.get("mean_B", 0.0)),
                    "fisher_strength": float(record.get("fisher_strength", 0.0)),
                    "score": float(record.get("score", 0.0)),
                    "h": float(record.get("h", 0.0)),
                    "old_mu": float(record.get("old_mu", 0.0)),
                    "old_P": float(record.get("old_P", 0.0)),
                    "old_age": int(record.get("old_age", 0)),
                    "mu_minus": float(record.get("mu_minus", 0.0)),
                    "P_minus": float(record.get("P_minus", 0.0)),
                    "residual": float(record.get("residual", 0.0)),
                    "W2": float(record.get("W2", 1.0)),
                    "R_eff": float(record.get("R_eff", 0.0)),
                    "K": float(record.get("K", 0.0)),
                    "mu_plus": float(record.get("mu_plus", 0.0)),
                    "P_plus": float(record.get("P_plus", 0.0)),
                    "age_plus": int(record.get("age_plus", 0)),
                    "final_weight": float(
                        final_weights.get(int(record.get("client_id", -1)), 0.0)
                    ),
                    "fisher_weight": float(
                        fisher_weights.get(int(record.get("client_id", -1)), 0.0)
                    ),
                    "filtered_weight": float(
                        filtered_weights.get(int(record.get("client_id", -1)), 0.0)
                    ),
                }
                for record in records
            ]

        return diag

    def _build_history_diagnostics(
        self,
        round_id: int,
        client_updates: Sequence[ClientUpdate],
        param_names: Sequence[str],
        expert_diag_map: Mapping[int, Mapping[str, Any]],
        expert_weight_map: Mapping[int, Mapping[int, float]],
        expert_fisher_weight_map: Mapping[int, Mapping[int, float]],
        expert_filtered_weight_map: Mapping[int, Mapping[int, float]],
        avg_weights: Mapping[int, float],
    ) -> Dict[str, Any]:
        """构建整轮 history_filter 诊断信息。"""
        num_experts = int(len(expert_diag_map))
        num_keep_global_experts = int(
            sum(1 for diag in expert_diag_map.values() if bool(diag.get("keep_global", False)))
        )

        if not self.diagnostics_enabled:
            return {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "history_diag_enabled": False,
                "round_id": int(round_id),
                "num_clients": int(len(client_updates)),
                "param_count": int(len(param_names)),
                "num_experts": num_experts,
                "num_keep_global_experts": num_keep_global_experts,
                "fallback_ratio": _safe_divide(
                    num_keep_global_experts,
                    max(num_experts, 1),
                ),
            }

        all_diags = list(expert_diag_map.values())
        updated_diags = [
            diag for diag in all_diags if not bool(diag.get("keep_global", False))
        ]

        diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "history_diag_enabled": True,
            "diagnostics_include_records": bool(self.diagnostics_include_records),
            "round_id": int(round_id),
            "num_clients": int(len(client_updates)),
            "param_count": int(len(param_names)),
            "num_experts": num_experts,
            "global_warmup_rounds": int(self.global_warmup_rounds),
            "is_global_warmup": bool(self._is_global_warmup(round_id)),
            "active_count_min": int(self.active_count_min),
            "min_valid_clients": int(self.min_valid_clients),
            "history_warmup_age": int(self.history_warmup_age),
            "min_history_clients": int(self.min_history_clients),
            "num_keep_global_experts": num_keep_global_experts,
            "fallback_ratio": _safe_divide(num_keep_global_experts, max(num_experts, 1)),
            "keep_global_experts": [
                int(expert_id)
                for expert_id, diag in sorted(expert_diag_map.items())
                if bool(diag.get("keep_global", False))
            ],
            "mean_valid_clients": _mean_clean(
                [diag.get("valid_clients", 0.0) for diag in all_diags]
            ),
            "mean_history_clients": _mean_clean(
                [diag.get("history_clients", 0.0) for diag in all_diags]
            ),
            "mean_cold_start_clients": _mean_clean(
                [diag.get("cold_start_clients", 0.0) for diag in all_diags]
            ),
            "mean_R": _mean_clean([diag.get("R", 0.0) for diag in all_diags]),
            "mean_K": _mean_clean([diag.get("K_mean", 0.0) for diag in all_diags]),
            "mean_W2": _mean_clean([diag.get("W2_mean", 0.0) for diag in all_diags]),
            "mean_abs_residual": _mean_clean(
                [diag.get("abs_residual_mean", 0.0) for diag in all_diags]
            ),
            "mean_abs_mu_update": _mean_clean(
                [diag.get("abs_mu_update_mean", 0.0) for diag in all_diags]
            ),
            "mean_fisher_filtered_l1": _mean_clean(
                [diag.get("fisher_filtered_l1", 0.0) for diag in all_diags]
            ),
            "mean_final_weight_entropy_norm": _mean_clean(
                [diag.get("final_weight_entropy_norm", 0.0) for diag in updated_diags]
            ),
            "mean_fisher_weight_entropy_norm": _mean_clean(
                [diag.get("fisher_weight_entropy_norm", 0.0) for diag in all_diags]
            ),
            "mean_filtered_weight_entropy_norm": _mean_clean(
                [diag.get("filtered_weight_entropy_norm", 0.0) for diag in all_diags]
            ),
            "mean_effective_clients": _mean_clean(
                [diag.get("effective_clients", 0.0) for diag in updated_diags]
            ),
            "mean_weight_max": _mean_clean(
                [diag.get("weight_max", 0.0) for diag in updated_diags]
            ),
            "mean_weight_active_corr": _mean_clean(
                [diag.get("weight_active_corr", 0.0) for diag in updated_diags]
            ),
            "mean_weight_fisher_corr": _mean_clean(
                [diag.get("weight_fisher_corr", 0.0) for diag in updated_diags]
            ),
            "update_mode_counts": _count_by_key(all_diags, "update_mode"),
            "weight_source_counts": _count_by_key(all_diags, "weight_source"),
            "expert_diagnostics": {
                int(expert_id): dict(diag)
                for expert_id, diag in sorted(expert_diag_map.items())
            },
        }

        if self.diagnostics_include_records:
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
            diagnostics["expert_fisher_weights"] = {
                int(expert_id): {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                for expert_id, weights in expert_fisher_weight_map.items()
            }
            diagnostics["expert_filtered_weights"] = {
                int(expert_id): {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                for expert_id, weights in expert_filtered_weight_map.items()
            }

        return diagnostics

    def _count_expert_payload_status(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[str, int]:
        """
        统计单个 expert 的原始 payload 状态。

        注意：
        - score=0 不算 invalid，因为新算法允许 h=log(eps)。
        - NaN / Inf score 算 invalid，因为 evidence 统计异常，
          会在构建 records 时直接跳过。
        """
        missing_payload_clients = 0
        zero_score_clients = 0
        zero_active_clients = 0
        nan_score_clients = 0
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
            raw_score = _extract_raw_score(payload)

            # NaN / Inf score 表示 evidence 统计异常，会在 _build_expert_records() 中直接跳过。
            # 这类样本只计入 nan_score_clients / invalid_clients，不再计入 zero_score_clients。
            is_nan_score = not math.isfinite(raw_score)

            if is_nan_score:
                score = 0.0
                is_zero_score = False
            else:
                # score=0 是合法低 evidence，会保留并得到 h=log(eps)。
                # 有限负数 score 压成 0，也按 zero_score 统计。
                score = max(float(raw_score), 0.0)
                is_zero_score = score <= 0.0

            is_zero_active = active_count <= 0

            if is_nan_score:
                nan_score_clients += 1
            if is_zero_score:
                zero_score_clients += 1
            if is_zero_active:
                zero_active_clients += 1
            if active_count < self.active_count_min or is_nan_score:
                invalid_clients += 1

        return {
            "missing_payload_clients": int(missing_payload_clients),
            "zero_score_clients": int(zero_score_clients),
            "zero_active_clients": int(zero_active_clients),
            "nan_score_clients": int(nan_score_clients),
            "invalid_clients": int(invalid_clients),
        }


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """解析当前聚合器需要处理的参数名。"""
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
            # fisher_history_wolf 只处理 expert 参数。
            # 如果 server 误传了 non_expert 参数，这里直接跳过。
            continue
        result.setdefault(int(expert_id), []).append(name)

    return {int(expert_id): names for expert_id, names in sorted(result.items())}


def _get_expert_payloads(update: ClientUpdate) -> Mapping[Any, Any]:
    """
    从 ClientUpdate.extra 中读取 expert_kfac payload。

    支持两种格式：
    1. extra["expert_kfac"]["experts"]
    2. extra["expert_kfac"]
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
    """读取单个客户端、单个 expert 的 K-FAC payload。"""
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


def _extract_raw_score(payload: Mapping[str, Any]) -> float:
    """
    提取原始 score，用于判断 NaN / Inf。

    优先级：
    1. payload["score"]
    2. active_count * fisher_strength
    3. active_count * mean_A * mean_B
    """
    if "score" in payload:
        return _safe_float(payload.get("score", 0.0), default=float("nan"))

    active_count = _safe_float(payload.get("active_count", 0.0), default=0.0)
    if "fisher_strength" in payload:
        fisher_strength = _safe_float(
            payload.get("fisher_strength", 0.0),
            default=float("nan"),
        )
        return float(active_count * fisher_strength)

    mean_A = _safe_float(payload.get("mean_A", 0.0), default=float("nan"))
    mean_B = _safe_float(payload.get("mean_B", 0.0), default=float("nan"))
    return float(active_count * mean_A * mean_B)


def _extract_nonnegative_score(payload: Mapping[str, Any]) -> float:
    """提取非负 score，NaN / Inf / 负数统一压成 0。"""
    score = _extract_raw_score(payload)
    if not math.isfinite(score):
        return 0.0
    return max(float(score), 0.0)


def _extract_nonnegative_fisher_strength(
    payload: Mapping[str, Any],
    mean_A: float,
    mean_B: float,
) -> float:
    """提取非负 fisher_strength。"""
    if "fisher_strength" in payload:
        value = _safe_float(payload.get("fisher_strength", 0.0), default=0.0)
        return max(float(value), 0.0)
    return max(float(mean_A), 0.0) * max(float(mean_B), 0.0)


def _infer_round_id(client_updates: Sequence[ClientUpdate]) -> int:
    """从 client_updates 中推断本轮 round_id，并检查所有客户端一致。"""
    round_ids = {int(update.round_id) for update in client_updates}
    if len(round_ids) != 1:
        raise ValueError(f"本轮 client_updates 的 round_id 不一致：{sorted(round_ids)}")
    return int(next(iter(round_ids)))


def _softmax_records(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
    temperature: float = 1.0,
) -> Dict[int, float]:
    """对单个 expert 的 records 按指定字段做 softmax。"""
    if len(records) == 0:
        return {}

    tau = max(float(temperature), 1.0e-12)
    values = torch.tensor(
        [float(record[value_key]) / tau for record in records],
        dtype=torch.float64,
    )
    weights = torch.softmax(values, dim=0).tolist()

    return {
        int(record["client_id"]): float(weight)
        for record, weight in zip(records, weights)
    }


def _average_expert_weights(
    client_updates: Sequence[ClientUpdate],
    expert_weight_map: Mapping[int, Mapping[int, float]],
    expert_keep_global_map: Mapping[int, bool],
) -> Dict[int, float]:
    """
    把 expert-wise 权重压成一套 client-wise 平均权重，仅用于诊断。

    真实聚合使用的是 expert_weight_map。
    """
    client_ids = [int(update.client_id) for update in client_updates]
    avg_weights = {client_id: 0.0 for client_id in client_ids}

    num_updated_experts = 0
    for expert_id, weights in expert_weight_map.items():
        if bool(expert_keep_global_map.get(expert_id, False)):
            continue
        if len(weights) == 0:
            continue

        num_updated_experts += 1
        for client_id in client_ids:
            avg_weights[client_id] += float(weights.get(client_id, 0.0))

    if num_updated_experts <= 0:
        return avg_weights

    for client_id in avg_weights:
        avg_weights[client_id] /= float(num_updated_experts)

    return avg_weights


def _median_even_mean(values: Sequence[Any]) -> float:
    """
    计算中位数。

    偶数个元素时取中间两个数的平均，
    避免 torch.median 在偶数长度下偏向 lower median。
    """
    clean_values = sorted(float(value) for value in values if _is_finite_number(value))
    if len(clean_values) == 0:
        return 0.0

    n = len(clean_values)
    mid = n // 2
    if n % 2 == 1:
        return float(clean_values[mid])

    return float(0.5 * (clean_values[mid - 1] + clean_values[mid]))


def _median_abs_deviation(
    values: Sequence[Any],
    center: Optional[float] = None,
) -> float:
    """计算 MAD = median(|x - center|)。"""
    clean_values = [float(value) for value in values if _is_finite_number(value)]
    if len(clean_values) == 0:
        return 0.0

    if center is None:
        center = _median_even_mean(clean_values)

    deviations = [abs(value - float(center)) for value in clean_values]
    return _median_even_mean(deviations)


def _weight_l1(
    weights_a: Mapping[int, float],
    weights_b: Mapping[int, float],
) -> float:
    """计算两套权重的 L1 距离。"""
    client_ids = set(int(key) for key in weights_a.keys()) | set(
        int(key) for key in weights_b.keys()
    )
    return float(
        sum(
            abs(float(weights_a.get(client_id, 0.0)) - float(weights_b.get(client_id, 0.0)))
            for client_id in client_ids
        )
    )


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
    """计算归一化权重熵。"""
    if len(weights) <= 1:
        return 0.0

    entropy = _weight_entropy(weights)
    max_entropy = math.log(float(len(weights)) + 1.0e-12)
    return _safe_divide(entropy, max_entropy)


def _effective_clients(weights: Mapping[int, float]) -> float:
    """计算有效客户端数：1 / sum_i w_i^2。"""
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
    clean_values = [float(value) for value in values if _is_finite_number(value)]
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
    clean_values = [float(value) for value in values if _is_finite_number(value)]
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
    clean_values = [float(value) for value in values if _is_finite_number(value)]
    if len(clean_values) == 0:
        return 0.0
    return float(sum(clean_values) / len(clean_values))


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


def _nested_stat_mean(mapping: Mapping[str, Any], key: str) -> float:
    """从嵌套 stat_dict 中安全读取 mean。"""
    stat = mapping.get(key, {})
    if not isinstance(stat, Mapping):
        return 0.0
    return _safe_float(stat.get("mean", 0.0), default=0.0)


def _nested_stat_min(mapping: Mapping[str, Any], key: str) -> float:
    """从嵌套 stat_dict 中安全读取 min。"""
    stat = mapping.get(key, {})
    if not isinstance(stat, Mapping):
        return 0.0
    return _safe_float(stat.get("min", 0.0), default=0.0)


def _nested_stat_max(mapping: Mapping[str, Any], key: str) -> float:
    """从嵌套 stat_dict 中安全读取 max。"""
    stat = mapping.get(key, {})
    if not isinstance(stat, Mapping):
        return 0.0
    return _safe_float(stat.get("max", 0.0), default=0.0)


def _count_by_key(
    items: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, int]:
    """统计 diagnostics 中某个字段的取值次数。"""
    result: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        result[value] = result.get(value, 0) + 1
    return result


def _is_finite_number(value: Any) -> bool:
    """判断 value 是否能转成有限浮点数。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(result)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转 float。NaN / Inf / 非数值都会返回 default。"""
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
    - cfg.get(key, default)
    - getattr(cfg, key, default)
    """
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)