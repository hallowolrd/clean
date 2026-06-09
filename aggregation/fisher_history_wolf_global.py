from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from aggregation.base import Aggregator
from fl.types import ClientUpdate


class FisherHistoryWolfGlobalAggregator(Aggregator):
    """
    纯 FL 版 Fisher-History-WoLF 聚合器。

    这个聚合器用于“整模型 client-wise 聚合”，不是 MoE expert-wise 聚合。

    和原来的 FisherHistoryWolfExpertAggregator 的区别：
    - 原版 fisher_history_wolf：
        client i + expert e -> history_state_i,e
        每个 expert 单独一套客户端权重
        读取 update.extra["expert_kfac"]
        聚合 expert 参数组

    - 当前 fisher_history_wolf_global：
        client i -> history_state_i
        整个模型共用一套客户端权重
        读取 update.extra["global_fisher"]
        聚合整个纯 FL 模型参数组

    推荐配合 fl/full_model_fisher.py 使用。客户端本地训练后应写入：
        update.extra["global_fisher"] = {
            "fisher_strength": ...,
            "num_samples": ...,
            "score": ...,
            "meta": {...},
        }

    核心思想：
    1. fisher_strength 是当前客户端的整模型 Fisher evidence。
    2. num_samples 只作为 support，低样本 evidence 降权，高样本不额外奖励。
    3. 服务端为每个 client_id 维护历史状态 mu/P。
    4. 当前观测和历史预测偏差过大时，使用 WoLF-IMQ 降低本轮观测影响。
    5. 最终用 filtered_mu + log(support) 得到客户端 logit。
    6. 对所有客户端 softmax 得到整模型聚合权重。

    聚合公式仍然走 Aggregator 基类：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    def __init__(self, cfg: Any, param_group_name: str) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)

        # 纯 FL 没有 expert，所以只按 client_id 保存历史滤波状态。
        # history_states[client_id] = {"mu": ..., "P": ..., "age": ...}
        self.history_states: Dict[int, Dict[str, float]] = {}

        # compute_weights() 会更新历史状态。
        # Aggregator.aggregate() 后续还会调用 build_diagnostics()。
        # 为避免 diagnostics 重复更新历史状态，这里缓存最近一轮的 records。
        self._last_raw_records: Sequence[Dict[str, Any]] = []
        self._last_filtered_records: Sequence[Dict[str, Any]] = []
        self._last_fallback_to_uniform: bool = False
        self._last_raw_weights: Dict[int, float] = {}

    @property
    def method_name(self) -> str:
        """
        返回当前聚合方法名称。

        注意：
        - fisher_history_wolf 是原来的 expert-wise 版本。
        - fisher_history_wolf_global 是当前 pure-FL client-wise 版本。
        """
        return "fisher_history_wolf_global"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算纯 FL Fisher-History-WoLF 的客户端原始权重。

        返回的是 raw weight，后续 Aggregator.aggregate() 会统一 normalize。

        正常路径：
            1. 读取每个客户端 update.extra["global_fisher"]。
            2. 用 log(fisher_strength) 构造当前观测 h。
            3. 对 h 做 median / MAD robust normalization 得到 z。
            4. 对每个 client_id 做 WoLF-IMQ 历史滤波。
            5. logit_i = mu_new_i + log(support_i + eps)。
            6. softmax(logit) 得到 raw weight。

        fallback：
            如果有效 Fisher record 数量不足，则退化为 uniform raw weights。
        """
        self._validate_client_updates(client_updates)

        raw_records = self._build_raw_records(client_updates)

        if len(raw_records) < self.min_valid_clients:
            raw_weights = _build_uniform_raw_weights(client_updates)

            self._last_raw_records = list(raw_records)
            self._last_filtered_records = []
            self._last_fallback_to_uniform = True
            self._last_raw_weights = dict(raw_weights)

            return raw_weights

        filtered_records = self._filter_records_with_wolf(raw_records)
        raw_weights = _softmax_records(filtered_records)

        if _sum_positive(raw_weights.values()) <= self.eps:
            raw_weights = _build_uniform_raw_weights(client_updates)
            fallback_to_uniform = True
        else:
            fallback_to_uniform = False

        self._last_raw_records = list(raw_records)
        self._last_filtered_records = list(filtered_records)
        self._last_fallback_to_uniform = bool(fallback_to_uniform)
        self._last_raw_weights = dict(raw_weights)

        return raw_weights

    @property
    def full_model_fisher_cfg(self) -> Any:
        """
        读取 full_model_fisher 配置块。

        推荐配置：
            full_model_fisher:
              enabled: true
              max_batches: 10
              model_mode: eval
              eps: 1.0e-8
              min_valid_clients: 2
              missing_policy: error
        """
        return _cfg_get(self.cfg, "full_model_fisher", {})

    @property
    def wolf_cfg(self) -> Any:
        """
        读取 fisher_history_wolf_global 配置块。

        推荐配置：
            fisher_history_wolf_global:
              eps: 1.0e-8
              init_P: 1.0
              process_noise_Q: 0.05
              observation_R: 1.0
              robust_c: 2.0
              diagnostics_enabled: true
              diagnostics_include_records: false
        """
        return _cfg_get(self.cfg, "fisher_history_wolf_global", {})

    @property
    def eps(self) -> float:
        """
        数值稳定项。

        优先读取 fisher_history_wolf_global.eps；
        如果没有，则读取 full_model_fisher.eps；
        再没有则使用默认 1e-8。
        """
        return float(
            _cfg_get(
                self.wolf_cfg,
                "eps",
                _cfg_get(self.full_model_fisher_cfg, "eps", 1.0e-8),
            )
        )

    @property
    def min_valid_clients(self) -> int:
        """
        至少需要多少个有效客户端 Fisher record。

        如果低于这个数量，说明 evidence 不足，退化为 uniform 聚合。
        """
        return int(
            _cfg_get(
                self.full_model_fisher_cfg,
                "min_valid_clients",
                2,
            )
        )

    @property
    def missing_policy(self) -> str:
        """
        缺少 update.extra["global_fisher"] 时的处理策略。

        支持：
            error:
                直接报错，推荐默认值，方便发现 client.py 没接好 Fisher evidence。

            skip:
                跳过该客户端。如果有效客户端不足，会 fallback 到 uniform。

            uniform:
                直接 fallback 到 uniform。
        """
        return str(
            _cfg_get(
                self.full_model_fisher_cfg,
                "missing_policy",
                "error",
            )
        ).lower()

    @property
    def init_P(self) -> float:
        """
        新客户端历史状态的初始不确定性。
        """
        return float(_cfg_get(self.wolf_cfg, "init_P", 1.0))

    @property
    def process_noise_Q(self) -> float:
        """
        历史 Fisher evidence 状态的过程噪声。

        越大表示历史状态允许更快变化；
        越小表示历史状态更平滑。
        """
        return float(_cfg_get(self.wolf_cfg, "process_noise_Q", 0.05))

    @property
    def observation_R(self) -> float:
        """
        normalized z 的基础观测噪声。

        因为 z 是基于本轮客户端 Fisher 的 robust normalized log Fisher，
        所以 R=1 可以理解为约 1 个 MAD 单位的正常观测波动。
        """
        return float(_cfg_get(self.wolf_cfg, "observation_R", 1.0))

    @property
    def robust_c(self) -> float:
        """
        WoLF-IMQ 的软阈值。

        residual 越大，rho 越小；
        rho 越小，本轮观测对历史状态的影响越弱。
        """
        return float(_cfg_get(self.wolf_cfg, "robust_c", 2.0))

    @property
    def diagnostics_enabled(self) -> bool:
        """
        是否生成 fisher_history_wolf_global 诊断字段。
        """
        return bool(_cfg_get(self.wolf_cfg, "diagnostics_enabled", True))

    @property
    def diagnostics_include_records(self) -> bool:
        """
        是否在 diagnostics 中保存完整 records。

        注意：
        如果每轮客户端很多，打开这个选项会让 summary / jsonl 更长。
        """
        return bool(_cfg_get(self.wolf_cfg, "diagnostics_include_records", False))

    def _build_raw_records(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Sequence[Dict[str, Any]]:
        """
        从 client_updates 中提取纯 FL Fisher records。

        每个有效 raw record 包含：
            client_id
            num_samples
            fisher_strength
            score
            log_fisher
            meta

        注意：
        - fisher_history_wolf_global 的有效性主要看 fisher_strength。
        - payload["score"] 主要保留做诊断。
        - num_samples 只用于 support，不作为高样本奖励。
        """
        records = []

        for update in client_updates:
            payload = _get_global_fisher_payload(
                update=update,
                missing_policy=self.missing_policy,
            )

            if payload is None:
                if self.missing_policy == "uniform":
                    return []
                continue

            fisher_strength = _extract_fisher_strength(payload)
            num_samples = _extract_num_samples(
                update=update,
                payload=payload,
            )
            score = _extract_score(
                payload=payload,
                fisher_strength=fisher_strength,
                num_samples=num_samples,
            )

            is_valid = (
                int(num_samples) > 0
                and float(fisher_strength) > 0.0
                and math.isfinite(float(fisher_strength))
            )

            if not is_valid:
                continue

            records.append(
                {
                    "client_id": int(update.client_id),
                    "num_samples": int(num_samples),
                    "fisher_strength": float(fisher_strength),
                    "score": float(score),
                    "log_fisher": float(
                        math.log(max(float(fisher_strength), 0.0) + self.eps)
                    ),
                    "source": str(payload.get("source", "global_fisher")),
                    "meta": payload.get("meta", {}),
                }
            )

        return records

    def _filter_records_with_wolf(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> Sequence[Dict[str, Any]]:
        """
        对所有客户端 records 做 pure-FL WoLF 历史滤波。

        对每个客户端得到：
            z:
                当前轮 robust normalized log Fisher observation。

            support:
                基于 num_samples 的支撑度。
                低样本客户端降权，高样本客户端最多到 1，不继续奖励。

            rho:
                WoLF-IMQ 可靠性。
                当前观测和历史预测差异越大，rho 越小。

            kalman_gain:
                历史状态更新步长。

            mu_new / P_new:
                更新后的客户端历史 Fisher evidence 状态。

            logit:
                最终用于 softmax 的客户端 logit。
        """
        if len(records) == 0:
            return []

        eps = self.eps

        h_values = [
            math.log(max(float(record["fisher_strength"]), 0.0) + eps)
            for record in records
        ]

        h_median = _median(h_values)
        h_mad = _median(
            [
                abs(float(value) - float(h_median))
                for value in h_values
            ]
        )

        sample_values = [
            float(record["num_samples"])
            for record in records
            if float(record["num_samples"]) > 0.0
        ]

        sample_median = _median(sample_values) if len(sample_values) > 0 else 1.0
        sample_median = max(float(sample_median), eps)

        filtered_records = []

        for record, h_value in zip(records, h_values):
            client_id = int(record["client_id"])

            # 如果本轮客户端之间 Fisher 几乎没有差异，
            # 就不制造虚假的极端 z。
            if h_mad <= eps:
                z = 0.0
            else:
                z = (float(h_value) - float(h_median)) / (float(h_mad) + eps)

            num_samples = float(record["num_samples"])

            # support 只用于低样本 evidence 降权。
            # 高样本客户端最多 support=1，不额外奖励。
            support = min(1.0, num_samples / (sample_median + eps))
            support = max(0.0, float(support))

            old_state = self.history_states.get(client_id)

            if old_state is None:
                # 冷启动：
                # 第一次有效观测直接初始化历史状态，不做异常观测惩罚。
                # 这样避免前几轮 Fisher 快速变化时被误伤。
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

                mu_old = _safe_float(
                    old_state.get("mu", 0.0),
                    default=0.0,
                )
                P_old = max(
                    _safe_float(
                        old_state.get("P", self.init_P),
                        default=self.init_P,
                    ),
                    eps,
                )
                age_old = _safe_float(
                    old_state.get("age", 0.0),
                    default=0.0,
                )

                mu_pred = float(mu_old)
                P_pred = float(P_old + self.process_noise_Q)

                residual = float(z - mu_pred)
                denom = float(P_pred + self.observation_R + eps)
                d2 = float((residual * residual) / denom)

                robust_c = max(float(self.robust_c), eps)
                rho = float((1.0 + d2 / (robust_c * robust_c)) ** -0.5)

                # WoLF 的作用等价于降低观测精度。
                # 这里写成有效观测噪声：
                #     R_eff = R / rho^2
                # residual 越异常，rho 越小，R_eff 越大，kalman_gain 越小。
                R_eff = float(self.observation_R / (rho * rho + eps))
                kalman_gain = float(P_pred / (P_pred + R_eff + eps))

                mu_new = float(mu_pred + kalman_gain * residual)
                P_new = float(max((1.0 - kalman_gain) * P_pred, eps))
                age = float(age_old + 1.0)

            abs_residual = abs(float(residual))
            mu_update_abs = abs(float(mu_new - mu_pred))

            self.history_states[client_id] = {
                "mu": float(mu_new),
                "P": float(P_new),
                "age": float(age),
            }

            # 四种状态只用于诊断，不写手工 gate：
            # current_good=True, history_good=True   -> 当前好，历史好
            # current_good=True, history_good=False  -> 当前好，历史差
            # current_good=False, history_good=True  -> 当前差，历史好
            # current_good=False, history_good=False -> 当前差，历史差
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
                    "h_median": float(h_median),
                    "h_mad": float(h_mad),
                    "z": float(z),
                    "sample_median": float(sample_median),
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
                    "current_good": bool(current_good),
                    "history_good": bool(history_good),
                    "state_label": str(state_label),
                    "logit": float(logit),
                }
            )

            filtered_records.append(enriched)

        return filtered_records

    def build_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        param_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建 fisher_history_wolf_global 诊断信息。

        诊断目标：
        1. 确认是否读到了 global_fisher。
        2. 看 WoLF 是否在压异常 Fisher 尖峰。
        3. 看最终权重是否仍保留 Fisher 区分度。
        4. 看 num_samples 是否重新支配权重。
        5. 看四种状态分布是否合理。
        """
        param_count = None
        if param_names is not None:
            param_count = len(list(param_names))

        raw_records = list(self._last_raw_records)
        filtered_records = list(self._last_filtered_records)

        missing_clients = []
        invalid_clients = []

        record_by_client = {
            int(record["client_id"]): record
            for record in raw_records
        }

        for update in client_updates:
            client_id = int(update.client_id)

            if "global_fisher" not in update.extra:
                missing_clients.append(client_id)
                continue

            if client_id not in record_by_client:
                invalid_clients.append(client_id)

        base_diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "diagnostics_enabled": bool(self.diagnostics_enabled),
            "num_clients": int(len(client_updates)),
            "param_count": param_count,
            "valid_clients": int(len(raw_records)),
            "filtered_clients": int(len(filtered_records)),
            "missing_clients": int(len(missing_clients)),
            "invalid_clients": int(len(invalid_clients)),
            "min_valid_clients": int(self.min_valid_clients),
            "fallback_to_uniform": bool(self._last_fallback_to_uniform),
            "history_size": int(len(self.history_states)),
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in weights.items()
            },
        }

        if not self.diagnostics_enabled:
            return base_diagnostics

        fisher_strengths = [
            float(record["fisher_strength"])
            for record in raw_records
        ]
        scores = [
            float(record["score"])
            for record in raw_records
        ]
        num_samples = [
            float(record["num_samples"])
            for record in raw_records
        ]
        log_fishers = [
            float(record["log_fisher"])
            for record in raw_records
        ]

        logits = [
            float(record["logit"])
            for record in filtered_records
        ]
        z_values = [
            float(record["z"])
            for record in filtered_records
        ]
        supports = [
            float(record["support"])
            for record in filtered_records
        ]
        residuals = [
            float(record["residual"])
            for record in filtered_records
        ]
        abs_residuals = [
            float(record["abs_residual"])
            for record in filtered_records
        ]
        rho_values = [
            float(record["rho"])
            for record in filtered_records
        ]
        kalman_gains = [
            float(record["kalman_gain"])
            for record in filtered_records
        ]
        mu_values = [
            float(record["mu_new"])
            for record in filtered_records
        ]
        P_values = [
            float(record["P_new"])
            for record in filtered_records
        ]
        mu_update_abs_values = [
            float(record["mu_update_abs"])
            for record in filtered_records
        ]

        record_weights = [
            float(weights.get(int(record["client_id"]), 0.0))
            for record in filtered_records
        ]
        record_fishers = [
            float(record["fisher_strength"])
            for record in filtered_records
        ]
        record_samples = [
            float(record["num_samples"])
            for record in filtered_records
        ]

        top_client = None
        if len(weights) > 0:
            top_client = max(weights.items(), key=lambda item: item[1])[0]

        top1_weight, top2_weight, top1_gap = _top_weight_stats(weights)

        state_counts = _state_count_dict(filtered_records)

        diagnostics = dict(base_diagnostics)
        diagnostics.update(
            {
                "top_client": int(top_client) if top_client is not None else None,
                "weight_entropy": float(_weight_entropy(weights)),
                "weight_entropy_norm": float(_weight_entropy_norm(weights)),
                "effective_clients": float(_effective_clients(weights)),
                "weight_min": float(min(weights.values())) if len(weights) > 0 else 0.0,
                "weight_max": float(max(weights.values())) if len(weights) > 0 else 0.0,
                "top1_weight": float(top1_weight),
                "top2_weight": float(top2_weight),
                "top1_gap": float(top1_gap),
                "state_counts": state_counts,
                "score_stats": _stat_dict(scores),
                "log_fisher_stats": _stat_dict(log_fishers),
                "fisher_strength_stats": _stat_dict(fisher_strengths),
                "num_samples_stats": _stat_dict(num_samples),
                "logit_stats": _stat_dict(logits),
                "z_stats": _stat_dict(z_values),
                "support_stats": _stat_dict(supports),
                "residual_stats": _stat_dict(residuals),
                "abs_residual_stats": _stat_dict(abs_residuals),
                "rho_stats": _stat_dict(rho_values),
                "kalman_gain_stats": _stat_dict(kalman_gains),
                "mu_stats": _stat_dict(mu_values),
                "P_stats": _stat_dict(P_values),
                "mu_update_abs_stats": _stat_dict(mu_update_abs_values),
                "score_cv": float(_coefficient_of_variation(scores)),
                "fisher_strength_cv": float(
                    _coefficient_of_variation(fisher_strengths)
                ),
                "num_samples_cv": float(_coefficient_of_variation(num_samples)),
                "weight_num_samples_corr": float(
                    _pearson_corr(record_weights, record_samples)
                ),
                "weight_fisher_corr": float(
                    _pearson_corr(record_weights, record_fishers)
                ),
                "weight_mu_corr": float(
                    _pearson_corr(record_weights, mu_values)
                ),
                "weight_support_corr": float(
                    _pearson_corr(record_weights, supports)
                ),
                "weight_rho_corr": float(
                    _pearson_corr(record_weights, rho_values)
                ),
            }
        )

        if self.diagnostics_include_records:
            diagnostics["missing_client_ids"] = [
                int(client_id)
                for client_id in missing_clients
            ]
            diagnostics["invalid_client_ids"] = [
                int(client_id)
                for client_id in invalid_clients
            ]
            diagnostics["raw_records"] = [
                _compact_record(record)
                for record in raw_records
            ]
            diagnostics["filtered_records"] = [
                {
                    **_compact_record(record),
                    "weight": float(
                        weights.get(int(record["client_id"]), 0.0)
                    ),
                    "z": float(record.get("z", 0.0)),
                    "support": float(record.get("support", 0.0)),
                    "mu_pred": float(record.get("mu_pred", 0.0)),
                    "mu_new": float(record.get("mu_new", 0.0)),
                    "P_new": float(record.get("P_new", 0.0)),
                    "residual": float(record.get("residual", 0.0)),
                    "abs_residual": float(record.get("abs_residual", 0.0)),
                    "rho": float(record.get("rho", 0.0)),
                    "kalman_gain": float(record.get("kalman_gain", 0.0)),
                    "logit": float(record.get("logit", 0.0)),
                    "state_label": str(record.get("state_label", "")),
                    "cold_start": bool(record.get("cold_start", False)),
                }
                for record in filtered_records
            ]

        return diagnostics


def _get_global_fisher_payload(
    update: ClientUpdate,
    missing_policy: str,
) -> Optional[Mapping[str, Any]]:
    """
    从 ClientUpdate.extra 中读取 global_fisher payload。

    期望格式：
        update.extra["global_fisher"] = {
            "fisher_strength": ...,
            "num_samples": ...,
            "score": ...,
            "meta": {...},
        }
    """
    if "global_fisher" not in update.extra:
        if missing_policy == "error":
            raise KeyError(
                f"客户端 {update.client_id} 缺少 extra['global_fisher']。"
                "请确认 full_model_fisher.enabled=true，且 client.py 已经调用 "
                "collect_full_model_fisher_stats(...)。"
            )

        if missing_policy in {"skip", "uniform"}:
            return None

        raise ValueError(
            "full_model_fisher.missing_policy 只支持 error / skip / uniform，"
            f"当前值：{missing_policy}"
        )

    payload = update.extra["global_fisher"]

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"客户端 {update.client_id} 的 extra['global_fisher'] 类型错误，"
            f"期望 Mapping，实际是 {type(payload)}。"
        )

    return payload


def _extract_fisher_strength(payload: Mapping[str, Any]) -> float:
    """
    从 global_fisher payload 中提取 fisher_strength。
    """
    return _safe_float(
        payload.get("fisher_strength", 0.0),
        default=0.0,
    )


def _extract_num_samples(
    update: ClientUpdate,
    payload: Mapping[str, Any],
) -> int:
    """
    提取 Fisher evidence 对应的支撑样本数。

    优先级：
    1. payload["num_samples"]
    2. payload["total_samples"]
    3. update.num_samples

    说明：
    - full_model_fisher.py 的 to_payload() 推荐写 num_samples。
    - 如果 evidence pass 只用了 max_batches，那么 num_samples 可能小于 update.num_samples。
    - 这里优先使用 payload 里的 evidence 样本数。
    """
    if "num_samples" in payload:
        return _safe_int(payload.get("num_samples", 0), default=0)

    if "total_samples" in payload:
        return _safe_int(payload.get("total_samples", 0), default=0)

    return int(update.num_samples)


def _extract_score(
    payload: Mapping[str, Any],
    fisher_strength: float,
    num_samples: int,
) -> float:
    """
    提取 Fisher score。

    优先级：
    1. payload["score"]
    2. num_samples * fisher_strength

    注意：
    fisher_history_wolf_global 的主信号仍然是 fisher_strength；
    score 主要用于诊断和对齐 fisher_only_global。
    """
    if "score" in payload:
        return _safe_float(payload.get("score", 0.0), default=0.0)

    return float(num_samples) * float(fisher_strength)


def _softmax_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[int, float]:
    """
    对 records 的 logit 做 softmax，得到客户端权重。
    """
    if len(records) == 0:
        return {}

    logits = [
        float(record.get("logit", 0.0))
        for record in records
    ]

    max_logit = max(logits)

    exp_values = [
        math.exp(float(logit) - float(max_logit))
        for logit in logits
    ]

    total = sum(exp_values)

    if total <= 0.0 or not math.isfinite(total):
        weight = 1.0 / float(len(records))
        return {
            int(record["client_id"]): float(weight)
            for record in records
        }

    weights = {}

    for record, exp_value in zip(records, exp_values):
        weights[int(record["client_id"])] = float(exp_value / total)

    return weights


def _build_uniform_raw_weights(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    构建 uniform raw weights。

    注意：
    这里返回 raw weight = 1.0。
    后续 Aggregator.aggregate() 会统一 normalize。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    return {
        int(update.client_id): 1.0
        for update in client_updates
    }


def _sum_positive(values: Sequence[Any]) -> float:
    """
    对正的有限数值求和。
    """
    total = 0.0

    for value in values:
        value = _safe_float(value, default=0.0)
        if value > 0.0 and math.isfinite(value):
            total += value

    return float(total)


def _median(values: Sequence[Any]) -> float:
    """
    计算中位数。

    这里不用 numpy，避免给聚合器引入额外依赖。
    """
    clean_values = sorted(
        [
            float(value)
            for value in values
            if _is_finite_number(value)
        ]
    )

    if len(clean_values) == 0:
        return 0.0

    mid = len(clean_values) // 2

    if len(clean_values) % 2 == 1:
        return float(clean_values[mid])

    return float(0.5 * (clean_values[mid - 1] + clean_values[mid]))


def _state_label(
    current_good: bool,
    history_good: bool,
) -> str:
    """
    根据当前观测和历史状态生成四种状态标签。

    这些标签只用于诊断，不参与手工调权。
    """
    if current_good and history_good:
        return "current_good_history_good"

    if current_good and not history_good:
        return "current_good_history_bad"

    if not current_good and history_good:
        return "current_bad_history_good"

    return "current_bad_history_bad"


def _state_count_dict(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """
    统计四种 current/history 状态数量。
    """
    counts = {
        "current_good_history_good": 0,
        "current_good_history_bad": 0,
        "current_bad_history_good": 0,
        "current_bad_history_bad": 0,
    }

    for record in records:
        label = str(record.get("state_label", ""))
        if label not in counts:
            continue

        counts[label] += 1

    return counts


def _weight_entropy(weights: Mapping[int, float]) -> float:
    """
    计算权重熵。

    权重越均匀，熵越大。
    某个客户端支配时，熵会变小。
    """
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

    含义：
    - 接近 1：权重接近 uniform。
    - 接近 0：某个客户端强烈支配。
    """
    if len(weights) <= 1:
        return 0.0

    entropy = _weight_entropy(weights)
    max_entropy = math.log(float(len(weights)) + 1.0e-12)

    return _safe_divide(entropy, max_entropy)


def _effective_clients(weights: Mapping[int, float]) -> float:
    """
    计算有效客户端数。

    公式：
        effective_clients = 1 / sum_i w_i^2

    含义：
    - 接近参与客户端数：权重接近 uniform。
    - 接近 1：单个客户端支配。
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
) -> tuple[float, float, float]:
    """
    返回 top1_weight、top2_weight、top1_gap。
    """
    if len(weights) == 0:
        return 0.0, 0.0, 0.0

    sorted_weights = sorted(
        [
            float(weight)
            for weight in weights.values()
        ],
        reverse=True,
    )

    top1 = sorted_weights[0]
    top2 = sorted_weights[1] if len(sorted_weights) >= 2 else 0.0

    return float(top1), float(top2), float(top1 - top2)


def _coefficient_of_variation(
    values: Sequence[Any],
    eps: float = 1.0e-12,
) -> float:
    """
    计算变异系数 CV = std / abs(mean)。

    用途：
    判断 fisher_strength / num_samples / score 是否有区分度。
    """
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

    var = sum(
        (value - mean) ** 2
        for value in clean_values
    ) / len(clean_values)

    std = math.sqrt(max(var, 0.0))

    return float(std / (abs(mean) + eps))


def _pearson_corr(
    xs: Sequence[Any],
    ys: Sequence[Any],
    eps: float = 1.0e-12,
) -> float:
    """
    计算 Pearson 相关系数。

    用途：
    - weight_num_samples_corr：判断权重是否主要由样本数控制。
    - weight_fisher_corr：判断权重是否真的受 Fisher 强度影响。
    - weight_mu_corr：判断历史状态是否影响权重。
    """
    clean_pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if _is_finite_number(x) and _is_finite_number(y)
    ]

    if len(clean_pairs) <= 1:
        return 0.0

    clean_xs = [
        pair[0]
        for pair in clean_pairs
    ]
    clean_ys = [
        pair[1]
        for pair in clean_pairs
    ]

    mean_x = sum(clean_xs) / len(clean_xs)
    mean_y = sum(clean_ys) / len(clean_ys)

    centered_xs = [
        value - mean_x
        for value in clean_xs
    ]
    centered_ys = [
        value - mean_y
        for value in clean_ys
    ]

    numerator = sum(
        x * y
        for x, y in zip(centered_xs, centered_ys)
    )

    denom_x = math.sqrt(
        sum(x * x for x in centered_xs)
    )
    denom_y = math.sqrt(
        sum(y * y for y in centered_ys)
    )

    denominator = denom_x * denom_y

    if denominator <= eps:
        return 0.0

    return float(numerator / (denominator + eps))


def _stat_dict(values: Sequence[Any]) -> Dict[str, float]:
    """
    生成一组数值的基础统计量。
    """
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
    var = sum(
        (value - mean) ** 2
        for value in clean_values
    ) / len(clean_values)
    std = math.sqrt(max(var, 0.0))

    return {
        "count": float(len(clean_values)),
        "mean": float(mean),
        "std": float(std),
        "min": float(min(clean_values)),
        "max": float(max(clean_values)),
    }


def _compact_record(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    将 record 压成适合写日志的普通 dict。
    """
    return {
        "client_id": int(record.get("client_id", -1)),
        "num_samples": int(record.get("num_samples", 0)),
        "fisher_strength": float(record.get("fisher_strength", 0.0)),
        "score": float(record.get("score", 0.0)),
        "log_fisher": float(record.get("log_fisher", 0.0)),
        "source": str(record.get("source", "global_fisher")),
    }


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """
    安全除法。
    """
    numerator = _safe_float(numerator, default=0.0)
    denominator = _safe_float(denominator, default=0.0)

    if denominator == 0.0:
        return float(default)

    return float(numerator / denominator)


def _is_finite_number(value: Any) -> bool:
    """
    判断 value 是否能转成有限浮点数。
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(result)


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
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


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    安全转 int。
    """
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

    另外兼容简单 dotted key：
        _cfg_get(cfg, "fisher_history_wolf_global.eps", 1e-8)
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        value = cfg.get(key, None)
        if value is not None:
            return value

    if hasattr(cfg, key):
        return getattr(cfg, key)

    if "." in key:
        current = cfg
        for part in key.split("."):
            current = _cfg_get(current, part, None)
            if current is None:
                return default
        return current

    return default