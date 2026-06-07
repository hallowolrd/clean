from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aggregation.base import Aggregator, build_uniform_weights
from fl.types import AggregationResult, ClientUpdate
from utils.state_dict_ops import check_finite_state_dict, clone_state_dict, normalize_weights


@dataclass
class _ClientExpertRecord:
    """
    单个 client-expert 的本轮统计记录。

    注意：
    - fisher_score 是本轮 K-FAC 分数，不乘 active_count。
    - active_count 只用于判断观测是否可靠，不直接放大观测值。
    """

    client_id: int
    expert_id: int
    fisher_score: float
    active_count: float
    has_kfac: bool


class HistoryWoLFKFACScoreExpertAggregator(Aggregator):
    """
    History-WoLF K-FAC Score 专家聚合器。

    核心流程：
    1. 从每个客户端上传的 expert K-FAC 统计中得到 fisher_score_i,e。
    2. 把 log(fisher_score_i,e + eps) 当作滤波器本轮观测。
    3. active_count 只用于 valid client 判断、route_quality、观测精度缩放。
    4. WoLF 残差权重用于抑制和可靠历史强冲突的异常观测。
    5. 对同一个 expert 下的客户端 filtered log-score 做 softmax，得到聚合权重。
    6. 用 per-expert client weights 聚合 expert 参数。

    这个方法不是严格的 FedFisher K-FAC 二次目标求解器；
    它是 K-FAC score + WoLF 历史滤波 + expert 加权平均。
    """

    def __init__(self, cfg: Any, param_group_name: str) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)

        if str(param_group_name) != "expert":
            raise ValueError(
                "HistoryWoLFKFACScoreExpertAggregator 只支持 expert 参数组，"
                f"当前 param_group_name={param_group_name}"
            )

        # ------------------------------
        # 消融开关
        # ------------------------------
        # 注意：
        # 当前项目采用极致解耦配置风格。
        # history_wolf_kfac_score 的参数放在顶层 history_wolf_kfac_score.xxx，
        # 不放在 agg.expert.xxx 下面。
        self.fisher_score_enabled = _cfg_bool(
            cfg, "history_wolf_kfac_score.fisher_score_enabled", True
        )
        self.history_filter_enabled = _cfg_bool(
            cfg, "history_wolf_kfac_score.history_filter_enabled", True
        )

        # ------------------------------
        # 历史滤波超参数
        # ------------------------------
        self.min_active_count = int(
            _cfg_get(cfg, "history_wolf_kfac_score.min_active_count", 1)
        )
        self.min_valid_clients = int(
            _cfg_get(cfg, "history_wolf_kfac_score.min_valid_clients", 2)
        )
        self.fallback = str(
            _cfg_get(cfg, "history_wolf_kfac_score.fallback", "keep_global")
        ).lower().strip()

        self.active_count_ref = float(
            _cfg_get(cfg, "history_wolf_kfac_score.active_count_ref", 32.0)
        )
        self.rho = float(_cfg_get(cfg, "history_wolf_kfac_score.rho", 0.95))
        self.c_wolf = float(
            _cfg_get(cfg, "history_wolf_kfac_score.c_wolf", 2.5)
        )
        self.min_obs_scale = float(
            _cfg_get(cfg, "history_wolf_kfac_score.min_obs_scale", 0.05)
        )
        self.seen_ref = float(
            _cfg_get(cfg, "history_wolf_kfac_score.seen_ref", 5.0)
        )
        self.q_scale = float(
            _cfg_get(cfg, "history_wolf_kfac_score.q_scale", 0.05)
        )
        self.tau_cur = float(
            _cfg_get(cfg, "history_wolf_kfac_score.tau_cur", 1.0)
        )
        self.tau_hist = float(
            _cfg_get(cfg, "history_wolf_kfac_score.tau_hist", 1.0)
        )
        self.init_P = float(
            _cfg_get(cfg, "history_wolf_kfac_score.init_P", 1.0)
        )
        self.eps = float(_cfg_get(cfg, "history_wolf_kfac_score.eps", 1.0e-8))

        if self.min_active_count < 0:
            raise ValueError("history_wolf_kfac_score.min_active_count 不能小于 0。")
        if self.min_valid_clients <= 0:
            raise ValueError("history_wolf_kfac_score.min_valid_clients 必须大于 0。")
        if self.fallback not in {"keep_global", "uniform"}:
            raise ValueError(
                "history_wolf_kfac_score.fallback 只支持 keep_global / uniform，"
                f"当前值：{self.fallback}"
            )

        # 每个 (client_id, expert_id) 保存一个标量历史滤波状态。
        self.history: Dict[Tuple[int, int], Dict[str, float | int]] = {}

        # 最近一轮诊断缓存。compute_weights() 不负责真实权重，这里仅用于外部查看。
        self._last_diagnostics: Dict[str, Any] = {}
        self._last_client_weights: Dict[int, float] = {}

    @property
    def method_name(self) -> str:
        return "history_wolf_kfac_score"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        为了兼容 Aggregator 抽象接口保留。

        注意：
        这个聚合器真正使用的是 per-expert client weights，
        不是一个全局 client weight。
        """
        if self._last_client_weights:
            return dict(self._last_client_weights)
        return build_uniform_weights(client_updates)

    def state_dict(self) -> Dict[str, Any]:
        """
        导出历史滤波状态，方便后续接入 server checkpoint。

        当前如果 server 还没有保存 aggregator state 的机制，
        这个方法也可以先不被调用。
        """
        return {
            "method": self.method_name,
            "history": {
                f"{client_id}:{expert_id}": dict(state)
                for (client_id, expert_id), state in self.history.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        恢复历史滤波状态。
        """
        raw_history = state.get("history", {})
        history: Dict[Tuple[int, int], Dict[str, float | int]] = {}

        for key, value in raw_history.items():
            try:
                client_str, expert_str = str(key).split(":")
                client_id = int(client_str)
                expert_id = int(expert_str)
            except Exception:
                continue

            if not isinstance(value, Mapping):
                continue

            history[(client_id, expert_id)] = {
                "mu": float(value.get("mu", 0.0)),
                "P": float(value.get("P", self.init_P)),
                "seen": int(value.get("seen", 0)),
                "bad_streak": int(value.get("bad_streak", 0)),
            }

        self.history = history

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        执行 expert 参数聚合。

        和普通 Aggregator.aggregate() 不同：
        - 普通聚合器是所有参数共用一个 client weight。
        - 这里是每个 expert 单独计算 client weights。
        """
        self._validate_client_updates(client_updates)

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        names = _resolve_param_names(
            state_dict=global_state,
            param_names=param_names,
        )
        expert_to_param_names = _group_param_names_by_expert(names)

        if len(expert_to_param_names) == 0:
            # 如果没有识别到 expert 参数，就保持原状态，避免误伤非专家参数。
            diagnostics = {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "num_clients": len(client_updates),
                "param_count": len(names),
                "fisher_score_enabled": bool(self.fisher_score_enabled),
                "history_filter_enabled": bool(self.history_filter_enabled),
                "reason": "no_expert_params_matched",
            }
            check_finite_state_dict(new_state_dict, param_names=names)
            return AggregationResult(
                new_state_dict=new_state_dict,
                weights=build_uniform_weights(client_updates),
                diagnostics=diagnostics,
            )

        records_by_expert = self._build_records_by_expert(
            client_updates=client_updates,
            expert_ids=sorted(expert_to_param_names.keys()),
        )

        per_expert_weights: Dict[int, Dict[int, float]] = {}
        per_expert_fallback: Dict[int, bool] = {}
        per_expert_valid_clients: Dict[int, int] = {}
        per_expert_weight_stats: Dict[int, Dict[str, float]] = {}
        per_expert_state_counts: Dict[int, Dict[str, int]] = {}

        all_fisher_scores: List[float] = []
        all_active_counts: List[float] = []
        all_current_quality: List[float] = []
        all_history_quality: List[float] = []
        all_wolf: List[float] = []
        all_kalman_gain: List[float] = []

        for expert_id in sorted(expert_to_param_names.keys()):
            records = records_by_expert.get(expert_id, [])

            result = self._compute_expert_weights_and_update_history(
                expert_id=expert_id,
                records=records,
            )

            per_expert_weights[expert_id] = result["weights"]
            per_expert_fallback[expert_id] = bool(result["fallback"])
            per_expert_valid_clients[expert_id] = int(result["valid_clients"])
            per_expert_state_counts[expert_id] = dict(result["state_counts"])

            weights = list(result["weights"].values())
            if weights:
                per_expert_weight_stats[expert_id] = {
                    "min": float(min(weights)),
                    "mean": float(sum(weights) / len(weights)),
                    "max": float(max(weights)),
                }
            else:
                per_expert_weight_stats[expert_id] = {
                    "min": 0.0,
                    "mean": 0.0,
                    "max": 0.0,
                }

            all_fisher_scores.extend(result["fisher_scores"])
            all_active_counts.extend(result["active_counts"])
            all_current_quality.extend(result["current_quality"])
            all_history_quality.extend(result["history_quality"])
            all_wolf.extend(result["wolf"])
            all_kalman_gain.extend(result["kalman_gain"])

        # 根据每个 expert 自己的 client weights 写回 expert 参数。
        for expert_id, expert_param_names in expert_to_param_names.items():
            weights = per_expert_weights.get(expert_id, {})
            fallback_used = per_expert_fallback.get(expert_id, False)

            if fallback_used and self.fallback == "keep_global":
                # keep_global：保留 base_state/global_state 中已有 expert 参数，不做更新。
                continue

            if not weights:
                continue

            for name in expert_param_names:
                global_tensor = global_state[name]
                if not torch.is_tensor(global_tensor):
                    continue
                if not torch.is_floating_point(global_tensor):
                    continue

                total_delta = torch.zeros_like(global_tensor)

                for update in client_updates:
                    client_id = int(update.client_id)
                    if client_id not in weights:
                        continue

                    if name not in update.model_delta:
                        if strict:
                            raise KeyError(
                                f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                            )
                        continue

                    weight = float(weights[client_id])
                    delta_tensor = update.model_delta[name].to(global_tensor.device)
                    total_delta = total_delta + weight * delta_tensor

                new_state_dict[name] = global_tensor + total_delta

        check_finite_state_dict(new_state_dict, param_names=names)

        client_weights = _average_per_expert_weights(
            per_expert_weights=per_expert_weights,
            client_updates=client_updates,
        )
        self._last_client_weights = dict(client_weights)

        diagnostics = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": len(names),
            "fisher_score_enabled": bool(self.fisher_score_enabled),
            "history_filter_enabled": bool(self.history_filter_enabled),
            "min_active_count": int(self.min_active_count),
            "min_valid_clients": int(self.min_valid_clients),
            "fallback": self.fallback,
            "per_expert_valid_clients": {
                int(k): int(v) for k, v in per_expert_valid_clients.items()
            },
            "per_expert_fallback": {
                int(k): bool(v) for k, v in per_expert_fallback.items()
            },
            "per_expert_weight_stats": {
                int(k): dict(v) for k, v in per_expert_weight_stats.items()
            },
            "per_expert_state_counts": {
                int(k): dict(v) for k, v in per_expert_state_counts.items()
            },
            "mean_fisher_score": _safe_mean(all_fisher_scores),
            "mean_active_count": _safe_mean(all_active_counts),
            "mean_current_quality": _safe_mean(all_current_quality),
            "mean_history_quality": _safe_mean(all_history_quality),
            "mean_wolf": _safe_mean(all_wolf),
            "mean_kalman_gain": _safe_mean(all_kalman_gain),
            "expert_client_weights": {
                int(expert_id): {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                for expert_id, weights in per_expert_weights.items()
            },
            # AggregationResult.weights 只能放一份 client-level 权重；
            # 这里给的是 per-expert 权重的平均值，仅用于日志兼容。
            "weights": {int(k): float(v) for k, v in client_weights.items()},
        }

        self._last_diagnostics = dict(diagnostics)

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=client_weights,
            diagnostics=diagnostics,
        )

    def _build_records_by_expert(
        self,
        client_updates: Sequence[ClientUpdate],
        expert_ids: Sequence[int],
    ) -> Dict[int, List[_ClientExpertRecord]]:
        """
        从 client_updates 中构造每个 expert 的本轮记录。
        """
        records_by_expert: Dict[int, List[_ClientExpertRecord]] = {
            int(expert_id): [] for expert_id in expert_ids
        }

        for update in client_updates:
            client_id = int(update.client_id)
            extra = dict(update.extra or {})
            expert_kfac = extra.get("expert_kfac", None)
            expert_usage = extra.get("expert_usage", None)

            kfac_stats = _extract_kfac_stats_by_expert(
                expert_kfac=expert_kfac,
                eps=self.eps,
            )

            for expert_id in expert_ids:
                stats = kfac_stats.get(int(expert_id), None)

                has_kfac = stats is not None
                if stats is not None:
                    fisher_score = float(stats["fisher_score"])
                    kfac_count = float(stats["active_count"])
                else:
                    fisher_score = 1.0
                    kfac_count = 0.0

                usage_count = _extract_active_count_from_usage(
                    expert_usage=expert_usage,
                    expert_id=int(expert_id),
                )

                if usage_count is not None:
                    active_count = float(usage_count)
                elif kfac_count > 0:
                    active_count = float(kfac_count)
                else:
                    # 如果完全没有 usage/K-FAC 计数，并且关闭 Fisher，
                    # 允许用 num_samples 作为一个兜底观测计数，保证消融能跑通。
                    if not self.fisher_score_enabled:
                        active_count = float(update.num_samples)
                    else:
                        active_count = 0.0

                if not self.fisher_score_enabled:
                    fisher_score = 1.0

                records_by_expert[int(expert_id)].append(
                    _ClientExpertRecord(
                        client_id=client_id,
                        expert_id=int(expert_id),
                        fisher_score=float(max(fisher_score, self.eps)),
                        active_count=float(max(active_count, 0.0)),
                        has_kfac=bool(has_kfac),
                    )
                )

        return records_by_expert

    def _compute_expert_weights_and_update_history(
        self,
        expert_id: int,
        records: Sequence[_ClientExpertRecord],
    ) -> Dict[str, Any]:
        """
        对单个 expert 计算 per-client weights，并更新历史状态。
        """
        if len(records) == 0:
            return {
                "weights": {},
                "fallback": True,
                "valid_clients": 0,
                "state_counts": _empty_state_counts(),
                "fisher_scores": [],
                "active_counts": [],
                "current_quality": [],
                "history_quality": [],
                "wolf": [],
                "kalman_gain": [],
            }

        # 必做修正 1：
        # 低 active_count 的 client 不参与 median/MAD，也不参与本轮 expert 聚合。
        valid_records = [
            record
            for record in records
            if record.active_count >= float(self.min_active_count)
            and (record.has_kfac or not self.fisher_score_enabled)
        ]

        if len(valid_records) < int(self.min_valid_clients):
            weights = self._fallback_weights(records=records)
            return {
                "weights": weights,
                "fallback": True,
                "valid_clients": len(valid_records),
                "state_counts": _empty_state_counts(),
                "fisher_scores": [float(r.fisher_score) for r in records],
                "active_counts": [float(r.active_count) for r in records],
                "current_quality": [],
                "history_quality": [],
                "wolf": [],
                "kalman_gain": [],
            }

        y_by_client: Dict[int, float] = {}
        for record in valid_records:
            if self.fisher_score_enabled:
                y_by_client[record.client_id] = math.log(
                    float(record.fisher_score) + self.eps
                )
            else:
                y_by_client[record.client_id] = 0.0

        y_values = list(y_by_client.values())
        med_cur = _median(y_values)
        mad_cur = _mad(y_values, center=med_cur) + self.eps

        R_e = max(mad_cur * mad_cur, self.eps)
        Q_e = max(self.q_scale * mad_cur * mad_cur, self.eps)
        base_e = med_cur - mad_cur

        # 历史 median/MAD 只在 seen>0 的 valid clients 上计算。
        hist_mu_values: List[float] = []
        for record in valid_records:
            state = self.history.get((record.client_id, expert_id), None)
            if state is None:
                continue
            if int(state.get("seen", 0)) <= 0:
                continue
            hist_mu_values.append(float(state.get("mu", base_e)))

        if len(hist_mu_values) > 0:
            med_hist = _median(hist_mu_values)
            mad_hist = _mad(hist_mu_values, center=med_hist) + self.eps
        else:
            med_hist = base_e
            mad_hist = max(mad_cur, self.eps)

        if not self.history_filter_enabled:
            logits = {
                int(client_id): float(y_value)
                for client_id, y_value in y_by_client.items()
            }
            weights = _softmax_dict(logits)
            return {
                "weights": weights,
                "fallback": False,
                "valid_clients": len(valid_records),
                "state_counts": _empty_state_counts(),
                "fisher_scores": [float(r.fisher_score) for r in valid_records],
                "active_counts": [float(r.active_count) for r in valid_records],
                "current_quality": [],
                "history_quality": [],
                "wolf": [],
                "kalman_gain": [],
            }

        final_logits: Dict[int, float] = {}
        state_counts = _empty_state_counts()

        fisher_scores: List[float] = []
        active_counts: List[float] = []
        current_quality_values: List[float] = []
        history_quality_values: List[float] = []
        wolf_values: List[float] = []
        kalman_gain_values: List[float] = []

        for record in valid_records:
            client_id = int(record.client_id)
            y = float(y_by_client[client_id])

            state_key = (client_id, int(expert_id))
            old_state = self.history.get(state_key, None)

            if old_state is None:
                mu = float(base_e)
                P = float(self.init_P)
                seen = 0
                bad_streak = 0
            else:
                mu = float(old_state.get("mu", base_e))
                P = float(old_state.get("P", self.init_P))
                seen = int(old_state.get("seen", 0))
                bad_streak = int(old_state.get("bad_streak", 0))

            P = max(P, self.eps)

            # 当前质量：只用 active_count 判断观测可靠性，不把 active_count 乘进观测值。
            z_cur = (y - med_cur) / (mad_cur + self.eps)
            route_quality = math.sqrt(
                float(record.active_count)
                / (float(record.active_count) + self.active_count_ref + self.eps)
            )
            current_quality = route_quality * _sigmoid(
                z_cur / max(self.tau_cur, self.eps)
            )
            current_quality = _clamp(current_quality, 0.0, 1.0)

            # 历史质量：历史强度 + seen/P 置信度。
            if seen <= 0:
                history_quality = 0.0
                history_conf_old = 0.0
            else:
                z_hist = (mu - med_hist) / (mad_hist + self.eps)
                history_strength = _sigmoid(z_hist / max(self.tau_hist, self.eps))
                history_conf_old = (
                    float(seen) / (float(seen) + self.seen_ref + self.eps)
                ) * (R_e / (P + R_e + self.eps))
                history_quality = history_strength * history_conf_old
                history_quality = _clamp(history_quality, 0.0, 1.0)

            # 历史预测：长期历史向本轮低基准轻微衰减，避免历史永久霸占权重。
            mu_pred = self.rho * mu + (1.0 - self.rho) * base_e
            P_pred = max(P + Q_e, self.eps)

            # 必做修正 3：
            # residual 距离加入 sqrt(P_pred)，避免用不可靠历史惩罚本轮观测。
            residual_dist = abs(y - mu_pred) / (
                math.sqrt(P_pred) + mad_cur + self.eps
            )

            wolf = 1.0 / math.sqrt(
                1.0 + (residual_dist / max(self.c_wolf, self.eps)) ** 2
            )
            wolf = _clamp(wolf, 0.0, 1.0)

            # 历史不好时，不让历史压制本轮。
            wolf_eff = (1.0 - history_quality) + history_quality * wolf
            wolf_eff = _clamp(wolf_eff, 0.0, 1.0)

            obs_scale = max(current_quality, self.min_obs_scale) * (wolf_eff**2)
            obs_scale = max(obs_scale, self.eps)
            R_eff = R_e / obs_scale

            K = P_pred / (P_pred + R_eff + self.eps)
            K = _clamp(K, 0.0, 1.0)

            mu_new = mu_pred + K * (y - mu_pred)
            P_new = max((1.0 - K) * P_pred, self.eps)

            current_good = bool(current_quality >= 0.5)
            history_good = bool(history_quality >= 0.5)
            state_name = _state_name(
                current_good=current_good,
                history_good=history_good,
            )
            state_counts[state_name] += 1

            bad_streak_new = bad_streak + 1 if not current_good else 0
            seen_new = int(seen + 1)

            self.history[state_key] = {
                "mu": float(mu_new),
                "P": float(P_new),
                "seen": int(seen_new),
                "bad_streak": int(bad_streak_new),
            }

            # 用更新后的 P/seen 得到最终置信度。
            history_conf_new = (
                float(seen_new) / (float(seen_new) + self.seen_ref + self.eps)
            ) * (R_e / (P_new + R_e + self.eps))
            history_conf_new = _clamp(history_conf_new, self.eps, 1.0)

            # 稳定 softmax 前的 logit。
            # 这里不用先 exp，避免 filtered_score 数值爆炸。
            final_logits[client_id] = float(
                mu_new + math.log(history_conf_new + self.eps)
            )

            fisher_scores.append(float(record.fisher_score))
            active_counts.append(float(record.active_count))
            current_quality_values.append(float(current_quality))
            history_quality_values.append(float(history_quality))
            wolf_values.append(float(wolf_eff))
            kalman_gain_values.append(float(K))

        weights = _softmax_dict(final_logits)

        return {
            "weights": weights,
            "fallback": False,
            "valid_clients": len(valid_records),
            "state_counts": state_counts,
            "fisher_scores": fisher_scores,
            "active_counts": active_counts,
            "current_quality": current_quality_values,
            "history_quality": history_quality_values,
            "wolf": wolf_values,
            "kalman_gain": kalman_gain_values,
        }

    def _fallback_weights(
        self,
        records: Sequence[_ClientExpertRecord],
    ) -> Dict[int, float]:
        """
        expert 有效 client 太少时的 fallback。

        keep_global：
            返回空权重，上层保持该 expert 的 global 参数不变。
        uniform：
            对 active_count > 0 的客户端做均匀聚合；
            如果没有 active_count > 0 的客户端，则返回空权重。
        """
        if self.fallback == "keep_global":
            return {}

        candidates = [
            int(record.client_id)
            for record in records
            if float(record.active_count) > 0.0
        ]
        candidates = sorted(set(candidates))
        if len(candidates) == 0:
            return {}

        weight = 1.0 / float(len(candidates))
        return {int(client_id): float(weight) for client_id in candidates}


def _extract_kfac_stats_by_expert(
    expert_kfac: Any,
    eps: float,
) -> Dict[int, Dict[str, float]]:
    """
    从 update.extra["expert_kfac"] 中提取每个 expert 的 K-FAC score。

    当前 fl.kfac.collect_expert_kfac 返回大致格式：
    {
        "switch_layers.0.switch_ffn.experts.2.0": {
            "A": Tensor,
            "B": Tensor,
            "count": int,
            "trace_A": float,
            "trace_B": float,
            ...
        }
    }

    这里将每个 expert 的多层 K-FAC score 做 count 加权平均。
    """
    if not isinstance(expert_kfac, Mapping):
        return {}

    per_expert_layer_scores: Dict[int, List[Tuple[float, float]]] = {}
    per_expert_counts: Dict[int, List[float]] = {}

    for module_name, payload in expert_kfac.items():
        if not isinstance(payload, Mapping):
            continue

        expert_id = _parse_expert_id(
            str(payload.get("module_name", module_name))
        )
        if expert_id is None:
            weight_name = payload.get("weight_name", None)
            if weight_name is not None:
                expert_id = _parse_expert_id(str(weight_name))

        if expert_id is None:
            continue

        mean_A = _extract_mean_diag_from_kfac_payload(
            payload=payload,
            factor_name="A",
            trace_name="trace_A",
            dim_keys=("in_features",),
            include_bias=bool(payload.get("include_bias", False)),
            eps=eps,
        )
        mean_B = _extract_mean_diag_from_kfac_payload(
            payload=payload,
            factor_name="B",
            trace_name="trace_B",
            dim_keys=("out_features",),
            include_bias=False,
            eps=eps,
        )

        if mean_A <= 0.0 or mean_B <= 0.0:
            continue
        if not math.isfinite(mean_A) or not math.isfinite(mean_B):
            continue

        layer_score = math.sqrt(max(mean_A, eps) * max(mean_B, eps))
        count = float(payload.get("count", 0.0))
        if not math.isfinite(count) or count < 0.0:
            count = 0.0

        per_expert_layer_scores.setdefault(int(expert_id), []).append(
            (float(layer_score), max(count, 1.0))
        )
        per_expert_counts.setdefault(int(expert_id), []).append(float(count))

    result: Dict[int, Dict[str, float]] = {}
    for expert_id, layer_scores in per_expert_layer_scores.items():
        if len(layer_scores) == 0:
            continue

        total_weight = sum(weight for _, weight in layer_scores)
        if total_weight <= 0.0:
            fisher_score = sum(score for score, _ in layer_scores) / float(
                len(layer_scores)
            )
        else:
            fisher_score = sum(score * weight for score, weight in layer_scores)
            fisher_score /= total_weight

        counts = per_expert_counts.get(expert_id, [])
        active_count = max(counts) if counts else 0.0

        result[int(expert_id)] = {
            "fisher_score": float(max(fisher_score, eps)),
            "active_count": float(max(active_count, 0.0)),
        }

    return result


def _extract_mean_diag_from_kfac_payload(
    payload: Mapping[str, Any],
    factor_name: str,
    trace_name: str,
    dim_keys: Tuple[str, ...],
    include_bias: bool,
    eps: float,
) -> float:
    """
    提取 K-FAC 因子的平均对角量 trace(F) / dim。

    注意：
    如果 payload 里已经有 A/B tensor，那么 tensor.size(0) 就是真实维度。
    对 A 因子来说，include_bias=True 时，fl/kfac.py 通常已经把 bias 维度拼进 A 里。
    所以只有在没有 tensor、只能靠 in_features 推断维度时，才额外 +1。
    """
    trace_value = payload.get(trace_name, None)
    factor = payload.get(factor_name, None)

    if trace_value is None and torch.is_tensor(factor) and factor.dim() >= 2:
        trace_value = float(torch.trace(factor.detach().float()).item())

    if trace_value is None:
        return eps

    trace_float = float(trace_value)
    if not math.isfinite(trace_float):
        return eps

    dim = None
    dim_from_tensor = False

    if torch.is_tensor(factor) and factor.dim() >= 2:
        dim = int(factor.size(0))
        dim_from_tensor = True

    if dim is None:
        for key in dim_keys:
            if key in payload:
                dim = int(payload[key])
                break

    if dim is None or dim <= 0:
        dim = 1

    # 只有靠 in_features 推断 A 维度时才补 bias。
    # 如果 A tensor 已经存在，它的维度通常已经包含 bias，不能重复 +1。
    if include_bias and not dim_from_tensor:
        dim += 1

    return float(trace_float) / float(max(dim, 1))


def _extract_active_count_from_usage(
    expert_usage: Any,
    expert_id: int,
) -> Optional[float]:
    """
    从 update.extra["expert_usage"] 中读取 expert_counts[expert_id]。
    """
    if not isinstance(expert_usage, Mapping):
        return None

    expert_counts = expert_usage.get("expert_counts", None)
    if not isinstance(expert_counts, Mapping):
        return None

    value = None
    if expert_id in expert_counts:
        value = expert_counts[expert_id]
    elif str(expert_id) in expert_counts:
        value = expert_counts[str(expert_id)]

    if value is None:
        return None

    try:
        value_float = float(value)
    except Exception:
        return None

    if not math.isfinite(value_float):
        return None

    return max(value_float, 0.0)


def _group_param_names_by_expert(
    param_names: Sequence[str],
) -> Dict[int, List[str]]:
    """
    按 expert_id 分组参数名。

    兼容常见命名：
    - "...experts.2..."
    - "...expert.2..."
    - "...experts.2.0.weight"
    """
    result: Dict[int, List[str]] = {}

    for name in param_names:
        expert_id = _parse_expert_id(name)
        if expert_id is None:
            continue
        result.setdefault(int(expert_id), []).append(name)

    return result


def _parse_expert_id(name: str) -> Optional[int]:
    """
    从参数名或 module_name 中解析 expert id。
    """
    patterns = [
        r"(?:^|\.)(?:experts)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:expert)\.(\d+)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, str(name))
        if match is not None:
            return int(match.group(1))

    return None


def _resolve_param_names(
    state_dict: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """
    解析需要聚合的参数名。
    """
    if param_names is None:
        return list(state_dict.keys())

    names = list(param_names)
    for name in names:
        if name not in state_dict:
            raise KeyError(f"state_dict 中不存在参数：{name}")
    return names


def _average_per_expert_weights(
    per_expert_weights: Mapping[int, Mapping[int, float]],
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    把 per-expert weights 压成一份 client-level weights。

    这份权重只用于 AggregationResult.weights / 日志兼容，
    真正聚合 expert 参数时使用的是 per_expert_weights。
    """
    accum: Dict[int, float] = {int(update.client_id): 0.0 for update in client_updates}
    counts: Dict[int, int] = {int(update.client_id): 0 for update in client_updates}

    for weights in per_expert_weights.values():
        for client_id, weight in weights.items():
            client_id = int(client_id)
            accum[client_id] = accum.get(client_id, 0.0) + float(weight)
            counts[client_id] = counts.get(client_id, 0) + 1

    averaged: Dict[int, float] = {}
    for client_id in accum:
        if counts.get(client_id, 0) > 0:
            averaged[client_id] = accum[client_id] / float(counts[client_id])
        else:
            averaged[client_id] = 0.0

    total = sum(averaged.values())
    if total <= 0.0:
        return build_uniform_weights(client_updates)

    return normalize_weights(averaged)


def _softmax_dict(logits: Mapping[int, float]) -> Dict[int, float]:
    """
    稳定版 softmax，避免 exp(mu) 数值爆炸。
    """
    if len(logits) == 0:
        return {}

    finite_logits = {
        int(k): float(v)
        for k, v in logits.items()
        if math.isfinite(float(v))
    }
    if len(finite_logits) == 0:
        weight = 1.0 / float(len(logits))
        return {int(k): weight for k in logits.keys()}

    max_logit = max(finite_logits.values())
    exp_values: Dict[int, float] = {}
    total = 0.0

    for client_id, logit in finite_logits.items():
        value = math.exp(float(logit) - max_logit)
        exp_values[int(client_id)] = value
        total += value

    if total <= 0.0 or not math.isfinite(total):
        weight = 1.0 / float(len(finite_logits))
        return {int(k): weight for k in finite_logits.keys()}

    return {
        int(client_id): float(value) / float(total)
        for client_id, value in exp_values.items()
    }


def _median(values: Sequence[float]) -> float:
    """
    Python list 版 median。
    """
    clean_values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if len(clean_values) == 0:
        return 0.0

    n = len(clean_values)
    mid = n // 2

    if n % 2 == 1:
        return float(clean_values[mid])

    return 0.5 * (float(clean_values[mid - 1]) + float(clean_values[mid]))


def _mad(values: Sequence[float], center: Optional[float] = None) -> float:
    """
    Median Absolute Deviation。
    """
    clean_values = [float(v) for v in values if math.isfinite(float(v))]
    if len(clean_values) == 0:
        return 0.0

    if center is None:
        center = _median(clean_values)

    deviations = [abs(float(v) - float(center)) for v in clean_values]
    return _median(deviations)


def _sigmoid(x: float) -> float:
    """
    数值稳定 sigmoid。
    """
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(float(min_value), min(float(max_value), float(value))))


def _safe_mean(values: Sequence[float]) -> float:
    clean_values = [float(v) for v in values if math.isfinite(float(v))]
    if len(clean_values) == 0:
        return 0.0
    return float(sum(clean_values) / len(clean_values))


def _empty_state_counts() -> Dict[str, int]:
    return {
        "S1_current_good_history_good": 0,
        "S2_current_good_history_bad": 0,
        "S3_current_bad_history_good": 0,
        "S4_current_bad_history_bad": 0,
    }


def _state_name(current_good: bool, history_good: bool) -> str:
    """
    四种状态只用于日志诊断，不写死聚合权重。
    """
    if current_good and history_good:
        return "S1_current_good_history_good"
    if current_good and not history_good:
        return "S2_current_good_history_bad"
    if (not current_good) and history_good:
        return "S3_current_bad_history_good"
    return "S4_current_bad_history_bad"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的读取。

    支持：
    - cfg.get("history_wolf_kfac_score.rho", default)
    - cfg["history_wolf_kfac_score"]["rho"]
    - cfg.history_wolf_kfac_score.rho
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        value = cfg.get(key, None)
        if value is not None:
            return value

    current = cfg
    for part in str(key).split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        return default

    return current


def _cfg_bool(cfg: Any, key: str, default: bool) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower().strip() in {"1", "true", "yes", "y", "on"}
    return bool(value)