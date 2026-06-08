from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aggregation.base import Aggregator
from fl.types import AggregationResult, ClientUpdate
from models.param_groups import get_expert_id_from_name
from utils.state_dict_ops import check_finite_state_dict, clone_state_dict


class FisherOnlyExpertAggregator(Aggregator):
    """
    Fisher-only 专家聚合器。

    这个聚合器只用于 expert 参数组，不用于 non_expert 参数组。

    核心思想：
        客户端本地训练完成后，额外通过 fl/expert_kfac.py 统计每个 expert 的 K-FAC evidence：

            active_count:
                evidence pass 中 routed 到该 expert 的 token / sample 数。

            mean_A:
                expert Linear 层输入激活二阶统计的平均强度。

            mean_B:
                expert Linear 层反向梯度二阶统计的平均强度。

        服务端对每个 expert 单独计算客户端权重：

            score_i,e = active_count_i,e * mean_A_i,e * mean_B_i,e

        或者如果客户端 payload 已经提供 score，则优先使用：

            score_i,e = payload["score"]

        然后：

            logit_i,e = log(score_i,e + eps)
            weight_i,e = softmax_i(logit_i,e)

    注意：
        1. 这是 expert-wise 权重，不是 client-wise 全局单权重。
        2. client 0 对 expert 0 权重大，不代表它对 expert 1 也权重大。
        3. 所以这里必须重写 aggregate()，不能只实现 compute_weights()。
        4. non_expert 参数仍然应该交给 uniform / sample_weighted 聚合器。
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "fisher_only"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        返回客户端 Fisher score 总和。

        说明：
            fisher_only 的真实聚合权重是每个 expert 一套权重，
            因此这个函数不参与真正的 expert 聚合流程。

            这里保留实现只是为了满足 Aggregator 抽象接口，
            也方便某些外部调试代码直接调用 compute_weights()。

        返回：
            {
                client_id: sum_positive_expert_score
            }
        """
        self._validate_client_updates(client_updates)

        weights: Dict[int, float] = {}

        for update in client_updates:
            total_score = 0.0
            expert_payloads = _get_expert_payloads(update)

            for payload in expert_payloads.values():
                active_count = _safe_int(payload.get("active_count", 0), default=0)
                if active_count < self.min_active_count:
                    continue

                score = _extract_score(payload)
                if score > 0.0 and math.isfinite(score):
                    total_score += float(score)

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
        执行 fisher_only expert-wise delta 聚合。

        聚合方式：
            对每个 expert e：

                1. 收集每个客户端的 expert K-FAC evidence。
                2. 计算 score_i,e。
                3. 计算 logit_i,e = log(score_i,e + eps)。
                4. 在同一个 expert 内对客户端做 softmax。
                5. 用 weight_i,e 聚合该 expert 的参数 delta。

        参数：
            global_state:
                本轮聚合前的全局模型参数。

            client_updates:
                本轮参与训练的客户端更新。
                每个 update.extra 必须包含：
                    extra["expert_kfac"]

            param_names:
                当前聚合器负责的 expert 参数名。
                server.py 中一般传 self.param_groups.expert。

            base_state:
                聚合结果写入的基础 state_dict。
                在极致解耦流程中，通常是 non_expert 聚合后的 state_dict。
                这里会在 base_state 基础上更新 expert 参数。

            strict:
                如果为 True，缺少 expert_kfac / delta / 参数时直接报错。
                如果为 False，则尽量跳过缺失项。
        """
        if self.param_group_name != "expert":
            raise ValueError(
                "FisherOnlyExpertAggregator 只能用于 expert 参数组，"
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
                "fisher_only 没有收到任何 expert 参数名。"
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
            records = self._build_expert_records(
                expert_id=expert_id,
                client_updates=client_updates,
                strict=strict,
            )
            expert_record_map[int(expert_id)] = records

            if len(records) < self.min_valid_clients:
                # fallback=keep_global：
                # 不更新该 expert 参数，保留 base_state 中的旧值。
                if self.fallback != "keep_global":
                    raise ValueError(
                        f"不支持的 fisher_only fallback：{self.fallback}。"
                        "当前只支持 keep_global。"
                    )

                expert_weight_map[int(expert_id)] = {}
                expert_fallback_map[int(expert_id)] = True
                continue

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

        diagnostics = self._build_fisher_diagnostics(
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
        return float(_cfg_get(self.expert_fisher_cfg, "eps", 1.0e-8))

    @property
    def diagnostics_enabled(self) -> bool:
        """
        是否生成 fisher_only 诊断字段。

        注意：
            这个开关只控制 diagnostics 内容是否展开。
            是否打印到控制台 / train.log 由 server.py 中的 diagnostics_print 控制。
        """
        return bool(_cfg_get(self.expert_fisher_cfg, "diagnostics_enabled", True))

    @property
    def diagnostics_include_records(self) -> bool:
        """
        是否在 summary.json 里保存完整 client-expert records 和 expert_weights。

        第一版建议 false，避免日志和 summary.json 太大。
        """
        return bool(
            _cfg_get(
                self.expert_fisher_cfg,
                "diagnostics_include_records",
                False,
            )
        )

    def _build_expert_records(
        self,
        expert_id: int,
        client_updates: Sequence[ClientUpdate],
        strict: bool,
    ) -> List[Dict[str, Any]]:
        """
        为单个 expert 收集所有有效客户端的 Fisher record。

        record 字段：
            client_id
            active_count
            mean_A
            mean_B
            fisher_strength
            score
            logit
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
            logit = math.log(max(score, 0.0) + self.eps)

            is_valid = (
                active_count >= self.min_active_count
                and score > 0.0
                and math.isfinite(score)
                and math.isfinite(logit)
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
                    "logit": float(logit),
                }
            )

        return records

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

    def _build_fisher_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        param_names: Sequence[str],
        expert_weight_map: Mapping[int, Mapping[int, float]],
        expert_record_map: Mapping[int, Sequence[Mapping[str, Any]]],
        expert_fallback_map: Mapping[int, bool],
        avg_weights: Mapping[int, float],
    ) -> Dict[str, Any]:
        """
        构建 fisher_only 聚合诊断信息。

        第一版诊断目标：
            1. 看 Fisher evidence 是否有效。
            2. 看权重是不是接近 uniform。
            3. 看权重是不是被 active_count 支配。
            4. 看是否有 expert 因有效客户端不足而 keep_global。
        """
        num_experts = int(len(expert_record_map))
        num_fallback_experts = int(
            sum(1 for value in expert_fallback_map.values() if value)
        )

        if not self.diagnostics_enabled:
            return {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "fisher_diag_enabled": False,
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

            scores = [float(record["score"]) for record in records]
            logits = [float(record["logit"]) for record in records]
            active_counts = [float(record["active_count"]) for record in records]
            mean_As = [float(record["mean_A"]) for record in records]
            mean_Bs = [float(record["mean_B"]) for record in records]
            fisher_strengths = [
                float(record["fisher_strength"])
                for record in records
            ]

            record_weights = [
                float(weights.get(int(record["client_id"]), 0.0))
                for record in records
            ]

            top_client = None
            if len(weights) > 0:
                top_client = max(weights.items(), key=lambda item: item[1])[0]

            status_counts = self._count_expert_payload_status(
                expert_id=expert_id,
                client_updates=client_updates,
            )

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
                "zero_score_clients": int(status_counts["zero_score_clients"]),
                "zero_active_clients": int(status_counts["zero_active_clients"]),
                "nan_score_clients": int(status_counts["nan_score_clients"]),
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
                "logit_stats": _stat_dict(logits),
                "active_count_stats": _stat_dict(active_counts),
                "mean_A_stats": _stat_dict(mean_As),
                "mean_B_stats": _stat_dict(mean_Bs),
                "fisher_strength_stats": _stat_dict(fisher_strengths),
                "score_cv": _coefficient_of_variation(scores),
                "active_count_cv": _coefficient_of_variation(active_counts),
                "fisher_strength_cv": _coefficient_of_variation(fisher_strengths),
                "score_active_corr": _pearson_corr(scores, active_counts),
                "score_fisher_corr": _pearson_corr(scores, fisher_strengths),
                "weight_active_corr": _pearson_corr(record_weights, active_counts),
                "weight_fisher_corr": _pearson_corr(
                    record_weights,
                    fisher_strengths,
                ),
            }

            if include_records:
                # 详细 records 第一版默认不保存。
                # 需要排查单个 client-expert 时，再在 yaml 中打开。
                expert_diag["weights"] = {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                }
                expert_diag["records"] = [
                    {
                        "client_id": int(record["client_id"]),
                        "active_count": int(record["active_count"]),
                        "mean_A": float(record["mean_A"]),
                        "mean_B": float(record["mean_B"]),
                        "fisher_strength": float(record["fisher_strength"]),
                        "score": float(record["score"]),
                        "logit": float(record["logit"]),
                        "weight": float(weights.get(int(record["client_id"]), 0.0)),
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
            "fisher_diag_enabled": True,
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
                [
                    diag.get("weight_entropy_norm", 0.0)
                    for diag in non_fallback_diags
                ]
            ),
            "mean_effective_clients": _mean_clean(
                [
                    diag.get("effective_clients", 0.0)
                    for diag in non_fallback_diags
                ]
            ),
            "mean_weight_max": _mean_clean(
                [
                    diag.get("weight_max", 0.0)
                    for diag in non_fallback_diags
                ]
            ),
            "mean_score_cv": _mean_clean(
                [diag.get("score_cv", 0.0) for diag in all_expert_diags]
            ),
            "mean_active_count_cv": _mean_clean(
                [diag.get("active_count_cv", 0.0) for diag in all_expert_diags]
            ),
            "mean_fisher_strength_cv": _mean_clean(
                [
                    diag.get("fisher_strength_cv", 0.0)
                    for diag in all_expert_diags
                ]
            ),
            "mean_weight_active_corr": _mean_clean(
                [
                    diag.get("weight_active_corr", 0.0)
                    for diag in non_fallback_diags
                ]
            ),
            "mean_weight_fisher_corr": _mean_clean(
                [
                    diag.get("weight_fisher_corr", 0.0)
                    for diag in non_fallback_diags
                ]
            ),
            "expert_diagnostics": expert_diagnostics,
        }

        if include_records:
            # AggregationResult.weights 只能是一套 client 权重。
            # fisher_only 的真实权重在 expert_weights 里。
            # 这里默认不保存，防止 summary.json 太大。
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

        注意：
            _build_expert_records() 只保留有效 records。
            这个函数用于统计被过滤掉的原因，比如 active_count=0 或 score=0。
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
            score = _extract_score(payload)

            is_nan_score = not math.isfinite(score)
            is_zero_score = score <= 0.0
            is_zero_active = active_count <= 0

            if is_nan_score:
                nan_score_clients += 1

            if is_zero_score:
                zero_score_clients += 1

            if is_zero_active:
                zero_active_clients += 1

            if (
                active_count < self.min_active_count
                or is_zero_score
                or is_nan_score
            ):
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
    """
    把 expert 参数名按照 expert_id 分组。
    """
    result: Dict[int, List[str]] = {}

    for name in param_names:
        expert_id = get_expert_id_from_name(name)
        if expert_id is None:
            # fisher_only 只处理 expert 参数。
            # 如果 server 误传了 non_expert 参数，这里直接跳过，避免污染。
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

    推荐格式是 fl/expert_kfac.py 返回的：
        {
            "experts": {
                expert_id: {...}
            },
            "meta": {...}
        }
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

    优先级：
        1. payload["score"]
        2. active_count * fisher_strength
        3. active_count * mean_A * mean_B
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
    """
    对单个 expert 的客户端 logit 做 softmax。
    """
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
    """
    计算权重熵。

    权重越均匀，熵越大。
    某个客户端支配 expert 时，熵会变小。
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
        接近 1：权重接近 uniform。
        接近 0：某个客户端强烈支配。
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
        接近参与客户端数：权重接近 uniform。
        接近 1：单个客户端支配。
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
    """
    返回 top1_weight、top2_weight、top1_gap。
    """
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
    """
    计算变异系数 CV = std / abs(mean)。

    用途：
        判断 score / active_count / fisher_strength 是否有区分度。
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

    var = sum((value - mean) ** 2 for value in clean_values) / len(clean_values)
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
        weight_active_corr:
            判断权重是否主要由 active_count 控制。

        weight_fisher_corr:
            判断权重是否真的受 Fisher 强度影响。
    """
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
    """
    对有限数值求平均。
    """
    clean_values = [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]

    if len(clean_values) == 0:
        return 0.0

    return float(sum(clean_values) / len(clean_values))


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
        cfg.get(key, default)
        getattr(cfg, key, default)
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)