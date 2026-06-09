from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from aggregation.base import Aggregator
from fl.types import ClientUpdate


class FisherOnlyGlobalAggregator(Aggregator):
    """
    纯 FL 版 Fisher-only 聚合器。

    这个聚合器用于“整模型 client-wise 聚合”，不是 MoE expert-wise 聚合。

    和原来的 FisherOnlyExpertAggregator 的区别：
    - 原版 fisher_only：
        client i + expert e -> score_i,e
        每个 expert 单独一套客户端权重
        读取 update.extra["expert_kfac"]

    - 当前 fisher_only_global：
        client i -> score_i
        整个模型共用一套客户端权重
        读取 update.extra["global_fisher"]

    推荐配合 fl/full_model_fisher.py 使用。客户端本地训练后应写入：
        update.extra["global_fisher"] = {
            "fisher_strength": ...,
            "num_samples": ...,
            "score": ...,
            "meta": {...},
        }

    聚合权重：
        score_i = num_samples_i * fisher_strength_i

    如果 payload 里已经有 score，则优先使用 payload["score"]。

    聚合公式仍然走 Aggregator 基类：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    @property
    def method_name(self) -> str:
        """
        返回当前聚合方法名称。

        注意：
        - fisher_only 是原来的 expert-wise 版本。
        - fisher_only_global 是当前 pure-FL client-wise 版本。
        """
        return "fisher_only_global"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算纯 FL Fisher-only 的客户端原始权重。

        返回的是 raw weight，后续会由 Aggregator.aggregate() 统一归一化。

        规则：
            raw_w_i = score_i
                    = num_samples_i * fisher_strength_i

        如果有效客户端数量不足，则 fallback 到 uniform raw weights：
            raw_w_i = 1.0
        """
        self._validate_client_updates(client_updates)

        records = self._build_records(client_updates)

        if len(records) < self.min_valid_clients:
            return _build_uniform_raw_weights(client_updates)

        raw_weights = {
            int(record["client_id"]): float(record["score"])
            for record in records
        }

        if _sum_positive(raw_weights.values()) <= self.eps:
            return _build_uniform_raw_weights(client_updates)

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
              diagnostics_enabled: true
        """
        return _cfg_get(self.cfg, "full_model_fisher", {})

    @property
    def eps(self) -> float:
        """
        数值稳定项。
        """
        return float(
            _cfg_get(
                self.full_model_fisher_cfg,
                "eps",
                1.0e-8,
            )
        )

    @property
    def min_valid_clients(self) -> int:
        """
        至少需要多少个有效客户端 Fisher record。

        如果低于这个数量，说明 evidence 不足，直接退化为 uniform。
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
    def diagnostics_enabled(self) -> bool:
        """
        是否生成详细诊断字段。

        注意：
        这里只控制 diagnostics 内容是否展开。
        是否打印到控制台或 train.log，由 server.py 的日志逻辑控制。
        """
        return bool(
            _cfg_get(
                self.full_model_fisher_cfg,
                "diagnostics_enabled",
                True,
            )
        )

    def _build_records(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Sequence[Dict[str, Any]]:
        """
        从 client_updates 中提取纯 FL Fisher records。

        每个有效 record 包含：
            client_id
            num_samples
            fisher_strength
            score
            log_score
            source
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
                update=update,
                payload=payload,
                fisher_strength=fisher_strength,
                num_samples=num_samples,
            )

            is_valid = (
                num_samples > 0
                and fisher_strength > 0.0
                and score > 0.0
                and math.isfinite(fisher_strength)
                and math.isfinite(score)
            )

            if not is_valid:
                continue

            records.append(
                {
                    "client_id": int(update.client_id),
                    "num_samples": int(num_samples),
                    "fisher_strength": float(fisher_strength),
                    "score": float(score),
                    "log_score": float(math.log(score + self.eps)),
                    "source": str(payload.get("source", "global_fisher")),
                    "meta": payload.get("meta", {}),
                }
            )

        return records

    def build_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        param_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建 fisher_only_global 诊断信息。

        诊断目标：
        1. 确认是否读到了 global_fisher。
        2. 确认 Fisher score 是否有区分度。
        3. 确认最终权重是否接近 uniform。
        4. 判断权重主要被 num_samples 控制，还是 fisher_strength 控制。
        """
        param_count = None
        if param_names is not None:
            param_count = len(list(param_names))

        try:
            records = list(self._build_records(client_updates))
        except Exception as exc:
            return {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "diagnostics_enabled": bool(self.diagnostics_enabled),
                "num_clients": int(len(client_updates)),
                "param_count": param_count,
                "diagnostics_error": str(exc),
                "weights": {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                },
            }

        missing_clients = []
        invalid_clients = []
        record_by_client = {
            int(record["client_id"]): record
            for record in records
        }

        for update in client_updates:
            client_id = int(update.client_id)

            if "global_fisher" not in update.extra:
                missing_clients.append(client_id)
                continue

            if client_id not in record_by_client:
                invalid_clients.append(client_id)

        if not self.diagnostics_enabled:
            return {
                "method": self.method_name,
                "param_group": self.param_group_name,
                "diagnostics_enabled": False,
                "num_clients": int(len(client_updates)),
                "param_count": param_count,
                "valid_clients": int(len(records)),
                "missing_clients": int(len(missing_clients)),
                "invalid_clients": int(len(invalid_clients)),
                "weights": {
                    int(client_id): float(weight)
                    for client_id, weight in weights.items()
                },
            }

        scores = [float(record["score"]) for record in records]
        log_scores = [float(record["log_score"]) for record in records]
        fisher_strengths = [
            float(record["fisher_strength"])
            for record in records
        ]
        num_samples = [
            float(record["num_samples"])
            for record in records
        ]
        record_weights = [
            float(weights.get(int(record["client_id"]), 0.0))
            for record in records
        ]

        top_client = None
        if len(weights) > 0:
            top_client = max(weights.items(), key=lambda item: item[1])[0]

        top1_weight, top2_weight, top1_gap = _top_weight_stats(weights)

        diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "diagnostics_enabled": True,
            "num_clients": int(len(client_updates)),
            "param_count": param_count,
            "valid_clients": int(len(records)),
            "missing_clients": int(len(missing_clients)),
            "invalid_clients": int(len(invalid_clients)),
            "min_valid_clients": int(self.min_valid_clients),
            "fallback_to_uniform": bool(len(records) < self.min_valid_clients),
            "top_client": int(top_client) if top_client is not None else None,
            "weight_entropy": float(_weight_entropy(weights)),
            "weight_entropy_norm": float(_weight_entropy_norm(weights)),
            "effective_clients": float(_effective_clients(weights)),
            "weight_min": float(min(weights.values())) if len(weights) > 0 else 0.0,
            "weight_max": float(max(weights.values())) if len(weights) > 0 else 0.0,
            "top1_weight": float(top1_weight),
            "top2_weight": float(top2_weight),
            "top1_gap": float(top1_gap),
            "score_stats": _stat_dict(scores),
            "log_score_stats": _stat_dict(log_scores),
            "fisher_strength_stats": _stat_dict(fisher_strengths),
            "num_samples_stats": _stat_dict(num_samples),
            "score_cv": float(_coefficient_of_variation(scores)),
            "fisher_strength_cv": float(
                _coefficient_of_variation(fisher_strengths)
            ),
            "num_samples_cv": float(_coefficient_of_variation(num_samples)),
            "score_num_samples_corr": float(
                _pearson_corr(scores, num_samples)
            ),
            "score_fisher_corr": float(
                _pearson_corr(scores, fisher_strengths)
            ),
            "weight_num_samples_corr": float(
                _pearson_corr(record_weights, num_samples)
            ),
            "weight_fisher_corr": float(
                _pearson_corr(record_weights, fisher_strengths)
            ),
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in weights.items()
            },
        }

        if bool(
            _cfg_get(
                self.full_model_fisher_cfg,
                "diagnostics_include_records",
                False,
            )
        ):
            diagnostics["missing_client_ids"] = [
                int(client_id)
                for client_id in missing_clients
            ]
            diagnostics["invalid_client_ids"] = [
                int(client_id)
                for client_id in invalid_clients
            ]
            diagnostics["records"] = [
                {
                    "client_id": int(record["client_id"]),
                    "num_samples": int(record["num_samples"]),
                    "fisher_strength": float(record["fisher_strength"]),
                    "score": float(record["score"]),
                    "log_score": float(record["log_score"]),
                    "weight": float(
                        weights.get(int(record["client_id"]), 0.0)
                    ),
                    "source": str(record.get("source", "global_fisher")),
                }
                for record in records
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
    update: ClientUpdate,
    payload: Mapping[str, Any],
    fisher_strength: float,
    num_samples: int,
) -> float:
    """
    提取 Fisher-only score。

    优先级：
    1. payload["score"]
    2. num_samples * fisher_strength

    这样可以和 fl/full_model_fisher.py 的 to_payload() 对齐。
    """
    if "score" in payload:
        return _safe_float(payload.get("score", 0.0), default=0.0)

    return float(num_samples) * float(fisher_strength)


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
    判断 score / fisher_strength / num_samples 是否有区分度。
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
    - cfg.get(key, default)
    - getattr(cfg, key, default)

    另外兼容简单 dotted key：
        _cfg_get(cfg, "full_model_fisher.eps", 1e-8)
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