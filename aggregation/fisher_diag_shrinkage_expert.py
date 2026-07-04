from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from aggregation.base import Aggregator, build_sample_weights
from fl.types import AggregationResult, ClientUpdate
from models.param_groups import get_expert_id_from_name
from utils.state_dict_ops import (
    check_finite_state_dict,
    clone_state_dict,
    normalize_weights,
)


@dataclass(frozen=True)
class _ExpertClientEntry:
    """
    某个客户端参与某个 expert 聚合时所需的轻量信息。

    update:
        客户端上传的 ClientUpdate。
    usage:
        客户端上传的路由比例：

            u_{i,e} = routed_samples_{i,e} / num_evidence_samples_i

        对 top-k 路由，所有 expert 的 usage 之和约等于 k，
        而不是 1；单个 expert 的 usage 通常位于 [0, 1]。
    raw_weight:
        当前客户端对 expert e 的未归一化有效权重：

            rho_{i,e} = n_i * (u_{i,e}) ** beta

    fisher_diag:
        客户端上传的 expert diagonal empirical Fisher 字典。
        key 必须与 model_delta / global_state 的完整参数名一致。
    """

    update: ClientUpdate
    usage: float
    raw_weight: float
    fisher_diag: Mapping[str, torch.Tensor]


class FisherDiagShrinkageExpertAggregator(Aggregator):
    """
    路由感知的 expert 聚合 + diagonal Fisher 参数级稳定收缩。

    该聚合器只负责 expert 参数，不负责 non-expert 参数。
    non-expert 参数应继续由 server 中独立的 non_expert aggregator 聚合，
    例如配置为 uniform：

        agg.non_expert.method = "uniform"
        agg.expert.method = "fisher_diag_shrinkage_expert"

    ------------------------------------------------------------------
    一、客户端权重
    ------------------------------------------------------------------

    对客户端 i 和 expert e，客户端上传路由比例：

        u_{i,e} = c_{i,e} / m_i

    其中 c_{i,e} 是 evidence 数据中路由到 expert e 的样本数，
    m_i 是 Fisher evidence 样本总数。

    服务端计算 expert-specific 有效权重：

        rho_{i,e} = n_i * (u_{i,e}) ** beta

    其中 n_i 使用 ClientUpdate.num_samples，也就是客户端训练集样本数。
    beta 默认等于 1。

    ------------------------------------------------------------------
    二、普通 expert delta 聚合
    ------------------------------------------------------------------

        alpha_{i,e} = rho_{i,e} / sum_k rho_{k,e}

        Delta_e = sum_i alpha_{i,e} * Delta_{i,e}

    Fisher 不参与“哪个客户端占多少权重”的决定；客户端权重只由
    样本量和路由比例决定。

    ------------------------------------------------------------------
    三、Fisher 聚合与稳定收缩
    ------------------------------------------------------------------

    使用与 expert delta 完全相同的 alpha_{i,e} 聚合 Fisher：

        f_bar_e = sum_i alpha_{i,e} * f_{i,e}

    对整个 expert 的所有参数统一计算 Fisher 均值：

        f_hat_e = f_bar_e / (mean(f_bar_e) + eps)

    然后只对已经聚合完成的 expert update 做逐参数收缩：

        scale_e = 1 / (1 + lambda * f_hat_e)

        theta_e^{t+1}
            = theta_e^t + server_lr * scale_e * Delta_e

    当 Fisher 非负、lambda >= 0 且 0 < server_lr <= 1 时：

        0 < scale_e <= 1

    因此该方法不会把任何参数维度的原始聚合更新放大，只会保留或压缩。

    ------------------------------------------------------------------
    四、客户端上传格式
    ------------------------------------------------------------------

    本文件与修改后的 fl/client.py 和 fl/fisher_diag.py 对应，读取：

        update.extra["expert_fisher_diag"] = {
            "expert_usage": {
                expert_id: usage,
                ...
            },
            "diag": {
                "moe_head.experts.0.fc1.weight": Tensor,
                ...
            },
            ...
        }

    注意：diag 中的 Fisher 已经由客户端按该 expert 的 routed samples
    做过条件平均。服务端不要再次除以 routed count。
    """

    @property
    def method_name(self) -> str:
        """返回配置文件中使用的聚合方法名称。"""

        return "fisher_diag_shrinkage_expert"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        为了满足 Aggregator 统一接口，返回按客户端样本数计算的权重。

        注意：
        这里返回的权重不是实际的 expert-specific 权重。
        真正用于每个 expert 的权重是：

            rho_{i,e} = n_i * usage_{i,e} ** beta

        因为不同 expert 的 usage 不同，不可能用一套全局 weights 完整表示。
        每个 expert 的真实权重会写入 diagnostics["expert_weights"]。
        """

        return build_sample_weights(client_updates)

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        聚合 expert 参数并执行 diagonal Fisher 稳定收缩。

        参数：
        global_state:
            本轮聚合前的完整全局模型 state_dict。
            expert 更新必须以这里的旧全局 expert 参数为起点。
        client_updates:
            本轮参与客户端上传的 ClientUpdate 列表。
        param_names:
            server 传入的 expert 参数名列表。
        base_state:
            上一步 non-expert 聚合完成后的完整 state_dict。
            本聚合器会克隆它，并且只覆盖 expert 参数，因此不会破坏
            已经完成的 uniform non-expert 聚合结果。
        strict:
            True 时，缺失 Fisher payload、缺失参数、shape 不一致等问题
            直接报错；False 时跳过对应无效客户端。
        """

        self._validate_client_updates(client_updates)

        if self.param_group_name != "expert":
            raise ValueError(
                "fisher_diag_shrinkage_expert 只能用于 expert 参数聚合。"
            )

        beta = float(_cfg_get(self.cfg, "fisher_diag.beta", 1.0))
        server_lr = float(
            _cfg_get(self.cfg, "fisher_diag.server_lr", 1.0)
        )
        shrinkage_lambda = float(
            _cfg_get(
                self.cfg,
                "fisher_diag.shrinkage_lambda",
                1.0,
            )
        )
        eps = float(_cfg_get(self.cfg, "fisher_diag.eps", 1.0e-12))
        min_usage = float(
            _cfg_get(self.cfg, "fisher_diag.min_usage", 0.0)
        )
        usage_tolerance = float(
            _cfg_get(
                self.cfg,
                "fisher_diag.usage_tolerance",
                1.0e-6,
            )
        )
        fallback = str(
            _cfg_get(
                self.cfg,
                "fisher_diag.fallback",
                "keep_global",
            )
        ).lower().strip()

        _validate_hyperparameters(
            beta=beta,
            server_lr=server_lr,
            shrinkage_lambda=shrinkage_lambda,
            eps=eps,
            min_usage=min_usage,
            usage_tolerance=usage_tolerance,
            fallback=fallback,
        )

        target_param_names = _resolve_target_param_names(
            global_state=global_state,
            param_names=param_names,
        )
        expert_param_names = _group_param_names_by_expert(
            target_param_names
        )

        if not expert_param_names:
            raise ValueError(
                "param_names 中没有可识别的 expert 参数；"
                "请检查参数名是否包含 experts.<id>。"
            )

        # server 先完成 non-expert 聚合，再把结果作为 base_state 传进来。
        # 因此这里必须从 base_state 克隆，才能保留 non-expert 的新值。
        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            _validate_base_state(
                global_state=global_state,
                base_state=base_state,
                target_param_names=target_param_names,
            )
            new_state_dict = clone_state_dict(base_state)

        # AggregationResult.weights 只能存一套客户端权重。
        # 这里沿用项目现有 K-FAC aggregator 的兼容做法，放样本数权重；
        # 每个 expert 的真实路由权重放到 diagnostics 中。
        compatibility_weights = normalize_weights(
            self.compute_weights(client_updates)
        )

        diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": len(target_param_names),
            "beta": beta,
            "server_lr": server_lr,
            "shrinkage_lambda": shrinkage_lambda,
            "eps": eps,
            "min_usage": min_usage,
            "fallback": fallback,
            "weights_note": (
                "AggregationResult.weights 是接口兼容用的样本数权重；"
                "真实 expert-specific 权重见 expert_weights。"
            ),
            "expert_weights": {},
            "expert_stats": {},
            "skipped_experts": [],
        }

        for expert_id, names in sorted(expert_param_names.items()):
            entries = _collect_expert_entries(
                expert_id=expert_id,
                expert_param_names=names,
                client_updates=client_updates,
                global_state=global_state,
                beta=beta,
                min_usage=min_usage,
                usage_tolerance=usage_tolerance,
                strict=strict,
            )

            # 本轮没有任何客户端路由到该 expert。
            # 此时不应拿未使用该 expert 的客户端做平均，而是保留旧全局 expert。
            if not entries:
                if fallback != "keep_global":
                    raise RuntimeError(
                        f"expert {expert_id} 没有有效客户端，"
                        f"但 fallback={fallback} 不受支持。"
                    )

                for name in names:
                    new_state_dict[name] = global_state[name].detach().clone()

                diagnostics["skipped_experts"].append(expert_id)
                diagnostics["expert_stats"][str(expert_id)] = {
                    "status": "kept_global_no_valid_client",
                    "num_clients": 0,
                    "total_raw_weight": 0.0,
                }
                continue

            raw_weight_sum = sum(entry.raw_weight for entry in entries)
            if not math.isfinite(raw_weight_sum) or raw_weight_sum <= 0.0:
                raise ValueError(
                    f"expert {expert_id} 的有效权重总和非法："
                    f"{raw_weight_sum}。"
                )

            expert_weights = {
                int(entry.update.client_id): (
                    float(entry.raw_weight) / raw_weight_sum
                )
                for entry in entries
            }

            diagnostics["expert_weights"][str(expert_id)] = {
                str(client_id): float(weight)
                for client_id, weight in expert_weights.items()
            }

            aggregated_delta: Dict[str, torch.Tensor] = {}
            aggregated_fisher: Dict[str, torch.Tensor] = {}

            # delta 和 Fisher 必须使用同一套 alpha_{i,e}。
            for name in names:
                reference = global_state[name]

                if not reference.is_floating_point():
                    # 当前 expert 是 Linear 权重/偏置，正常不会进入这里。
                    # 对非浮点 buffer 不做算术，直接保留旧全局值。
                    new_state_dict[name] = reference.detach().clone()
                    continue

                delta_sum = torch.zeros_like(
                    reference,
                    dtype=torch.float32,
                    device=reference.device,
                )
                fisher_sum = torch.zeros_like(
                    reference,
                    dtype=torch.float32,
                    device=reference.device,
                )

                for entry in entries:
                    client_id = int(entry.update.client_id)
                    alpha = expert_weights[client_id]

                    client_delta = entry.update.model_delta[name].detach().to(
                        device=reference.device,
                        dtype=torch.float32,
                    )
                    client_fisher = entry.fisher_diag[name].detach().to(
                        device=reference.device,
                        dtype=torch.float32,
                    )

                    delta_sum.add_(client_delta, alpha=alpha)
                    fisher_sum.add_(client_fisher, alpha=alpha)

                # 客户端 Fisher 是逐样本梯度平方，理论上必须非负。
                # 前面已经严格检查；这里 clamp_min 只用于消除极小数值误差。
                aggregated_delta[name] = delta_sum
                aggregated_fisher[name] = fisher_sum.clamp_min(0.0)

            floating_names = [
                name
                for name in names
                if name in aggregated_fisher
            ]
            if not floating_names:
                raise RuntimeError(
                    f"expert {expert_id} 没有可聚合的浮点参数。"
                )

            # 对“整个 expert”统一计算均值，而不是每层分别归一化。
            # 这样能够保留 expert 内不同层之间的相对 Fisher 尺度。
            fisher_total = sum(
                aggregated_fisher[name].to(torch.float64).sum()
                for name in floating_names
            )
            fisher_numel = sum(
                aggregated_fisher[name].numel()
                for name in floating_names
            )
            mean_fisher = float(
                (fisher_total / float(fisher_numel)).item()
            )

            if not math.isfinite(mean_fisher) or mean_fisher < 0.0:
                raise ValueError(
                    f"expert {expert_id} 的 mean_fisher 非法："
                    f"{mean_fisher}。"
                )

            scale_min = math.inf
            scale_max = -math.inf
            scale_sum = 0.0
            scale_numel = 0
            delta_l2_sq = 0.0
            applied_delta_l2_sq = 0.0

            for name in floating_names:
                reference = global_state[name]
                fisher_hat = aggregated_fisher[name] / (
                    mean_fisher + eps
                )

                # 稳定版本：只压缩，不放大。
                # Fisher 越大，该参数方向的更新保留比例越小。
                scale = torch.reciprocal(
                    1.0 + shrinkage_lambda * fisher_hat
                )

                # 按数学公式本来就位于 (0, 1]；clamp 是额外数值保护。
                scale = scale.clamp(min=0.0, max=1.0)

                original_delta = aggregated_delta[name]
                applied_delta = server_lr * scale * original_delta

                updated_value = (
                    reference.detach().to(torch.float32)
                    + applied_delta
                )
                new_state_dict[name] = updated_value.to(
                    dtype=reference.dtype,
                    device=reference.device,
                )

                current_scale_min = float(scale.min().item())
                current_scale_max = float(scale.max().item())
                scale_min = min(scale_min, current_scale_min)
                scale_max = max(scale_max, current_scale_max)
                scale_sum += float(
                    scale.to(torch.float64).sum().item()
                )
                scale_numel += int(scale.numel())

                delta_l2_sq += float(
                    original_delta.to(torch.float64).square().sum().item()
                )
                applied_delta_l2_sq += float(
                    applied_delta.to(torch.float64).square().sum().item()
                )

            scale_mean = scale_sum / float(scale_numel)
            original_delta_l2 = math.sqrt(max(delta_l2_sq, 0.0))
            applied_delta_l2 = math.sqrt(
                max(applied_delta_l2_sq, 0.0)
            )

            diagnostics["expert_stats"][str(expert_id)] = {
                "status": "updated",
                "num_clients": len(entries),
                "client_ids": [
                    int(entry.update.client_id)
                    for entry in entries
                ],
                "total_raw_weight": float(raw_weight_sum),
                "mean_fisher": float(mean_fisher),
                "scale_min": float(scale_min),
                "scale_mean": float(scale_mean),
                "scale_max": float(scale_max),
                "original_delta_l2": float(original_delta_l2),
                "applied_delta_l2": float(applied_delta_l2),
                "applied_to_original_l2_ratio": (
                    float(applied_delta_l2 / original_delta_l2)
                    if original_delta_l2 > 0.0
                    else 0.0
                ),
                "client_usage": {
                    str(int(entry.update.client_id)): float(entry.usage)
                    for entry in entries
                },
                "client_raw_weights": {
                    str(int(entry.update.client_id)): float(
                        entry.raw_weight
                    )
                    for entry in entries
                },
            }

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=target_param_names,
        )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=compatibility_weights,
            diagnostics=diagnostics,
        )


def _collect_expert_entries(
    expert_id: int,
    expert_param_names: Sequence[str],
    client_updates: Sequence[ClientUpdate],
    global_state: Mapping[str, torch.Tensor],
    beta: float,
    min_usage: float,
    usage_tolerance: float,
    strict: bool,
) -> List[_ExpertClientEntry]:
    """
    收集所有实际使用 expert e 且 payload 完整的客户端。

    usage == 0 的客户端不参与该 expert 聚合，也不要求它上传该 expert 的
    Fisher tensor；usage > 0 时，必须同时存在该 expert 所有参数的 delta
    和 diagonal Fisher。
    """

    entries: List[_ExpertClientEntry] = []

    for update in client_updates:
        payload = update.extra.get("expert_fisher_diag")

        if not isinstance(payload, Mapping):
            if strict:
                raise KeyError(
                    f"客户端 {update.client_id} 缺少 "
                    "extra['expert_fisher_diag']。"
                )
            continue

        usage_map = payload.get("expert_usage")
        fisher_diag = payload.get("diag")

        if not isinstance(usage_map, Mapping):
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 "
                    "expert_fisher_diag['expert_usage'] 不是 Mapping。"
                )
            continue

        if not isinstance(fisher_diag, Mapping):
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 "
                    "expert_fisher_diag['diag'] 不是 Mapping。"
                )
            continue

        usage_value = _mapping_get_by_int_or_str_key(
            usage_map,
            expert_id,
            default=0.0,
        )

        try:
            usage = float(usage_value)
        except (TypeError, ValueError) as error:
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 expert {expert_id} "
                    f"usage 无法转成 float：{usage_value!r}。"
                ) from error
            continue

        if not math.isfinite(usage):
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 expert {expert_id} "
                    f"usage 不是有限数：{usage}。"
                )
            continue

        if usage < -usage_tolerance or usage > 1.0 + usage_tolerance:
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 expert {expert_id} "
                    f"usage={usage} 超出合理范围 [0, 1]。"
                )
            continue

        # 消除浮点统计中极小的负值或略大于 1 的误差。
        usage = min(max(usage, 0.0), 1.0)

        # 没有路由到该 expert 的客户端不参与当前 expert 的聚合。
        if usage <= min_usage:
            continue

        valid = _validate_client_expert_tensors(
            update=update,
            fisher_diag=fisher_diag,
            expert_id=expert_id,
            expert_param_names=expert_param_names,
            global_state=global_state,
            strict=strict,
        )
        if not valid:
            continue

        raw_weight = float(update.num_samples) * (usage ** beta)
        if not math.isfinite(raw_weight) or raw_weight <= 0.0:
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 对 expert {expert_id} 的 "
                    f"raw_weight 非法：{raw_weight}。"
                )
            continue

        entries.append(
            _ExpertClientEntry(
                update=update,
                usage=usage,
                raw_weight=raw_weight,
                fisher_diag=fisher_diag,
            )
        )

    return entries


def _validate_client_expert_tensors(
    update: ClientUpdate,
    fisher_diag: Mapping[str, torch.Tensor],
    expert_id: int,
    expert_param_names: Sequence[str],
    global_state: Mapping[str, torch.Tensor],
    strict: bool,
) -> bool:
    """检查客户端某个 expert 的 delta 和 Fisher 是否完整、有限且同形状。"""

    for name in expert_param_names:
        reference = global_state[name]

        if name not in update.model_delta:
            if strict:
                raise KeyError(
                    f"客户端 {update.client_id} 的 model_delta 缺少 "
                    f"expert {expert_id} 参数：{name}。"
                )
            return False

        delta = update.model_delta[name]
        if not torch.is_tensor(delta):
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 delta {name} 不是 Tensor。"
                )
            return False

        if tuple(delta.shape) != tuple(reference.shape):
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 delta {name} shape 错误："
                    f"{tuple(delta.shape)} != {tuple(reference.shape)}。"
                )
            return False

        if delta.is_floating_point() and not torch.isfinite(delta).all():
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 delta {name} 包含 NaN/Inf。"
                )
            return False

        # 非浮点 expert buffer 不需要 Fisher，聚合时会直接保留全局值。
        if not reference.is_floating_point():
            continue

        if name not in fisher_diag:
            if strict:
                raise KeyError(
                    f"客户端 {update.client_id} 的 diagonal Fisher 缺少 "
                    f"expert {expert_id} 参数：{name}。"
                )
            return False

        fisher = fisher_diag[name]
        if not torch.is_tensor(fisher):
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 Fisher {name} 不是 Tensor。"
                )
            return False

        if tuple(fisher.shape) != tuple(reference.shape):
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 Fisher {name} shape 错误："
                    f"{tuple(fisher.shape)} != {tuple(reference.shape)}。"
                )
            return False

        if not fisher.is_floating_point():
            if strict:
                raise TypeError(
                    f"客户端 {update.client_id} 的 Fisher {name} "
                    "必须是浮点 Tensor。"
                )
            return False

        if not torch.isfinite(fisher).all():
            if strict:
                raise ValueError(
                    f"客户端 {update.client_id} 的 Fisher {name} "
                    "包含 NaN/Inf。"
                )
            return False

        if torch.any(fisher < 0):
            if strict:
                min_value = float(fisher.min().item())
                raise ValueError(
                    f"客户端 {update.client_id} 的 Fisher {name} 出现负值："
                    f"min={min_value}。逐样本梯度平方应当非负。"
                )
            return False

    return True


def _resolve_target_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """解析并检查本聚合器需要处理的 expert 参数名。"""

    if param_names is None:
        names = [
            name
            for name in global_state.keys()
            if get_expert_id_from_name(name) is not None
        ]
    else:
        names = list(param_names)

    if not names:
        raise ValueError("expert param_names 不能为空。")

    if len(names) != len(set(names)):
        raise ValueError("expert param_names 中存在重复参数名。")

    for name in names:
        if name not in global_state:
            raise KeyError(f"global_state 缺少参数：{name}")
        if get_expert_id_from_name(name) is None:
            raise ValueError(
                f"参数 {name} 不属于 expert，不能交给 "
                "fisher_diag_shrinkage_expert 聚合。"
            )

    return names


def _group_param_names_by_expert(
    param_names: Sequence[str],
) -> Dict[int, List[str]]:
    """按照参数名中的 experts.<id> 将参数分组。"""

    result: Dict[int, List[str]] = {}

    for name in param_names:
        expert_id = get_expert_id_from_name(name)
        if expert_id is None:
            raise ValueError(f"无法从 expert 参数名解析 expert id：{name}")
        result.setdefault(int(expert_id), []).append(name)

    return result


def _validate_base_state(
    global_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    target_param_names: Sequence[str],
) -> None:
    """
    检查 non-expert 聚合器传入的 base_state 是否能作为完整结果基础。
    """

    missing_names = [
        name
        for name in global_state.keys()
        if name not in base_state
    ]
    if missing_names:
        raise KeyError(
            "base_state 不是完整 state_dict，缺少参数："
            f"{missing_names[:20]}"
        )

    for name in target_param_names:
        if tuple(base_state[name].shape) != tuple(global_state[name].shape):
            raise ValueError(
                f"base_state 参数 {name} shape 与 global_state 不一致："
                f"{tuple(base_state[name].shape)} != "
                f"{tuple(global_state[name].shape)}。"
            )


def _validate_hyperparameters(
    beta: float,
    server_lr: float,
    shrinkage_lambda: float,
    eps: float,
    min_usage: float,
    usage_tolerance: float,
    fallback: str,
) -> None:
    """集中检查配置，避免训练若干轮后才因非法超参数失败。"""

    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(
            "fisher_diag.beta 必须位于 [0, 1]，"
            f"当前值：{beta}。"
        )

    if not math.isfinite(server_lr) or not 0.0 < server_lr <= 1.0:
        raise ValueError(
            "为保证稳定版本不放大更新，fisher_diag.server_lr "
            f"必须位于 (0, 1]，当前值：{server_lr}。"
        )

    if (
        not math.isfinite(shrinkage_lambda)
        or shrinkage_lambda < 0.0
    ):
        raise ValueError(
            "fisher_diag.shrinkage_lambda 必须是非负有限数，"
            f"当前值：{shrinkage_lambda}。"
        )

    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            "fisher_diag.eps 必须是正有限数，"
            f"当前值：{eps}。"
        )

    if not math.isfinite(min_usage) or min_usage < 0.0:
        raise ValueError(
            "fisher_diag.min_usage 必须是非负有限数，"
            f"当前值：{min_usage}。"
        )

    if (
        not math.isfinite(usage_tolerance)
        or usage_tolerance < 0.0
    ):
        raise ValueError(
            "fisher_diag.usage_tolerance 必须是非负有限数，"
            f"当前值：{usage_tolerance}。"
        )

    if fallback != "keep_global":
        raise ValueError(
            "当前只支持 fisher_diag.fallback=keep_global，"
            f"当前值：{fallback}。"
        )


def _mapping_get_by_int_or_str_key(
    mapping: Mapping[Any, Any],
    key: int,
    default: Any,
) -> Any:
    """兼容 JSON 序列化后 expert id 从 int 变成 str 的情况。"""

    if key in mapping:
        return mapping[key]

    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]

    return default


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容项目 ConfigNode、普通 dict 和普通对象的配置读取。

    当前项目的 cfg.get() 支持诸如 ``fisher_diag.beta`` 的点号路径。
    这里额外保留嵌套 dict 回退，便于单元测试时直接传普通字典。
    """

    sentinel = object()

    if hasattr(cfg, "get"):
        value = cfg.get(key, sentinel)
        if value is not sentinel:
            return value

    if isinstance(cfg, Mapping):
        current: Any = cfg
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    return getattr(cfg, key, default)
