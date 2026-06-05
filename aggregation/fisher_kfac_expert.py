from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aggregation.base import Aggregator, build_sample_weights
from fl.types import AggregationResult, ClientUpdate
from utils.state_dict_ops import (
    check_finite_state_dict,
    clone_state_dict,
    normalize_weights,
)


class FisherKFACExpertAggregator(Aggregator):
    """
    基于 K-FAC Fisher 的专家参数聚合器。

    这个聚合器只用于 expert 参数聚合，不用于 non_expert 参数。

    目标函数：
        min_W sum_i p_i / 2 * <W - W_i, F_i(W - W_i)>

    其中：
        F_i ≈ A_i ⊗ B_i

    对 Linear 层，K-FAC matvec 为：
        F_i vec(DeltaW) ≈ vec(B_i @ DeltaW @ A_i)

    服务端不再额外引入更新步长，而是直接求解 K-FAC/Fisher
    加权聚合方程：
        sum_i p_i * (B_i @ W @ A_i + damping * W)
        =
        sum_i p_i * (B_i @ W_i @ A_i + damping * W_i)

    说明：
        1. 客户端上传的是 A_mean、B_mean、count。
        2. 服务端用 count 作为证据强度，内部归一化成 p_i。
        3. 没有 K-FAC 信息的参数 fallback 到默认保持上一轮参数。
        4. 这里不实现 WoLF / history filter / reliability。
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "fisher_kfac_expert"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        为了满足 Aggregator 接口，返回样本数权重。

        注意：
            fisher_kfac_expert 的主聚合逻辑不走普通加权 delta。
            这里的权重只在 fallback=sample_weighted 时使用。
            AggregationResult.weights 会在 aggregate() 里改成 K-FAC count 汇总权重。
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
        执行 K-FAC expert 聚合。

        参数：
            global_state:
                本轮聚合前的全局参数。

            client_updates:
                本轮客户端更新。

            param_names:
                expert 参数名列表。

            base_state:
                上一步 non_expert 聚合后的 state_dict。
                expert 聚合结果会写到这个基础 state_dict 上。

            strict:
                是否严格检查缺失字段。
        """
        self._validate_client_updates(client_updates)

        if self.param_group_name != "expert":
            raise ValueError("fisher_kfac_expert 只能用于 expert 参数聚合。")

        target_param_names = _resolve_param_names(
            global_state=global_state,
            param_names=param_names,
        )
        target_param_set = set(target_param_names)

        raw_weights = self.compute_weights(client_updates)
        sample_weights = normalize_weights(raw_weights)

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        min_count = int(_cfg_get(self.cfg, "kfac.min_count", 1))
        solver_steps = int(_cfg_get(self.cfg, "kfac.server_steps", 5))
        cg_tol = float(_cfg_get(self.cfg, "kfac.cg_tol", 1.0e-8))
        damping = float(_cfg_get(self.cfg, "kfac.damping", 1.0e-4))
        fallback = str(_cfg_get(self.cfg, "kfac.fallback", "none")).lower().strip()

        if min_count <= 0:
            min_count = 1

        if solver_steps < 0:
            raise ValueError(f"kfac.server_steps 不能小于 0，当前值：{solver_steps}")

        if cg_tol < 0:
            raise ValueError(f"kfac.cg_tol 不能小于 0，当前值：{cg_tol}")

        if damping < 0:
            raise ValueError(f"kfac.damping 不能小于 0，当前值：{damping}")

        layer_names = _collect_kfac_layer_names(client_updates)

        solved_params = set()
        fallback_params = set()
        skipped_layers: List[str] = []
        valid_client_ids = set()
        kfac_client_counts: Dict[int, int] = {}
        kfac_layer_weights: Dict[str, Dict[int, float]] = {}

        valid_layers = 0
        valid_client_layers = 0
        total_count = 0

        trace_A_values: List[float] = []
        trace_B_values: List[float] = []
        residual_norm_values: List[float] = []
        delta_norm_values: List[float] = []
        solver_delta_norm_values: List[float] = []

        for layer_name in layer_names:
            entries = _collect_valid_layer_entries(
                layer_name=layer_name,
                client_updates=client_updates,
                global_state=global_state,
                target_param_set=target_param_set,
                min_count=min_count,
                strict=False,
            )

            if len(entries) == 0:
                skipped_layers.append(layer_name)
                continue

            reference = entries[0]
            weight_name = reference["weight_name"]
            bias_name = reference["bias_name"]
            include_bias = bool(reference["include_bias"])

            if weight_name not in target_param_set:
                skipped_layers.append(layer_name)
                continue

            if bias_name is not None and bias_name not in target_param_set:
                include_bias = False
                bias_name = None

            try:
                solved_weight, solved_bias, layer_diag = _solve_kfac_linear_layer(
                    global_state=global_state,
                    entries=entries,
                    weight_name=weight_name,
                    bias_name=bias_name,
                    include_bias=include_bias,
                    solver_steps=solver_steps,
                    cg_tol=cg_tol,
                    damping=damping,
                )
            except Exception:
                if strict:
                    raise

                skipped_layers.append(layer_name)
                continue

            new_state_dict[weight_name] = solved_weight.detach().cpu()
            solved_params.add(weight_name)

            if bias_name is not None and solved_bias is not None:
                new_state_dict[bias_name] = solved_bias.detach().cpu()
                solved_params.add(bias_name)

            valid_layers += 1
            valid_client_layers += int(layer_diag["valid_clients"])
            total_count += int(layer_diag["total_count"])
            valid_client_ids.update(int(entry["client_id"]) for entry in entries)

            trace_A_values.extend(layer_diag["trace_A_values"])
            trace_B_values.extend(layer_diag["trace_B_values"])
            residual_norm_values.extend(layer_diag["residual_norm_values"])
            delta_norm_values.append(float(layer_diag["delta_norm"]))
            solver_delta_norm_values.append(float(layer_diag["solver_delta_norm"]))

            layer_client_counts = {
                int(entry["client_id"]): int(entry["count"])
                for entry in entries
            }
            layer_total_count = int(sum(layer_client_counts.values()))

            if layer_total_count > 0:
                kfac_layer_weights[layer_name] = {
                    int(client_id): float(count) / float(layer_total_count)
                    for client_id, count in layer_client_counts.items()
                }

            for client_id, count in layer_client_counts.items():
                client_id = int(client_id)
                kfac_client_counts[client_id] = (
                    int(kfac_client_counts.get(client_id, 0)) + int(count)
                )

        for name in target_param_names:
            if name in solved_params:
                continue

            if fallback == "none":
                continue

            if fallback != "sample_weighted":
                raise ValueError(
                    f"不支持的 kfac.fallback：{fallback}。"
                    "当前支持：sample_weighted, none"
                )

            if not torch.is_tensor(global_state[name]):
                continue

            if not torch.is_floating_point(global_state[name]):
                continue

            new_state_dict[name] = _sample_weighted_param(
                name=name,
                global_state=global_state,
                client_updates=client_updates,
                weights=sample_weights,
                strict=strict,
            ).detach().cpu()
            fallback_params.add(name)

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=target_param_names,
        )

        mean_count = float(total_count / max(valid_client_layers, 1))
        kfac_weights = _normalize_kfac_client_counts(
            client_counts=kfac_client_counts,
            client_updates=client_updates,
        )
        cos_kfac_uniform = _cos_kfac_uniform(
            global_state=global_state,
            new_state_dict=new_state_dict,
            client_updates=client_updates,
            param_names=sorted(solved_params),
        )

        diagnostics = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": len(target_param_names),
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in kfac_weights.items()
            },
            "kfac_client_counts": {
                int(client_id): int(count)
                for client_id, count in kfac_client_counts.items()
            },
            "kfac_layer_weights": kfac_layer_weights,
            "valid_layers": int(valid_layers),
            "valid_clients": int(len(valid_client_ids)),
            "skipped_layers": int(len(skipped_layers)),
            "skipped_layer_names": list(skipped_layers[:20]),
            "valid_client_layers": int(valid_client_layers),
            "total_count": int(total_count),
            "mean_count": float(mean_count),
            "mean_trace_A": _safe_mean(trace_A_values),
            "mean_trace_B": _safe_mean(trace_B_values),
            "max_trace_A": _safe_max(trace_A_values),
            "max_trace_B": _safe_max(trace_B_values),
            "mean_residual_norm": _safe_mean(residual_norm_values),
            "max_residual_norm": _safe_max(residual_norm_values),
            # 兼容旧字段名：这里的 grad_norm 实际表示 CG 残差范数。
            "mean_grad_norm": _safe_mean(residual_norm_values),
            "max_grad_norm": _safe_max(residual_norm_values),
            # mean_delta_norm 表示最终 K-FAC 参数相对上一轮 global 参数的真实更新幅度。
            "mean_delta_norm": _safe_mean(delta_norm_values),
            "mean_global_delta_norm": _safe_mean(delta_norm_values),
            # mean_solver_delta_norm 表示 K-FAC 解相对 FedAvg 初始点的修正幅度。
            "mean_solver_delta_norm": _safe_mean(solver_delta_norm_values),
            "cos_kfac_uniform": float(cos_kfac_uniform),
            "solver_steps": int(solver_steps),
            "server_steps": int(solver_steps),
            "cg_tol": float(cg_tol),
            "damping": float(damping),
            "min_count": int(min_count),
            "fallback": fallback,
            "solved_params": int(len(solved_params)),
            "fallback_params": int(len(fallback_params)),
        }

        if bool(_cfg_get(self.cfg, "kfac.log_detail", True)):
            print(
                "[ExpertKFAC] "
                f"valid_layers={diagnostics['valid_layers']} "
                f"valid_clients={diagnostics['valid_clients']} "
                f"skipped_layers={diagnostics['skipped_layers']} "
                f"total_count={diagnostics['total_count']} "
                f"mean_count={diagnostics['mean_count']:.2f} "
                f"mean_trace_A={diagnostics['mean_trace_A']:.6e} "
                f"mean_trace_B={diagnostics['mean_trace_B']:.6e} "
                f"solver_steps={diagnostics['solver_steps']} "
                f"mean_residual_norm={diagnostics['mean_residual_norm']:.6e} "
                f"mean_delta_norm={diagnostics['mean_delta_norm']:.6e} "
                f"mean_solver_delta_norm={diagnostics['mean_solver_delta_norm']:.6e} "
                f"fallback_params={diagnostics['fallback_params']} "
                f"cos_kfac_uniform={diagnostics['cos_kfac_uniform']:.6f}",
                flush=True,
            )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=kfac_weights,
            diagnostics=diagnostics,
        )


def _solve_kfac_linear_layer(
    global_state: Mapping[str, torch.Tensor],
    entries: Sequence[Dict[str, Any]],
    weight_name: str,
    bias_name: Optional[str],
    include_bias: bool,
    solver_steps: int,
    cg_tol: float,
    damping: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    对一个 Linear block 直接求解 K-FAC/Fisher 加权聚合结果。

    entries 中每个元素对应一个 client-layer：
        {
            "client_id": int,
            "count": int,
            "A": Tensor,
            "B": Tensor,
            "local_weight": Tensor,
            "local_bias": Tensor or None,
            ...
        }

    求解方程：
        sum_i p_i * (B_i @ W @ A_i + damping * W)
        =
        sum_i p_i * (B_i @ W_i @ A_i + damping * W_i)

    这里使用 CG，只需要 K-FAC matvec，不构造完整 Kronecker 矩阵。
    """
    if len(entries) == 0:
        raise ValueError(f"{weight_name} 没有有效 K-FAC entries。")

    device = global_state[weight_name].device
    dtype = global_state[weight_name].dtype

    processed_entries = []
    total_count = 0

    for entry in entries:
        count = int(entry["count"])
        if count <= 0:
            continue

        A = entry["A"].to(device=device, dtype=dtype)
        B = entry["B"].to(device=device, dtype=dtype)

        A = _symmetrize_square(A)
        B = _symmetrize_square(B)

        local_weight = entry["local_weight"].to(device=device, dtype=dtype)

        local_bias = None
        if include_bias and bias_name is not None and entry.get("local_bias") is not None:
            local_bias = entry["local_bias"].to(device=device, dtype=dtype)

        local_aug = _make_augmented_weight(
            weight=local_weight,
            bias=local_bias,
            include_bias=include_bias,
        )

        _validate_kfac_shapes(
            A=A,
            B=B,
            W_aug=local_aug,
            layer_name=str(entry.get("layer_name", weight_name)),
        )

        processed_entries.append(
            {
                "client_id": int(entry["client_id"]),
                "count": count,
                "A": A,
                "B": B,
                "local_aug": local_aug,
                "trace_A": float(torch.trace(A.detach().float()).item()),
                "trace_B": float(torch.trace(B.detach().float()).item()),
            }
        )
        total_count += count

    if len(processed_entries) == 0 or total_count <= 0:
        raise ValueError(f"{weight_name} 没有 count > 0 的有效 K-FAC entries。")

    weights = [
        float(entry["count"]) / float(total_count)
        for entry in processed_entries
    ]

    # count-weighted FedAvg 作为 CG 初始点，同时也是 damping 右端项里的 W_avg。
    W_avg = torch.zeros_like(processed_entries[0]["local_aug"])
    for weight, entry in zip(weights, processed_entries):
        W_avg = W_avg + float(weight) * entry["local_aug"]

    global_weight = global_state[weight_name].to(device=device, dtype=dtype)
    global_bias = None
    if include_bias and bias_name is not None and bias_name in global_state:
        global_bias = global_state[bias_name].to(device=device, dtype=dtype)

    W_global_aug = _make_augmented_weight(
        weight=global_weight,
        bias=global_bias,
        include_bias=include_bias,
    )

    rhs = torch.zeros_like(W_avg)
    for weight, entry in zip(weights, processed_entries):
        rhs = rhs + float(weight) * _kfac_matvec(
            delta=entry["local_aug"],
            A=entry["A"],
            B=entry["B"],
            damping=0.0,
        )

    if damping > 0:
        rhs = rhs + float(damping) * W_avg

    def matvec(x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)

        for weight, entry in zip(weights, processed_entries):
            result = result + float(weight) * _kfac_matvec(
                delta=x,
                A=entry["A"],
                B=entry["B"],
                damping=0.0,
            )

        if damping > 0:
            result = result + float(damping) * x

        return result

    if solver_steps == 0:
        W_aug = W_avg.detach().clone()
        residual = rhs - matvec(W_aug)
        residual_norm_values = [
            float(residual.detach().float().norm().item())
        ]
    else:
        W_aug, residual_norm_values = _conjugate_gradient_matrix(
            matvec=matvec,
            rhs=rhs,
            initial=W_avg,
            max_steps=solver_steps,
            tol=cg_tol,
            layer_name=weight_name,
        )

    if not torch.isfinite(W_aug).all():
        raise ValueError(f"{weight_name} 的 K-FAC CG 解出现 NaN 或 Inf。")

    solved_weight, solved_bias = _split_augmented_weight(
        W_aug=W_aug,
        include_bias=include_bias,
    )

    global_delta_norm = float(
        (W_aug.detach().float() - W_global_aug.detach().float()).norm().item()
    )
    solver_delta_norm = float(
        (W_aug.detach().float() - W_avg.detach().float()).norm().item()
    )

    diagnostics = {
        "valid_clients": int(len(processed_entries)),
        "total_count": int(total_count),
        "trace_A_values": [
            float(entry["trace_A"])
            for entry in processed_entries
        ],
        "trace_B_values": [
            float(entry["trace_B"])
            for entry in processed_entries
        ],
        "residual_norm_values": residual_norm_values,
        "delta_norm": float(global_delta_norm),
        "global_delta_norm": float(global_delta_norm),
        "solver_delta_norm": float(solver_delta_norm),
    }

    return solved_weight, solved_bias, diagnostics


def _conjugate_gradient_matrix(
    matvec: Any,
    rhs: torch.Tensor,
    initial: torch.Tensor,
    max_steps: int,
    tol: float,
    layer_name: str,
) -> Tuple[torch.Tensor, List[float]]:
    """
    用 Conjugate Gradient 解矩阵形状的线性系统。

    matvec 接收和 rhs 同形状的 Tensor，返回同形状 Tensor。
    为了避免构造完整 Kronecker 矩阵，这里直接在矩阵空间做点积和更新。
    """
    x = initial.detach().clone()

    if not torch.isfinite(rhs).all():
        raise ValueError(f"{layer_name} 的 K-FAC rhs 出现 NaN 或 Inf。")

    r = rhs - matvec(x)

    if not torch.isfinite(r).all():
        raise ValueError(f"{layer_name} 的 K-FAC 初始残差出现 NaN 或 Inf。")

    p = r.detach().clone()
    rs_old = torch.sum(r * r)

    residual_norm_values = [
        float(torch.sqrt(torch.clamp(rs_old.detach().float(), min=0.0)).item())
    ]

    if residual_norm_values[-1] <= float(tol):
        return x, residual_norm_values

    eps = torch.tensor(
        1.0e-30,
        device=rhs.device,
        dtype=rhs.dtype,
    )

    for _ in range(int(max_steps)):
        Ap = matvec(p)

        if not torch.isfinite(Ap).all():
            raise ValueError(f"{layer_name} 的 K-FAC matvec 出现 NaN 或 Inf。")

        denom = torch.sum(p * Ap)

        if not torch.isfinite(denom):
            raise ValueError(f"{layer_name} 的 K-FAC CG denom 出现 NaN 或 Inf。")

        if torch.abs(denom).detach().float().item() <= 1.0e-30:
            break

        alpha = rs_old / (denom + eps)

        x = x + alpha * p
        r = r - alpha * Ap

        if not torch.isfinite(x).all():
            raise ValueError(f"{layer_name} 的 K-FAC CG 解出现 NaN 或 Inf。")

        if not torch.isfinite(r).all():
            raise ValueError(f"{layer_name} 的 K-FAC CG 残差出现 NaN 或 Inf。")

        rs_new = torch.sum(r * r)
        residual_norm = float(
            torch.sqrt(torch.clamp(rs_new.detach().float(), min=0.0)).item()
        )
        residual_norm_values.append(residual_norm)

        if residual_norm <= float(tol):
            break

        beta = rs_new / (rs_old + eps)
        p = r + beta * p
        rs_old = rs_new

    return x, residual_norm_values


def _kfac_matvec(
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """
    K-FAC 矩阵向量乘法。

    对 Linear 层：
        F vec(delta) ≈ vec(B @ delta @ A)

    damping 使用简单的各向同性阻尼：
        matvec = B @ delta @ A + damping * delta
    """
    result = B.matmul(delta).matmul(A)

    if damping > 0:
        result = result + float(damping) * delta

    return result


def _symmetrize_square(matrix: torch.Tensor) -> torch.Tensor:
    """对方阵做对称化，减少 K-FAC 统计里的数值非对称误差。"""
    if matrix.dim() == 2 and matrix.size(0) == matrix.size(1):
        return 0.5 * (matrix + matrix.transpose(0, 1))

    return matrix


def _collect_kfac_layer_names(
    client_updates: Sequence[ClientUpdate],
) -> List[str]:
    """收集本轮所有客户端上传过的 K-FAC layer_name。"""
    layer_names = set()

    for update in client_updates:
        payload = update.extra.get("expert_kfac", None)

        if not isinstance(payload, Mapping):
            continue

        for layer_name in payload.keys():
            layer_names.add(str(layer_name))

    return sorted(layer_names)


def _collect_valid_layer_entries(
    layer_name: str,
    client_updates: Sequence[ClientUpdate],
    global_state: Mapping[str, torch.Tensor],
    target_param_set: set[str],
    min_count: int,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """收集某个 K-FAC layer 在所有客户端上的有效条目。"""
    entries: List[Dict[str, Any]] = []

    for update in client_updates:
        payload = update.extra.get("expert_kfac", None)

        if not isinstance(payload, Mapping):
            continue

        if layer_name not in payload:
            continue

        item = payload[layer_name]

        if not isinstance(item, Mapping):
            continue

        try:
            entry = _build_layer_entry(
                layer_name=layer_name,
                item=item,
                update=update,
                global_state=global_state,
                target_param_set=target_param_set,
                min_count=min_count,
            )
        except Exception:
            if strict:
                raise

            continue

        if entry is not None:
            entries.append(entry)

    return entries


def _build_layer_entry(
    layer_name: str,
    item: Mapping[str, Any],
    update: ClientUpdate,
    global_state: Mapping[str, torch.Tensor],
    target_param_set: set[str],
    min_count: int,
) -> Optional[Dict[str, Any]]:
    """把客户端上传的单层 K-FAC payload 转成服务端可用 entry。"""
    count = int(item.get("count", 0))

    if count < int(min_count):
        return None

    weight_name = str(item.get("weight_name", ""))
    bias_name_raw = item.get("bias_name", None)
    bias_name = None if bias_name_raw is None else str(bias_name_raw)

    if weight_name == "":
        return None

    if weight_name not in target_param_set:
        return None

    if weight_name not in global_state:
        return None

    if weight_name not in update.model_delta:
        return None

    A = item.get("A", None)
    B = item.get("B", None)

    if not torch.is_tensor(A) or not torch.is_tensor(B):
        return None

    if not torch.isfinite(A).all():
        return None

    if not torch.isfinite(B).all():
        return None

    global_weight = global_state[weight_name]
    local_weight = global_weight.detach().cpu() + update.model_delta[
        weight_name
    ].detach().cpu()

    local_bias = None
    include_bias = bool(item.get("include_bias", False))

    if include_bias and bias_name is not None:
        if (
            bias_name in target_param_set
            and bias_name in global_state
            and bias_name in update.model_delta
        ):
            global_bias = global_state[bias_name]
            local_bias = global_bias.detach().cpu() + update.model_delta[
                bias_name
            ].detach().cpu()
        else:
            include_bias = False
            bias_name = None

    return {
        "client_id": int(update.client_id),
        "layer_name": str(layer_name),
        "weight_name": weight_name,
        "bias_name": bias_name,
        "include_bias": include_bias,
        "count": int(count),
        "A": A.detach().cpu(),
        "B": B.detach().cpu(),
        "local_weight": local_weight,
        "local_bias": local_bias,
    }


def _make_augmented_weight(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    include_bias: bool,
) -> torch.Tensor:
    """
    把 Linear 的 weight 和 bias 合成 W_aug。

    weight: [out_features, in_features]
    bias: [out_features]

    include_bias=True 时：
        W_aug = [W, b]
        shape: [out_features, in_features + 1]
    """
    if include_bias and bias is not None:
        return torch.cat(
            [
                weight,
                bias.reshape(-1, 1),
            ],
            dim=1,
        )

    return weight


def _split_augmented_weight(
    W_aug: torch.Tensor,
    include_bias: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """把 W_aug 拆回 weight 和 bias。"""
    if include_bias:
        weight = W_aug[:, :-1]
        bias = W_aug[:, -1]
        return weight, bias

    return W_aug, None


def _validate_kfac_shapes(
    A: torch.Tensor,
    B: torch.Tensor,
    W_aug: torch.Tensor,
    layer_name: str,
) -> None:
    """检查 A、B、W_aug 的形状是否匹配。"""
    if A.dim() != 2 or A.size(0) != A.size(1):
        raise ValueError(
            f"{layer_name} 的 A 不是方阵，shape={tuple(A.shape)}"
        )

    if B.dim() != 2 or B.size(0) != B.size(1):
        raise ValueError(
            f"{layer_name} 的 B 不是方阵，shape={tuple(B.shape)}"
        )

    if W_aug.dim() != 2:
        raise ValueError(
            f"{layer_name} 的 W_aug 不是二维矩阵，shape={tuple(W_aug.shape)}"
        )

    if B.size(0) != W_aug.size(0):
        raise ValueError(
            f"{layer_name} 的 B 和 W_aug 输出维度不匹配："
            f"B={tuple(B.shape)}, W_aug={tuple(W_aug.shape)}"
        )

    if A.size(0) != W_aug.size(1):
        raise ValueError(
            f"{layer_name} 的 A 和 W_aug 输入维度不匹配："
            f"A={tuple(A.shape)}, W_aug={tuple(W_aug.shape)}"
        )


def _sample_weighted_param(
    name: str,
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    weights: Mapping[int, float],
    strict: bool,
) -> torch.Tensor:
    """对单个参数执行 sample_weighted fallback。"""
    global_tensor = global_state[name]
    total_delta = torch.zeros_like(global_tensor)

    for update in client_updates:
        client_id = int(update.client_id)

        if client_id not in weights:
            if strict:
                raise KeyError(f"weights 缺少客户端 {client_id} 的权重。")

            continue

        if name not in update.model_delta:
            if strict:
                raise KeyError(
                    f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                )

            continue

        delta = update.model_delta[name].to(global_tensor.device)
        total_delta = total_delta + float(weights[client_id]) * delta

    return global_tensor + total_delta


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """解析 expert 参数名列表。"""
    if param_names is None:
        return list(global_state.keys())

    names = list(param_names)

    for name in names:
        if name not in global_state:
            raise KeyError(f"global_state 中不存在参数：{name}")

    return names


def _normalize_kfac_client_counts(
    client_counts: Mapping[int, int],
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    把所有 solved K-FAC layer 的 routed count 汇总成 client 级别权重。

    注意：
        这个是 K-FAC evidence 的汇总权重，不是每一层真实使用的唯一权重。
        每一层真实权重在 diagnostics["kfac_layer_weights"] 里。
    """
    result = {
        int(update.client_id): 0.0
        for update in client_updates
    }

    total_count = int(sum(int(count) for count in client_counts.values()))

    if total_count <= 0:
        if len(client_updates) == 0:
            return result

        uniform_weight = 1.0 / float(len(client_updates))
        return {
            int(update.client_id): float(uniform_weight)
            for update in client_updates
        }

    for client_id, count in client_counts.items():
        result[int(client_id)] = float(count) / float(total_count)

    return result


def _cos_kfac_uniform(
    global_state: Mapping[str, torch.Tensor],
    new_state_dict: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    param_names: Sequence[str],
) -> float:
    """
    计算 K-FAC 聚合方向和 uniform 直接平均方向的余弦相似度。

    cos 接近 1：
        K-FAC 基本退化成 uniform 直接平均。

    cos 明显小于 1：
        K-FAC 改变了专家聚合方向。
    """
    if len(param_names) == 0:
        return 0.0

    if len(client_updates) == 0:
        return 0.0

    uniform_weight = 1.0 / float(len(client_updates))

    dot = 0.0
    norm_kfac = 0.0
    norm_uniform = 0.0

    for name in param_names:
        if name not in global_state or name not in new_state_dict:
            continue

        if not torch.is_tensor(global_state[name]):
            continue

        if not torch.is_floating_point(global_state[name]):
            continue

        kfac_delta = (
            new_state_dict[name].detach().cpu().float()
            - global_state[name].detach().cpu().float()
        )

        uniform_delta = torch.zeros_like(kfac_delta)

        for update in client_updates:
            if name not in update.model_delta:
                continue

            uniform_delta = uniform_delta + uniform_weight * update.model_delta[
                name
            ].detach().cpu().float()

        kfac_flat = kfac_delta.reshape(-1)
        uniform_flat = uniform_delta.reshape(-1)

        dot += float(torch.dot(kfac_flat, uniform_flat).item())
        norm_kfac += float(torch.dot(kfac_flat, kfac_flat).item())
        norm_uniform += float(torch.dot(uniform_flat, uniform_flat).item())

    if norm_kfac <= 0 or norm_uniform <= 0:
        return 0.0

    return float(dot / ((norm_kfac ** 0.5) * (norm_uniform ** 0.5) + 1.0e-12))


def _safe_mean(values: Sequence[float]) -> float:
    """安全计算均值。"""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if len(finite_values) == 0:
        return 0.0

    return float(sum(finite_values) / len(finite_values))


def _safe_max(values: Sequence[float]) -> float:
    """安全计算最大值。"""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if len(finite_values) == 0:
        return 0.0

    return float(max(finite_values))


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的读取。

    dict 或 ConfigNode:
        cfg.get(key, default)

    普通对象:
        getattr(cfg, key, default)
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)