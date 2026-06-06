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

    paper-like FedFisher 目标：
        min_W sum_i p_i / 2 * <W - W_i, F_i(W - W_i)>

    其中：
        F_i ≈ A_i ⊗ B_i

    对 Linear 层，K-FAC matvec 为：
        F_i vec(W) ≈ vec(B_i @ W @ A_i)

    默认 paper-like 模式不再把 routed count 当聚合权重，也不再默认加入
    damping 软正则。routed count 只用于判断该 expert layer 的 K-FAC 是否有效。

    支持两种求解范围：
        1. per_layer：逐个 expert Linear layer 求解，兼容旧实现。
        2. global_expert：把所有 expert layer 放进同一个服务端优化过程，
           等价于在 expert 参数空间上做一个 block-diagonal K-FAC FedFisher 求解。

    支持三种求解方式：
        1. cg：Conjugate Gradient 求解线性系统。
        2. gd：FedFisher Algorithm 1 风格固定步数梯度下降。
        3. adam：作者实践中使用的 Adam-like 服务端优化，但这里不使用
           server validation 选 best，固定返回最后一步。
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
            这里的权重主要用于 fallback=sample_weighted，以及
            kfac.weight_mode=sample_weighted 时的客户端级权重。
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
        server_lr = float(_cfg_get(self.cfg, "kfac.server_lr", 0.01))
        adam_beta1 = float(_cfg_get(self.cfg, "kfac.adam_beta1", 0.9))
        adam_beta2 = float(_cfg_get(self.cfg, "kfac.adam_beta2", 0.99))
        adam_eps = float(_cfg_get(self.cfg, "kfac.adam_eps", 0.01))
        damping = float(_cfg_get(self.cfg, "kfac.damping", 0.0))
        use_damping = bool(_cfg_get(self.cfg, "kfac.use_damping", False))
        fallback = str(_cfg_get(self.cfg, "kfac.fallback", "none")).lower().strip()
        weight_mode = str(_cfg_get(self.cfg, "kfac.weight_mode", "sample_weighted")).lower().strip()
        solve_scope = str(_cfg_get(self.cfg, "kfac.solve_scope", "per_layer")).lower().strip()
        solve_mode = str(_cfg_get(self.cfg, "kfac.solve_mode", "cg")).lower().strip()
        fisher_timing = str(_cfg_get(self.cfg, "kfac.fisher_timing", "after_train")).lower().strip()

        if min_count <= 0:
            min_count = 1

        if solver_steps < 0:
            raise ValueError(f"kfac.server_steps 不能小于 0，当前值：{solver_steps}")

        if cg_tol < 0:
            raise ValueError(f"kfac.cg_tol 不能小于 0，当前值：{cg_tol}")

        if server_lr < 0:
            raise ValueError(f"kfac.server_lr 不能小于 0，当前值：{server_lr}")

        if damping < 0:
            raise ValueError(f"kfac.damping 不能小于 0，当前值：{damping}")

        if not use_damping:
            damping = 0.0

        _validate_choice(
            name="kfac.weight_mode",
            value=weight_mode,
            choices=("routed_count", "sample_weighted", "uniform"),
        )
        _validate_choice(
            name="kfac.solve_scope",
            value=solve_scope,
            choices=("per_layer", "global_expert"),
        )
        _validate_choice(
            name="kfac.solve_mode",
            value=solve_mode,
            choices=("cg", "gd", "adam"),
        )

        layer_names = _collect_kfac_layer_names(client_updates)
        layer_groups: List[Dict[str, Any]] = []
        skipped_layers: List[str] = []

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

            layer_groups.append(
                {
                    "layer_name": layer_name,
                    "entries": entries,
                    "weight_name": weight_name,
                    "bias_name": bias_name,
                    "include_bias": include_bias,
                }
            )

        solved_params = set()
        fallback_params = set()
        valid_client_ids = set()
        kfac_client_counts: Dict[int, int] = {}
        kfac_layer_weights: Dict[str, Dict[int, float]] = {}

        valid_layers = 0
        valid_client_layers = 0
        total_count = 0
        global_expert_param_count = 0

        trace_A_values: List[float] = []
        trace_B_values: List[float] = []
        residual_norm_values: List[float] = []
        delta_norm_values: List[float] = []
        solver_delta_norm_values: List[float] = []
        solver_grad_norm_values: List[float] = []
        solver_update_norm_values: List[float] = []

        if len(layer_groups) > 0:
            if solve_scope == "global_expert":
                try:
                    global_result = _solve_global_expert_layers(
                        global_state=global_state,
                        client_updates=client_updates,
                        sample_weights=sample_weights,
                        layer_groups=layer_groups,
                        weight_mode=weight_mode,
                        solve_mode=solve_mode,
                        solver_steps=solver_steps,
                        cg_tol=cg_tol,
                        server_lr=server_lr,
                        adam_beta1=adam_beta1,
                        adam_beta2=adam_beta2,
                        adam_eps=adam_eps,
                        damping=damping,
                        use_damping=use_damping,
                        strict=strict,
                    )
                except Exception:
                    if strict:
                        raise

                    global_result = {
                        "solutions": {},
                        "diagnostics": [],
                        "skipped_layers": [
                            str(group["layer_name"])
                            for group in layer_groups
                        ],
                        "solver_grad_norm_values": [],
                        "solver_update_norm_values": [],
                    }

                skipped_layers.extend(global_result.get("skipped_layers", []))
                solver_grad_norm_values.extend(
                    global_result.get("solver_grad_norm_values", [])
                )
                solver_update_norm_values.extend(
                    global_result.get("solver_update_norm_values", [])
                )

                for layer_result in global_result.get("diagnostics", []):
                    weight_name = str(layer_result["weight_name"])
                    bias_name = layer_result.get("bias_name", None)
                    solved_weight = layer_result["solved_weight"]
                    solved_bias = layer_result.get("solved_bias", None)

                    new_state_dict[weight_name] = solved_weight.detach().cpu()
                    solved_params.add(weight_name)

                    if bias_name is not None and solved_bias is not None:
                        new_state_dict[str(bias_name)] = solved_bias.detach().cpu()
                        solved_params.add(str(bias_name))

                    _accumulate_layer_diagnostics(
                        layer_diag=layer_result,
                        valid_client_ids=valid_client_ids,
                        kfac_client_counts=kfac_client_counts,
                        kfac_layer_weights=kfac_layer_weights,
                        trace_A_values=trace_A_values,
                        trace_B_values=trace_B_values,
                        residual_norm_values=residual_norm_values,
                        delta_norm_values=delta_norm_values,
                        solver_delta_norm_values=solver_delta_norm_values,
                    )

                    valid_layers += 1
                    valid_client_layers += int(layer_result["valid_clients"])
                    total_count += int(layer_result["total_count"])
                    global_expert_param_count += int(layer_result["param_count"])
            else:
                for group in layer_groups:
                    layer_name = str(group["layer_name"])
                    weight_name = str(group["weight_name"])
                    bias_name = group.get("bias_name", None)
                    include_bias = bool(group["include_bias"])

                    try:
                        solved_weight, solved_bias, layer_diag = _solve_kfac_linear_layer(
                            global_state=global_state,
                            client_updates=client_updates,
                            sample_weights=sample_weights,
                            entries=group["entries"],
                            weight_name=weight_name,
                            bias_name=bias_name,
                            include_bias=include_bias,
                            weight_mode=weight_mode,
                            solve_mode=solve_mode,
                            solver_steps=solver_steps,
                            cg_tol=cg_tol,
                            server_lr=server_lr,
                            adam_beta1=adam_beta1,
                            adam_beta2=adam_beta2,
                            adam_eps=adam_eps,
                            damping=damping,
                            use_damping=use_damping,
                        )
                    except Exception:
                        if strict:
                            raise

                        skipped_layers.append(layer_name)
                        continue

                    new_state_dict[weight_name] = solved_weight.detach().cpu()
                    solved_params.add(weight_name)

                    if bias_name is not None and solved_bias is not None:
                        new_state_dict[str(bias_name)] = solved_bias.detach().cpu()
                        solved_params.add(str(bias_name))

                    _accumulate_layer_diagnostics(
                        layer_diag=layer_diag,
                        valid_client_ids=valid_client_ids,
                        kfac_client_counts=kfac_client_counts,
                        kfac_layer_weights=kfac_layer_weights,
                        trace_A_values=trace_A_values,
                        trace_B_values=trace_B_values,
                        residual_norm_values=residual_norm_values,
                        delta_norm_values=delta_norm_values,
                        solver_delta_norm_values=solver_delta_norm_values,
                    )

                    solver_grad_norm_values.extend(layer_diag.get("solver_grad_norm_values", []))
                    solver_update_norm_values.extend(layer_diag.get("solver_update_norm_values", []))
                    valid_layers += 1
                    valid_client_layers += int(layer_diag["valid_clients"])
                    total_count += int(layer_diag["total_count"])
                    global_expert_param_count += int(layer_diag["param_count"])

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
        result_weights = _build_result_client_weights(
            weight_mode=weight_mode,
            client_counts=kfac_client_counts,
            client_updates=client_updates,
            sample_weights=sample_weights,
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
                for client_id, weight in result_weights.items()
            },
            "kfac_weight_mode": weight_mode,
            "weight_mode": weight_mode,
            "solve_scope": solve_scope,
            "solve_mode": solve_mode,
            "kfac_client_sample_weights": {
                int(client_id): float(weight)
                for client_id, weight in sample_weights.items()
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
            # 兼容旧字段名：cg 时表示残差范数，gd/adam 时表示最终 FedFisher 梯度范数。
            "mean_grad_norm": _safe_mean(residual_norm_values),
            "max_grad_norm": _safe_max(residual_norm_values),
            "mean_solver_grad_norm": _safe_mean(solver_grad_norm_values),
            "max_solver_grad_norm": _safe_max(solver_grad_norm_values),
            "mean_solver_update_norm": _safe_mean(solver_update_norm_values),
            "max_solver_update_norm": _safe_max(solver_update_norm_values),
            # mean_delta_norm 表示最终 K-FAC 参数相对上一轮 global 参数的真实更新幅度。
            "mean_delta_norm": _safe_mean(delta_norm_values),
            "mean_global_delta_norm": _safe_mean(delta_norm_values),
            # mean_solver_delta_norm 表示 K-FAC 解相对 FedAvg 初始化点的修正幅度。
            "mean_solver_delta_norm": _safe_mean(solver_delta_norm_values),
            "cos_kfac_uniform": float(cos_kfac_uniform),
            "solver_steps": int(solver_steps),
            "server_steps": int(solver_steps),
            "server_lr": float(server_lr),
            "adam_beta1": float(adam_beta1),
            "adam_beta2": float(adam_beta2),
            "adam_eps": float(adam_eps),
            "cg_tol": float(cg_tol),
            "damping": float(damping),
            "use_damping": bool(use_damping),
            "min_count": int(min_count),
            "fallback": fallback,
            "fisher_timing": fisher_timing,
            "model_selection": "final_step",
            "use_server_validation": False,
            "global_expert_param_count": int(global_expert_param_count),
            "solved_params": int(len(solved_params)),
            "fallback_params": int(len(fallback_params)),
        }

        if bool(_cfg_get(self.cfg, "kfac.log_detail", True)):
            print(
                "[ExpertKFAC] "
                f"weight_mode={diagnostics['weight_mode']} "
                f"solve_scope={diagnostics['solve_scope']} "
                f"solve_mode={diagnostics['solve_mode']} "
                f"valid_layers={diagnostics['valid_layers']} "
                f"valid_clients={diagnostics['valid_clients']} "
                f"skipped_layers={diagnostics['skipped_layers']} "
                f"total_count={diagnostics['total_count']} "
                f"mean_count={diagnostics['mean_count']:.2f} "
                f"mean_trace_A={diagnostics['mean_trace_A']:.6e} "
                f"mean_trace_B={diagnostics['mean_trace_B']:.6e} "
                f"server_steps={diagnostics['server_steps']} "
                f"server_lr={diagnostics['server_lr']:.6e} "
                f"damping={diagnostics['damping']:.6e} "
                f"use_damping={diagnostics['use_damping']} "
                f"mean_residual_norm={diagnostics['mean_residual_norm']:.6e} "
                f"mean_solver_grad_norm={diagnostics['mean_solver_grad_norm']:.6e} "
                f"mean_solver_update_norm={diagnostics['mean_solver_update_norm']:.6e} "
                f"mean_delta_norm={diagnostics['mean_delta_norm']:.6e} "
                f"mean_solver_delta_norm={diagnostics['mean_solver_delta_norm']:.6e} "
                f"global_expert_param_count={diagnostics['global_expert_param_count']} "
                f"fallback_params={diagnostics['fallback_params']} "
                f"cos_kfac_uniform={diagnostics['cos_kfac_uniform']:.6f}",
                flush=True,
            )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=result_weights,
            diagnostics=diagnostics,
        )


def _solve_kfac_linear_layer(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    entries: Sequence[Dict[str, Any]],
    weight_name: str,
    bias_name: Optional[str],
    include_bias: bool,
    weight_mode: str,
    solve_mode: str,
    solver_steps: int,
    cg_tol: float,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    damping: float,
    use_damping: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    对一个 Linear block 求解 K-FAC/Fisher 加权聚合结果。

    paper-like 方程：
        sum_i p_i * B_i @ W @ A_i = sum_i p_i * B_i @ W_i @ A_i

    只有 use_damping=True 且 damping>0 时才会额外加入：
        + damping * W = + damping * W_avg
    """
    system = _prepare_layer_system(
        global_state=global_state,
        client_updates=client_updates,
        sample_weights=sample_weights,
        entries=entries,
        weight_name=weight_name,
        bias_name=bias_name,
        include_bias=include_bias,
        weight_mode=weight_mode,
        damping=damping,
        use_damping=use_damping,
    )

    if solve_mode == "cg":
        solutions, residual_norm_values = _run_cg_on_systems(
            systems=[system],
            max_steps=solver_steps,
            tol=cg_tol,
        )
        W_aug = solutions[system["layer_name"]]
        solver_grad_norm_values = list(residual_norm_values)
        solver_update_norm_values: List[float] = []
    else:
        solutions, solver_grad_norm_values, solver_update_norm_values = _run_optimizer_on_systems(
            systems=[system],
            solve_mode=solve_mode,
            server_steps=solver_steps,
            server_lr=server_lr,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
        )
        W_aug = solutions[system["layer_name"]]
        residual_norm_values = _compute_system_residual_norms(
            systems=[system],
            solutions=solutions,
        )

    layer_diag = _build_solution_diagnostics(
        system=system,
        W_aug=W_aug,
        residual_norm_values=residual_norm_values,
    )
    layer_diag["solver_grad_norm_values"] = list(solver_grad_norm_values)
    layer_diag["solver_update_norm_values"] = list(solver_update_norm_values)

    solved_weight = layer_diag.pop("solved_weight")
    solved_bias = layer_diag.pop("solved_bias")

    return solved_weight, solved_bias, layer_diag


def _solve_global_expert_layers(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    layer_groups: Sequence[Dict[str, Any]],
    weight_mode: str,
    solve_mode: str,
    solver_steps: int,
    cg_tol: float,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    damping: float,
    use_damping: bool,
    strict: bool,
) -> Dict[str, Any]:
    """
    在所有 expert layer 上执行一个统一的 FedFisher K-FAC 服务端求解。

    实现上仍然按 layer 做 K-FAC matvec，但 CG/GD/Adam 的梯度范数、
    更新范数和迭代过程是在所有 expert layer 的联合参数空间上完成的。
    """
    systems: List[Dict[str, Any]] = []
    skipped_layers: List[str] = []

    for group in layer_groups:
        try:
            system = _prepare_layer_system(
                global_state=global_state,
                client_updates=client_updates,
                sample_weights=sample_weights,
                entries=group["entries"],
                weight_name=str(group["weight_name"]),
                bias_name=group.get("bias_name", None),
                include_bias=bool(group["include_bias"]),
                weight_mode=weight_mode,
                damping=damping,
                use_damping=use_damping,
            )
        except Exception:
            if strict:
                raise

            skipped_layers.append(str(group["layer_name"]))
            continue

        systems.append(system)

    if len(systems) == 0:
        return {
            "solutions": {},
            "diagnostics": [],
            "skipped_layers": skipped_layers,
            "solver_grad_norm_values": [],
            "solver_update_norm_values": [],
        }

    if solve_mode == "cg":
        solutions, solver_grad_norm_values = _run_cg_on_systems(
            systems=systems,
            max_steps=solver_steps,
            tol=cg_tol,
        )
        solver_update_norm_values: List[float] = []
    else:
        solutions, solver_grad_norm_values, solver_update_norm_values = _run_optimizer_on_systems(
            systems=systems,
            solve_mode=solve_mode,
            server_steps=solver_steps,
            server_lr=server_lr,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
        )

    diagnostics = []
    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, solutions[layer_name])
        layer_residual_norm_values = [
            float(residual.detach().float().norm().item())
        ]
        layer_diag = _build_solution_diagnostics(
            system=system,
            W_aug=solutions[layer_name],
            residual_norm_values=layer_residual_norm_values,
        )
        diagnostics.append(layer_diag)

    return {
        "solutions": solutions,
        "diagnostics": diagnostics,
        "skipped_layers": skipped_layers,
        "solver_grad_norm_values": solver_grad_norm_values,
        "solver_update_norm_values": solver_update_norm_values,
    }


def _prepare_layer_system(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    entries: Sequence[Dict[str, Any]],
    weight_name: str,
    bias_name: Optional[str],
    include_bias: bool,
    weight_mode: str,
    damping: float,
    use_damping: bool,
) -> Dict[str, Any]:
    """把某个 expert Linear layer 的 entries 转成可求解的 K-FAC 系统。"""
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
                "layer_name": str(entry.get("layer_name", weight_name)),
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

    weights = _compute_entry_weights(
        processed_entries=processed_entries,
        client_updates=client_updates,
        sample_weights=sample_weights,
        weight_mode=weight_mode,
    )

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

    if use_damping and damping > 0:
        rhs = rhs + float(damping) * W_avg

    layer_weights = {
        int(entry["client_id"]): float(weight)
        for entry, weight in zip(processed_entries, weights)
    }

    return {
        "layer_name": str(processed_entries[0].get("layer_name", weight_name)),
        "weight_name": str(weight_name),
        "bias_name": bias_name,
        "include_bias": bool(include_bias),
        "processed_entries": processed_entries,
        "weights": weights,
        "layer_weights": layer_weights,
        "total_count": int(total_count),
        "W_avg": W_avg,
        "W_global_aug": W_global_aug,
        "rhs": rhs,
        "damping": float(damping),
        "use_damping": bool(use_damping),
        "param_count": int(W_avg.numel()),
    }


def _compute_entry_weights(
    processed_entries: Sequence[Dict[str, Any]],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    weight_mode: str,
) -> List[float]:
    """根据 weight_mode 计算当前 layer 的客户端权重。"""
    if len(processed_entries) == 0:
        return []

    if weight_mode == "routed_count":
        counts = [float(max(int(entry["count"]), 0)) for entry in processed_entries]
        total = float(sum(counts))
        if total <= 0:
            return [1.0 / float(len(processed_entries)) for _ in processed_entries]
        return [float(count) / total for count in counts]

    if weight_mode == "sample_weighted":
        raw = [
            float(sample_weights.get(int(entry["client_id"]), 0.0))
            for entry in processed_entries
        ]
        total = float(sum(raw))
        if total <= 0:
            return [1.0 / float(len(processed_entries)) for _ in processed_entries]
        return [float(value) / total for value in raw]

    if weight_mode == "uniform":
        return [1.0 / float(len(processed_entries)) for _ in processed_entries]

    raise ValueError(f"不支持的 kfac.weight_mode：{weight_mode}")


def _layer_matvec(system: Mapping[str, Any], x: torch.Tensor) -> torch.Tensor:
    """计算当前 layer 系统的 sum_i p_i F_i x。"""
    result = torch.zeros_like(x)

    for weight, entry in zip(system["weights"], system["processed_entries"]):
        result = result + float(weight) * _kfac_matvec(
            delta=x,
            A=entry["A"],
            B=entry["B"],
            damping=0.0,
        )

    if bool(system.get("use_damping", False)) and float(system.get("damping", 0.0)) > 0:
        result = result + float(system["damping"]) * x

    return result


def _run_optimizer_on_systems(
    systems: Sequence[Dict[str, Any]],
    solve_mode: str,
    server_steps: int,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
) -> Tuple[Dict[str, torch.Tensor], List[float], List[float]]:
    """在多个 expert layer 系统上执行统一的 GD/Adam-like 服务端优化。"""
    if solve_mode not in {"gd", "adam"}:
        raise ValueError(f"_run_optimizer_on_systems 不支持 solve_mode={solve_mode}")

    current = {
        str(system["layer_name"]): system["W_avg"].detach().clone()
        for system in systems
    }
    first_moment = {
        str(system["layer_name"]): torch.zeros_like(system["W_avg"])
        for system in systems
    }
    second_moment = {
        str(system["layer_name"]): torch.zeros_like(system["W_avg"])
        for system in systems
    }

    grad_norm_values: List[float] = []
    update_norm_values: List[float] = []

    if server_steps == 0:
        grad_norm_values.append(
            _compute_global_grad_norm(
                systems=systems,
                current=current,
            )
        )
        return current, grad_norm_values, update_norm_values

    for _ in range(int(server_steps)):
        grads: Dict[str, torch.Tensor] = {}
        grad_sq_sum = 0.0

        for system in systems:
            layer_name = str(system["layer_name"])
            grad = _layer_matvec(system, current[layer_name]) - system["rhs"]

            if not torch.isfinite(grad).all():
                raise ValueError(f"{layer_name} 的 FedFisher 梯度出现 NaN 或 Inf。")

            grads[layer_name] = grad
            grad_sq_sum += float(torch.sum(grad.detach().float() * grad.detach().float()).item())

        grad_norm_values.append(float(grad_sq_sum ** 0.5))

        update_sq_sum = 0.0
        for system in systems:
            layer_name = str(system["layer_name"])
            grad = grads[layer_name]

            if solve_mode == "gd":
                update = float(server_lr) * grad
            else:
                # 对齐 FedFisher 作者实践中的 Adam-like 写法：不做 bias correction，
                # 且一阶/二阶动量不乘 (1-beta)。
                first_moment[layer_name] = (
                    float(adam_beta1) * first_moment[layer_name] + grad
                )
                second_moment[layer_name] = (
                    float(adam_beta2) * second_moment[layer_name] + grad * grad
                )
                update = float(server_lr) * first_moment[layer_name] / (
                    torch.sqrt(second_moment[layer_name]) + float(adam_eps)
                )

            if not torch.isfinite(update).all():
                raise ValueError(f"{layer_name} 的 FedFisher 更新出现 NaN 或 Inf。")

            current[layer_name] = current[layer_name] - update
            update_sq_sum += float(torch.sum(update.detach().float() * update.detach().float()).item())

            if not torch.isfinite(current[layer_name]).all():
                raise ValueError(f"{layer_name} 的 FedFisher 解出现 NaN 或 Inf。")

        update_norm_values.append(float(update_sq_sum ** 0.5))

    return current, grad_norm_values, update_norm_values


def _run_cg_on_systems(
    systems: Sequence[Dict[str, Any]],
    max_steps: int,
    tol: float,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """在多个 expert layer 系统上执行一个联合 CG 求解。"""
    x = {
        str(system["layer_name"]): system["W_avg"].detach().clone()
        for system in systems
    }
    r = {}
    p = {}

    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, x[layer_name])

        if not torch.isfinite(residual).all():
            raise ValueError(f"{layer_name} 的 K-FAC 初始残差出现 NaN 或 Inf。")

        r[layer_name] = residual
        p[layer_name] = residual.detach().clone()

    rs_old = _dict_dot(r, r)
    residual_norm_values = [float(max(rs_old, 0.0) ** 0.5)]

    if residual_norm_values[-1] <= float(tol):
        return x, residual_norm_values

    if max_steps == 0:
        return x, residual_norm_values

    for _ in range(int(max_steps)):
        Ap = {}
        for system in systems:
            layer_name = str(system["layer_name"])
            value = _layer_matvec(system, p[layer_name])

            if not torch.isfinite(value).all():
                raise ValueError(f"{layer_name} 的 K-FAC matvec 出现 NaN 或 Inf。")

            Ap[layer_name] = value

        denom = _dict_dot(p, Ap)

        if not math.isfinite(float(denom)):
            raise ValueError("K-FAC CG denom 出现 NaN 或 Inf。")

        if abs(float(denom)) <= 1.0e-30:
            break

        alpha = float(rs_old) / (float(denom) + 1.0e-30)

        for system in systems:
            layer_name = str(system["layer_name"])
            x[layer_name] = x[layer_name] + alpha * p[layer_name]
            r[layer_name] = r[layer_name] - alpha * Ap[layer_name]

            if not torch.isfinite(x[layer_name]).all():
                raise ValueError(f"{layer_name} 的 K-FAC CG 解出现 NaN 或 Inf。")

            if not torch.isfinite(r[layer_name]).all():
                raise ValueError(f"{layer_name} 的 K-FAC CG 残差出现 NaN 或 Inf。")

        rs_new = _dict_dot(r, r)
        residual_norm = float(max(rs_new, 0.0) ** 0.5)
        residual_norm_values.append(residual_norm)

        if residual_norm <= float(tol):
            break

        beta = float(rs_new) / (float(rs_old) + 1.0e-30)
        for system in systems:
            layer_name = str(system["layer_name"])
            p[layer_name] = r[layer_name] + beta * p[layer_name]

        rs_old = rs_new

    return x, residual_norm_values


def _compute_global_grad_norm(
    systems: Sequence[Dict[str, Any]],
    current: Mapping[str, torch.Tensor],
) -> float:
    """计算所有 expert layer 上的 FedFisher 梯度范数。"""
    grad_sq_sum = 0.0
    for system in systems:
        layer_name = str(system["layer_name"])
        grad = _layer_matvec(system, current[layer_name]) - system["rhs"]
        grad_sq_sum += float(torch.sum(grad.detach().float() * grad.detach().float()).item())

    return float(grad_sq_sum ** 0.5)


def _compute_system_residual_norms(
    systems: Sequence[Dict[str, Any]],
    solutions: Mapping[str, torch.Tensor],
) -> List[float]:
    """计算每个 layer 最终方程残差范数。"""
    residuals = []
    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, solutions[layer_name])
        residuals.append(float(residual.detach().float().norm().item()))

    return residuals


def _build_solution_diagnostics(
    system: Mapping[str, Any],
    W_aug: torch.Tensor,
    residual_norm_values: Sequence[float],
) -> Dict[str, Any]:
    """把某个 layer 的最终解和诊断信息打包。"""
    if not torch.isfinite(W_aug).all():
        raise ValueError(f"{system['weight_name']} 的 K-FAC 解出现 NaN 或 Inf。")

    solved_weight, solved_bias = _split_augmented_weight(
        W_aug=W_aug,
        include_bias=bool(system["include_bias"]),
    )

    global_delta_norm = float(
        (W_aug.detach().float() - system["W_global_aug"].detach().float()).norm().item()
    )
    solver_delta_norm = float(
        (W_aug.detach().float() - system["W_avg"].detach().float()).norm().item()
    )

    processed_entries = system["processed_entries"]

    return {
        "layer_name": str(system["layer_name"]),
        "weight_name": str(system["weight_name"]),
        "bias_name": system.get("bias_name", None),
        "include_bias": bool(system["include_bias"]),
        "solved_weight": solved_weight,
        "solved_bias": solved_bias,
        "valid_clients": int(len(processed_entries)),
        "client_ids": [int(entry["client_id"]) for entry in processed_entries],
        "client_counts": {
            int(entry["client_id"]): int(entry["count"])
            for entry in processed_entries
        },
        "layer_weights": {
            int(client_id): float(weight)
            for client_id, weight in system["layer_weights"].items()
        },
        "total_count": int(system["total_count"]),
        "trace_A_values": [
            float(entry["trace_A"])
            for entry in processed_entries
        ],
        "trace_B_values": [
            float(entry["trace_B"])
            for entry in processed_entries
        ],
        "residual_norm_values": list(float(value) for value in residual_norm_values),
        "delta_norm": float(global_delta_norm),
        "global_delta_norm": float(global_delta_norm),
        "solver_delta_norm": float(solver_delta_norm),
        "param_count": int(system["param_count"]),
    }


def _accumulate_layer_diagnostics(
    layer_diag: Mapping[str, Any],
    valid_client_ids: set[int],
    kfac_client_counts: Dict[int, int],
    kfac_layer_weights: Dict[str, Dict[int, float]],
    trace_A_values: List[float],
    trace_B_values: List[float],
    residual_norm_values: List[float],
    delta_norm_values: List[float],
    solver_delta_norm_values: List[float],
) -> None:
    """汇总单个 layer 的诊断信息。"""
    layer_name = str(layer_diag["layer_name"])

    for client_id in layer_diag.get("client_ids", []):
        valid_client_ids.add(int(client_id))

    for client_id, count in layer_diag.get("client_counts", {}).items():
        client_id = int(client_id)
        kfac_client_counts[client_id] = int(kfac_client_counts.get(client_id, 0)) + int(count)

    kfac_layer_weights[layer_name] = {
        int(client_id): float(weight)
        for client_id, weight in layer_diag.get("layer_weights", {}).items()
    }

    trace_A_values.extend(layer_diag.get("trace_A_values", []))
    trace_B_values.extend(layer_diag.get("trace_B_values", []))
    residual_norm_values.extend(layer_diag.get("residual_norm_values", []))
    delta_norm_values.append(float(layer_diag.get("delta_norm", 0.0)))
    solver_delta_norm_values.append(float(layer_diag.get("solver_delta_norm", 0.0)))


def _dict_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    """计算多个 tensor 组成的向量点积。"""
    value = 0.0
    for key in left.keys():
        value += float(torch.sum(left[key].detach().float() * right[key].detach().float()).item())
    return float(value)


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


def _build_result_client_weights(
    weight_mode: str,
    client_counts: Mapping[int, int],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
) -> Dict[int, float]:
    """构造 AggregationResult.weights 中展示的客户端权重。"""
    if weight_mode == "sample_weighted":
        return {
            int(update.client_id): float(sample_weights.get(int(update.client_id), 0.0))
            for update in client_updates
        }

    if weight_mode == "uniform":
        if len(client_updates) == 0:
            return {}
        weight = 1.0 / float(len(client_updates))
        return {
            int(update.client_id): float(weight)
            for update in client_updates
        }

    return _normalize_kfac_client_counts(
        client_counts=client_counts,
        client_updates=client_updates,
    )


def _normalize_kfac_client_counts(
    client_counts: Mapping[int, int],
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    把所有 solved K-FAC layer 的 routed count 汇总成 client 级别权重。

    注意：
        这个是 routed_count 模式下的 K-FAC evidence 汇总权重，
        不是 sample_weighted / uniform 模式下的真实权重。
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


def _validate_choice(
    name: str,
    value: str,
    choices: Sequence[str],
) -> None:
    """检查配置枚举值。"""
    if value not in set(choices):
        raise ValueError(
            f"{name} 必须是 {sorted(choices)} 之一，当前值：{value}"
        )


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
