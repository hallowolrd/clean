from __future__ import annotations

"""
严格逐样本的 expert diagonal empirical Fisher 采集器。

适配当前项目的稀疏 MoE：每个 expert 由 nn.Linear 层组成，参数名包含
``experts.<expert_id>``。本文件只统计 expert 参数，不统计 backbone、router
等 non-expert 参数。

对客户端 i 的 expert e：

    f_{i,e}
      = 1 / c_{i,e}
        * sum_{s: e in TopK(x_s)}
          (grad_{theta_{i,e}} ell_s) ** 2

其中 c_{i,e} 是 evidence 数据中路由到 expert e 的样本数。

对于 Linear 层 z_s = W a_s + b：

    grad_W ell_s = delta_s a_s^T

因此可以用 Linear hook 严格累计每个样本的梯度平方：

    sum_s (grad_W ell_s) ** 2
      = einsum("so,si->oi", delta**2, activation**2)

采集损失必须使用 reduction="sum"。如果使用 mean，grad_output 会被 batch
大小缩放，得到的 Fisher 尺度将不正确。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.param_groups import get_expert_id_from_name
from utils.eval import extract_logits, unpack_batch


ExpertDiagonalFisherPayload = Dict[str, Any]


@dataclass
class _LinearFisherBuffer:
    """单个 expert Linear 层的 diagonal Fisher 累计缓存。"""

    module_name: str
    module: nn.Linear
    expert_id: int

    weight_sum: torch.Tensor = field(init=False)
    bias_sum: Optional[torch.Tensor] = field(init=False)
    sample_count: int = 0

    def __post_init__(self) -> None:
        # 在模型所在设备累计，结束后再转到 CPU，避免每批频繁拷贝。
        self.weight_sum = torch.zeros_like(
            self.module.weight,
            dtype=torch.float32,
            device=self.module.weight.device,
        )
        self.bias_sum = (
            None
            if self.module.bias is None
            else torch.zeros_like(
                self.module.bias,
                dtype=torch.float32,
                device=self.module.bias.device,
            )
        )

    @property
    def weight_name(self) -> str:
        return f"{self.module_name}.weight"

    @property
    def bias_name(self) -> Optional[str]:
        if self.module.bias is None:
            return None
        return f"{self.module_name}.bias"

    def add(
        self,
        activation: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> None:
        """
        累计一个 batch 中路由到该 expert 的逐样本梯度平方。

        当前模型中 expert 输入为 [R, D]，R 是本批路由到该 expert 的
        样本数。这里严格要求二维，避免把额外 token/空间维度误当样本。
        """
        if activation.dim() != 2:
            raise ValueError(
                f"{self.module_name} 输入应为 [R, in_features]，"
                f"实际为 {tuple(activation.shape)}。"
            )
        if grad_output.dim() != 2:
            raise ValueError(
                f"{self.module_name} grad_output 应为 [R, out_features]，"
                f"实际为 {tuple(grad_output.shape)}。"
            )
        if activation.size(0) != grad_output.size(0):
            raise ValueError(
                f"{self.module_name} 的输入和输出梯度样本数不一致："
                f"{tuple(activation.shape)} vs {tuple(grad_output.shape)}。"
            )
        if activation.size(1) != self.module.in_features:
            raise ValueError(
                f"{self.module_name} 输入维度错误："
                f"{activation.size(1)} != {self.module.in_features}。"
            )
        if grad_output.size(1) != self.module.out_features:
            raise ValueError(
                f"{self.module_name} 输出梯度维度错误："
                f"{grad_output.size(1)} != {self.module.out_features}。"
            )

        count = int(activation.size(0))
        if count == 0:
            return

        # 每行对应一个路由样本。先逐样本平方，再沿样本维求和。
        a2 = activation.detach().float().square()
        d2 = grad_output.detach().float().square()

        self.weight_sum.add_(torch.einsum("ro,ri->oi", d2, a2))
        if self.bias_sum is not None:
            self.bias_sum.add_(d2.sum(dim=0))

        self.sample_count += count

    def export(self, routed_count: int) -> Dict[str, torch.Tensor]:
        """除以该 expert 的路由样本数，得到 conditional mean Fisher。"""
        routed_count = int(routed_count)
        if routed_count <= 0:
            return {}

        # 当前 ExpertFFN 的每个 Linear 都处理同一批路由样本，因此应相等。
        if self.sample_count != routed_count:
            raise RuntimeError(
                f"{self.module_name} 的 hook 计数与 router 计数不一致："
                f"hook={self.sample_count}, router={routed_count}。"
            )

        weight_fisher = self.weight_sum / float(routed_count)
        _check_fisher_tensor(self.weight_name, weight_fisher)

        result = {self.weight_name: weight_fisher.detach().cpu()}

        if self.bias_sum is not None:
            bias_fisher = self.bias_sum / float(routed_count)
            bias_name = str(self.bias_name)
            _check_fisher_tensor(bias_name, bias_fisher)
            result[bias_name] = bias_fisher.detach().cpu()

        return result


def collect_expert_diag_fisher(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device | str | None = None,
    cfg: Any = None,
) -> ExpertDiagonalFisherPayload:
    """
    在本地训练完成后的 local_model 上采集严格逐样本 expert Fisher。

    与当前 ``collect_expert_kfac`` 的调用方式保持一致。客户端应传入：

        collect_expert_diag_fisher(
            model=local_model,
            train_loader=self.evidence_loader,
            criterion=criterion,
            device=self.device,
            cfg=self.cfg,
        )

    使用的配置项：

        fisher_diag.fisher_timing: after_train
        fisher_diag.model_mode: eval
        fisher_diag.max_batches: 0       # 0 表示完整 evidence_loader
        fisher_diag.min_count: 1
        fisher_diag.expert_name_pattern: "experts."

    返回的 ``expert_usage`` 定义为：

        usage_{i,e} = routed_samples_{i,e} / num_evidence_samples_i

    因而 top-k 路由下，所有 expert usage 的和等于 topk，而不是 1。
    """
    if device is None:
        device = _infer_model_device(model)
    device = torch.device(device)

    fisher_timing = str(
        _cfg_get(
            cfg,
            "fisher_diag.fisher_timing",
            _cfg_get(cfg, "fisher_diag.collect_timing", "after_train"),
        )
    ).lower().strip()
    if fisher_timing != "after_train":
        raise ValueError(
            "当前只支持 fisher_diag.fisher_timing=after_train，"
            f"实际为 {fisher_timing}。"
        )

    model_mode = str(
        _cfg_get(cfg, "fisher_diag.model_mode", "eval")
    ).lower().strip()
    if model_mode not in {"eval", "train"}:
        raise ValueError(
            "fisher_diag.model_mode 只支持 eval 或 train，"
            f"实际为 {model_mode}。"
        )

    max_batches = int(_cfg_get(cfg, "fisher_diag.max_batches", 0))
    min_count = int(_cfg_get(cfg, "fisher_diag.min_count", 1))
    expert_name_pattern = str(
        _cfg_get(cfg, "fisher_diag.expert_name_pattern", "experts.")
    )

    if max_batches < 0:
        raise ValueError("fisher_diag.max_batches 不能小于 0。")
    if min_count <= 0:
        raise ValueError("fisher_diag.min_count 必须大于 0。")

    model.to(device)
    num_experts = _infer_num_experts(model, cfg)
    expected_topk = _infer_topk(model, cfg)

    # 1. 找出所有 expert Linear，并建立缓存。
    buffers: Dict[str, _LinearFisherBuffer] = {}
    covered_param_names: set[str] = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if expert_name_pattern not in module_name:
            continue

        expert_id = get_expert_id_from_name(module_name)
        if expert_id is None:
            raise ValueError(
                f"模块 {module_name} 命中了 expert pattern，"
                "但无法解析 experts.<id>。"
            )
        if expert_id < 0 or expert_id >= num_experts:
            raise ValueError(
                f"模块 {module_name} 的 expert_id={expert_id} 越界，"
                f"num_experts={num_experts}。"
            )

        buffer = _LinearFisherBuffer(
            module_name=module_name,
            module=module,
            expert_id=expert_id,
        )
        buffers[module_name] = buffer
        covered_param_names.add(buffer.weight_name)
        if buffer.bias_name is not None:
            covered_param_names.add(buffer.bias_name)

    if not buffers:
        raise RuntimeError(
            "没有找到 expert Linear，请检查参数命名是否包含 experts.<id>。"
        )

    # 当前实现只覆盖 Linear expert。未来加入 Conv/LayerNorm 时应单独实现。
    all_expert_params = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and get_expert_id_from_name(name) is not None
    }
    uncovered = sorted(all_expert_params - covered_param_names)
    if uncovered:
        raise NotImplementedError(
            "当前 fisher_diag.py 只支持 expert nn.Linear 参数。"
            f"未覆盖参数：{uncovered[:20]}"
        )

    # 2. 注册 Linear forward hook。
    # forward hook 保存该次调用的输入，并在 Linear 输出 Tensor 上注册梯度 hook。
    # 这样 activation 与对应 grad_output 一一绑定，不需要维护全局 stack。
    handles = []

    def make_forward_hook(buffer: _LinearFisherBuffer):
        def forward_hook(
            layer: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{buffer.module_name} 没有有效 Tensor 输入。")
            if not torch.is_tensor(output):
                raise RuntimeError(f"{buffer.module_name} 输出不是 Tensor。")
            if not output.requires_grad:
                raise RuntimeError(
                    f"{buffer.module_name} 输出不需要梯度，无法计算 Fisher。"
                )

            activation = inputs[0].detach()

            def grad_hook(grad_output: torch.Tensor) -> torch.Tensor:
                buffer.add(activation, grad_output)
                return grad_output

            output.register_hook(grad_hook)

        return forward_hook

    for buffer in buffers.values():
        handles.append(
            buffer.module.register_forward_hook(make_forward_hook(buffer))
        )

    # 3. 构建 sum reduction 的 CrossEntropyLoss。
    sum_criterion = _build_sum_cross_entropy(cfg, criterion).to(device)
    ignore_index = int(sum_criterion.ignore_index)

    was_training = bool(model.training)
    model.train(model_mode == "train")

    routed_counts = torch.zeros(num_experts, dtype=torch.long)
    total_samples = 0
    total_batches = 0
    actual_topk: Optional[int] = None

    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            for batch_idx, batch in enumerate(train_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                images, targets = unpack_batch(batch)
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if targets.dim() != 1:
                    raise ValueError(
                        f"targets 应为 [B]，实际为 {tuple(targets.shape)}。"
                    )
                if torch.any(targets == ignore_index):
                    raise ValueError(
                        "evidence 数据中出现 ignore_index。为保证路由计数与"
                        "有效梯度样本数严格一致，当前实现暂不支持忽略标签。"
                    )

                batch_size = int(targets.size(0))
                if batch_size == 0:
                    continue

                model.zero_grad(set_to_none=True)

                try:
                    outputs = model(images, return_router_info=True)
                except TypeError as exc:
                    raise TypeError(
                        "模型必须支持 model(images, return_router_info=True)。"
                    ) from exc

                logits = extract_logits(outputs)
                router_info = _extract_router_info(outputs)
                selected_experts = _validate_selected_experts(
                    router_info=router_info,
                    batch_size=batch_size,
                    num_experts=num_experts,
                )

                batch_topk = int(selected_experts.size(1))
                if actual_topk is None:
                    actual_topk = batch_topk
                elif actual_topk != batch_topk:
                    raise RuntimeError("不同 batch 的 topk 不一致。")

                if expected_topk is not None and batch_topk != expected_topk:
                    raise RuntimeError(
                        f"router 实际 topk={batch_topk}，"
                        f"但配置/模型属性为 {expected_topk}。"
                    )

                batch_counts = torch.bincount(
                    selected_experts.reshape(-1).detach().cpu(),
                    minlength=num_experts,
                )
                routed_counts.add_(batch_counts)

                loss = sum_criterion(logits, targets)
                if loss.dim() != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"第 {batch_idx} 个 evidence batch 的 loss 非法。"
                    )

                # 只做 backward 采集 Fisher，不执行 optimizer.step()。
                loss.backward()

                total_samples += batch_size
                total_batches += 1
                model.zero_grad(set_to_none=True)

    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    if total_samples <= 0:
        raise RuntimeError("没有 evidence 样本，无法计算 Fisher。")
    if actual_topk is None:
        raise RuntimeError("没有获得 router topk 信息。")

    # top-k 下每个样本应贡献 topk 次 expert 激活。
    total_activations = int(routed_counts.sum().item())
    expected_activations = int(total_samples * actual_topk)
    if total_activations != expected_activations:
        raise RuntimeError(
            f"router 计数错误：actual={total_activations}, "
            f"expected={expected_activations}。"
        )

    # 4. 每个 expert 的 Fisher 除以自己的路由样本数。
    fisher_diag: Dict[str, torch.Tensor] = {}
    valid_fisher_experts = []

    for expert_id in range(num_experts):
        routed_count = int(routed_counts[expert_id].item())
        if routed_count < min_count:
            continue

        expert_buffers = [
            buffer
            for buffer in buffers.values()
            if buffer.expert_id == expert_id
        ]
        if not expert_buffers:
            raise RuntimeError(f"expert {expert_id} 没有 Linear buffer。")

        for buffer in expert_buffers:
            fisher_diag.update(buffer.export(routed_count))

        valid_fisher_experts.append(expert_id)

    routed_count_dict = {
        expert_id: int(routed_counts[expert_id].item())
        for expert_id in range(num_experts)
    }
    usage_dict = {
        expert_id: routed_count_dict[expert_id] / float(total_samples)
        for expert_id in range(num_experts)
    }

    return {
        "estimator": "exact_per_sample_empirical_fisher_linear_hook",
        "fisher_timing": fisher_timing,
        "model_mode": model_mode,
        "max_batches": max_batches,
        "min_count": min_count,
        "num_evidence_samples": total_samples,
        "num_batches": total_batches,
        "num_experts": num_experts,
        "topk": actual_topk,
        "total_activations": total_activations,
        "expert_usage": usage_dict,
        "expert_routed_samples": routed_count_dict,
        "valid_fisher_experts": valid_fisher_experts,
        "dead_experts": [
            expert_id
            for expert_id, count in routed_count_dict.items()
            if count == 0
        ],
        "diag": fisher_diag,
    }


def summarize_expert_diag_fisher(
    payload: Optional[ExpertDiagonalFisherPayload],
) -> Dict[str, Any]:
    """生成不包含 Fisher Tensor 本体的轻量日志摘要。"""
    if not payload:
        return {
            "supported": False,
            "num_fisher_tensors": 0,
            "total_fisher_numel": 0,
        }

    diag = payload.get("diag", {})
    if not isinstance(diag, Mapping):
        raise TypeError('payload["diag"] 必须是 Mapping。')

    total_numel = 0
    total_sum = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    expert_trace: Dict[int, float] = {}

    for name, fisher in diag.items():
        if not torch.is_tensor(fisher):
            raise TypeError(f"Fisher {name} 不是 Tensor。")
        _check_fisher_tensor(str(name), fisher)

        fisher64 = fisher.detach().double()
        total_numel += int(fisher64.numel())
        value_sum = float(fisher64.sum().item())
        total_sum += value_sum
        min_value = min(min_value, float(fisher64.min().item()))
        max_value = max(max_value, float(fisher64.max().item()))

        expert_id = get_expert_id_from_name(str(name))
        if expert_id is not None:
            expert_trace[expert_id] = expert_trace.get(expert_id, 0.0) + value_sum

    if total_numel == 0:
        mean_value = min_value = max_value = 0.0
    else:
        mean_value = total_sum / float(total_numel)

    return {
        "supported": True,
        "estimator": str(payload.get("estimator", "")),
        "num_fisher_tensors": len(diag),
        "total_fisher_numel": total_numel,
        "num_evidence_samples": int(payload.get("num_evidence_samples", 0)),
        "num_batches": int(payload.get("num_batches", 0)),
        "num_experts": int(payload.get("num_experts", 0)),
        "topk": int(payload.get("topk", 0)),
        "mean_fisher": mean_value,
        "min_fisher": min_value,
        "max_fisher": max_value,
        "expert_trace": {
            int(k): float(v) for k, v in sorted(expert_trace.items())
        },
        "valid_fisher_experts": list(
            payload.get("valid_fisher_experts", [])
        ),
        "dead_experts": list(payload.get("dead_experts", [])),
    }


def _extract_router_info(outputs: Any) -> Mapping[str, Any]:
    """兼容 dataclass、dict 和 tuple/list 三种模型输出。"""
    if hasattr(outputs, "router_info") and isinstance(
        outputs.router_info, Mapping
    ):
        return outputs.router_info

    if isinstance(outputs, Mapping):
        router_info = outputs.get("router_info")
        if isinstance(router_info, Mapping):
            return router_info

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        if isinstance(outputs[1], Mapping):
            return outputs[1]

    raise RuntimeError("模型没有返回有效 router_info。")


def _validate_selected_experts(
    router_info: Mapping[str, Any],
    batch_size: int,
    num_experts: int,
) -> torch.Tensor:
    """读取并检查 router_info["selected_experts"]。"""
    selected = router_info.get("selected_experts")
    if not torch.is_tensor(selected):
        raise RuntimeError('router_info["selected_experts"] 不是 Tensor。')
    if selected.dim() != 2 or selected.size(0) != batch_size:
        raise ValueError(
            "selected_experts 应为 [B, topk]，"
            f"实际为 {tuple(selected.shape)}。"
        )

    selected = selected.detach().long()
    if selected.size(1) <= 0:
        raise ValueError("topk 必须大于 0。")
    if int(selected.min().item()) < 0 or int(selected.max().item()) >= num_experts:
        raise ValueError("selected_experts 中存在越界 expert id。")

    # torch.topk 正常不会重复；显式检查防止 usage 重复计数。
    if selected.size(1) > 1:
        sorted_ids = torch.sort(selected, dim=1).values
        if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise RuntimeError("同一样本的 top-k expert 中出现重复 id。")

    return selected


def _build_sum_cross_entropy(
    cfg: Any,
    criterion: Optional[nn.Module],
) -> nn.CrossEntropyLoss:
    """复制训练 CE 配置，但强制 reduction="sum"。"""
    if criterion is not None and not isinstance(criterion, nn.CrossEntropyLoss):
        raise TypeError(
            "严格逐样本 Fisher 当前只支持 nn.CrossEntropyLoss，"
            f"实际为 {type(criterion)}。"
        )

    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))
    ignore_index = -100
    weight = None

    if isinstance(criterion, nn.CrossEntropyLoss):
        label_smoothing = float(
            getattr(criterion, "label_smoothing", label_smoothing)
        )
        ignore_index = int(getattr(criterion, "ignore_index", -100))
        weight = getattr(criterion, "weight", None)

    return nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction="sum",
        label_smoothing=label_smoothing,
    )


def _infer_num_experts(model: nn.Module, cfg: Any) -> int:
    """从 cfg 或当前模型属性读取 expert 数量。"""
    value = _cfg_get(cfg, "num_experts", None)
    if value is None:
        value = getattr(model, "num_experts", None)
    if value is None and hasattr(model, "moe_head"):
        value = getattr(model.moe_head, "num_experts", None)
    if value is None:
        raise ValueError("无法推断 num_experts。")

    num_experts = int(value)
    if num_experts <= 0:
        raise ValueError("num_experts 必须大于 0。")
    return num_experts


def _infer_topk(model: nn.Module, cfg: Any) -> Optional[int]:
    """读取期望 topk；无法读取时由首个 evidence batch 确定。"""
    value = _cfg_get(cfg, "topk", None)
    if value is None:
        value = getattr(model, "topk", None)
    if value is None and hasattr(model, "moe_head"):
        value = getattr(model.moe_head, "topk", None)
    if value is None:
        return None

    topk = int(value)
    if topk <= 0:
        raise ValueError("topk 必须大于 0。")
    return topk


def _check_fisher_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fisher 对角项必须有限且非负。"""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} 的 Fisher 包含 NaN 或 Inf。")
    if torch.any(tensor < 0):
        raise FloatingPointError(f"{name} 的 Fisher 出现负数。")


def _infer_model_device(model: nn.Module) -> torch.device:
    """从模型参数推断设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("模型没有参数，无法推断 device。") from exc


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """兼容 dict、ConfigNode 和普通对象，并支持点号路径。"""
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        value = cfg.get(key, None)
        if value is not None:
            return value

    current = cfg
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "get"):
            value = current.get(part, None)
            if value is None:
                return default
            current = value
        else:
            return default

    return current
