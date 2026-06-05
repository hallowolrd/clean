from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.eval import extract_logits, unpack_batch


KFACLayerPayload = Dict[str, Any]
ExpertKFACPayload = Dict[str, KFACLayerPayload]


@dataclass
class _KFACLayerBuffer:
    """
    单个 expert Linear 层的 K-FAC 统计缓存。

    对 Linear 层 z = W a + b：
      A = E[a a^T]
      B = E[delta delta^T]

    注意：
      1. 这里累计的是 sum，最后导出时再除以 count 得到 mean。
      2. include_bias=True 时，把 bias 合并到 activation 里：
         a_aug = [a, 1]
         W_aug = [W, b]
      3. B 必须用每个样本/token 的 grad_output 外积后求和，
         不能先平均 grad_output 再外积，否则会出现梯度抵消。
    """

    module_name: str
    module: nn.Linear
    include_bias: bool

    A_sum: Optional[torch.Tensor] = None
    B_sum: Optional[torch.Tensor] = None
    a_count: int = 0
    b_count: int = 0

    def add_activation(self, activation: torch.Tensor) -> None:
        """累计 A_sum += a^T a。"""
        if activation is None:
            return

        a = _flatten_last_dim(
            tensor=activation,
            expected_dim=self.module.in_features,
            tensor_name=f"{self.module_name}.activation",
        )
        if a.numel() == 0 or a.size(0) <= 0:
            return

        a = a.detach().float()

        if self.include_bias and self.module.bias is not None:
            ones = torch.ones(
                a.size(0),
                1,
                device=a.device,
                dtype=a.dtype,
            )
            a = torch.cat([a, ones], dim=1)

        A_batch = a.transpose(0, 1).matmul(a)

        if self.A_sum is None:
            self.A_sum = torch.zeros_like(A_batch)
        self.A_sum.add_(A_batch)
        self.a_count += int(a.size(0))

    def add_grad_output(self, grad_output: torch.Tensor) -> None:
        """累计 B_sum += delta^T delta。"""
        if grad_output is None:
            return

        delta = _flatten_last_dim(
            tensor=grad_output,
            expected_dim=self.module.out_features,
            tensor_name=f"{self.module_name}.grad_output",
        )
        if delta.numel() == 0 or delta.size(0) <= 0:
            return

        delta = delta.detach().float()
        B_batch = delta.transpose(0, 1).matmul(delta)

        if self.B_sum is None:
            self.B_sum = torch.zeros_like(B_batch)
        self.B_sum.add_(B_batch)
        self.b_count += int(delta.size(0))

    def to_payload(self, min_count: int) -> Optional[KFACLayerPayload]:
        """
        导出 A_mean / B_mean / count。

        count 使用 a_count 和 b_count 的较小值。
        正常情况下两者应该相等；如果不等，说明某些 forward 没有对应 backward，
        这里保守使用 min，避免服务端误放大证据。
        """
        if self.A_sum is None or self.B_sum is None:
            return None

        if self.a_count <= 0 or self.b_count <= 0:
            return None

        count = min(int(self.a_count), int(self.b_count))
        if count < int(min_count):
            return None

        A_mean = self.A_sum / float(self.a_count)
        B_mean = self.B_sum / float(self.b_count)

        if not torch.isfinite(A_mean).all():
            return None
        if not torch.isfinite(B_mean).all():
            return None

        bias_name = None
        if self.module.bias is not None:
            bias_name = f"{self.module_name}.bias"

        return {
            "module_name": self.module_name,
            "weight_name": f"{self.module_name}.weight",
            "bias_name": bias_name,
            "A": A_mean.detach().cpu(),
            "B": B_mean.detach().cpu(),
            "count": int(count),
            "a_count": int(self.a_count),
            "b_count": int(self.b_count),
            "include_bias": bool(self.include_bias and self.module.bias is not None),
            "in_features": int(self.module.in_features),
            "out_features": int(self.module.out_features),
            "trace_A": float(torch.trace(A_mean).detach().cpu().item()),
            "trace_B": float(torch.trace(B_mean).detach().cpu().item()),
        }


def collect_expert_kfac(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device | str | None = None,
    cfg: Any = None,
) -> ExpertKFACPayload:
    """
    采集 expert 内部 Linear 层的 K-FAC 因子。

    返回格式：
        {
            "switch_layers.0.switch_ffn.experts.2.0": {
                "module_name": ...,
                "weight_name": "...weight",
                "bias_name": "...bias",
                "A": Tensor[in_dim(+1), in_dim(+1)],
                "B": Tensor[out_dim, out_dim],
                "count": int,
                ...
            },
            ...
        }

    设计约束：
      1. 只采集 module name 包含 experts. 的 nn.Linear。
      2. 默认使用 CrossEntropyLoss(reduction="sum")，避免 mean loss 缩放梯度。
      3. 默认 model.eval() 采集，避免 Dropout / BN 引入额外随机性。
      4. 不修改训练逻辑，不做 optimizer.step()。
    """
    if device is None:
        device = _infer_model_device(model)
    device = torch.device(device)

    include_bias = bool(_cfg_get(cfg, "kfac.include_bias", True))
    min_count = int(_cfg_get(cfg, "kfac.min_count", 1))
    max_batches = int(_cfg_get(cfg, "kfac.max_batches", 0))
    expert_name_pattern = str(_cfg_get(cfg, "kfac.expert_name_pattern", "experts."))
    model_mode = str(_cfg_get(cfg, "kfac.model_mode", "eval")).lower().strip()

    if min_count <= 0:
        min_count = 1

    buffers: Dict[str, _KFACLayerBuffer] = {}
    handles = []

    for module_name, module in model.named_modules():
        if not _is_expert_linear(
            module_name=module_name,
            module=module,
            expert_name_pattern=expert_name_pattern,
        ):
            continue

        buffers[module_name] = _KFACLayerBuffer(
            module_name=module_name,
            module=module,
            include_bias=include_bias,
        )

    if len(buffers) == 0:
        return {}

    for module_name, module_buffer in buffers.items():
        module = module_buffer.module

        def forward_hook(
            layer: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            name: str = module_name,
        ) -> None:
            if len(inputs) == 0:
                return
            buffers[name].add_activation(inputs[0])

        def backward_hook(
            layer: nn.Module,
            grad_input: tuple[Optional[torch.Tensor], ...],
            grad_output: tuple[Optional[torch.Tensor], ...],
            name: str = module_name,
        ) -> None:
            if len(grad_output) == 0:
                return
            buffers[name].add_grad_output(grad_output[0])

        handles.append(module.register_forward_hook(forward_hook))
        handles.append(module.register_full_backward_hook(backward_hook))

    was_training = bool(model.training)
    model.to(device)

    if model_mode == "train":
        model.train()
    else:
        model.eval()

    sum_criterion = _build_sum_criterion(cfg=cfg, fallback_criterion=criterion)
    sum_criterion.to(device)

    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            for batch_idx, batch in enumerate(train_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                images, targets = unpack_batch(batch)
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                model.zero_grad(set_to_none=True)

                outputs = model(images)
                logits = extract_logits(outputs)
                loss = sum_criterion(logits, targets)

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                # 只采集 Fisher，不更新参数。
                model.zero_grad(set_to_none=True)

    finally:
        for handle in handles:
            handle.remove()

        model.zero_grad(set_to_none=True)
        model.train(was_training)

    payload: ExpertKFACPayload = {}
    for module_name, module_buffer in buffers.items():
        layer_payload = module_buffer.to_payload(min_count=min_count)
        if layer_payload is None:
            continue
        payload[module_name] = layer_payload

    return payload


def summarize_expert_kfac(payload: ExpertKFACPayload) -> Dict[str, Any]:
    """
    生成轻量诊断信息，方便 client.py 或日志系统记录。
    不包含 A/B tensor 本体。
    """
    if not payload:
        return {
            "num_layers": 0,
            "total_count": 0,
            "mean_count": 0.0,
            "mean_trace_A": 0.0,
            "mean_trace_B": 0.0,
            "max_trace_A": 0.0,
            "max_trace_B": 0.0,
        }

    counts = [int(item["count"]) for item in payload.values()]
    trace_A = [float(item["trace_A"]) for item in payload.values()]
    trace_B = [float(item["trace_B"]) for item in payload.values()]

    return {
        "num_layers": int(len(payload)),
        "total_count": int(sum(counts)),
        "mean_count": float(sum(counts) / max(len(counts), 1)),
        "mean_trace_A": float(sum(trace_A) / max(len(trace_A), 1)),
        "mean_trace_B": float(sum(trace_B) / max(len(trace_B), 1)),
        "max_trace_A": float(max(trace_A)),
        "max_trace_B": float(max(trace_B)),
    }


def _is_expert_linear(
    module_name: str,
    module: nn.Module,
    expert_name_pattern: str,
) -> bool:
    """判断一个 module 是否是 expert 内部的 Linear 层。"""
    if not isinstance(module, nn.Linear):
        return False
    if expert_name_pattern not in module_name:
        return False
    return True


def _flatten_last_dim(
    tensor: torch.Tensor,
    expected_dim: int,
    tensor_name: str,
) -> torch.Tensor:
    """
    把 Linear 的输入或 grad_output 展平成 [num_items, feature_dim]。

    支持：
      [N, D]
      [B, N, D]
      [B, ..., D]
    """
    if tensor is None:
        raise ValueError(f"{tensor_name} 为空。")

    if tensor.dim() == 0:
        raise ValueError(f"{tensor_name} 维度错误：{tuple(tensor.shape)}")

    if tensor.size(-1) != int(expected_dim):
        raise ValueError(
            f"{tensor_name} 最后一维不匹配："
            f"实际={tensor.size(-1)}, 期望={expected_dim}, "
            f"shape={tuple(tensor.shape)}"
        )

    if tensor.dim() == 1:
        return tensor.reshape(1, -1)

    return tensor.reshape(-1, tensor.size(-1))


def _build_sum_criterion(
    cfg: Any,
    fallback_criterion: Optional[nn.Module] = None,
) -> nn.Module:
    """
    构建 K-FAC 采集用 loss。

    这里强制 reduction='sum'。
    如果直接复用训练时 CrossEntropyLoss 的 mean reduction，
    backward 得到的 delta 会被 batch size 缩小，K-FAC 尺度会不稳定。
    """
    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))

    if isinstance(fallback_criterion, nn.CrossEntropyLoss):
        label_smoothing = float(getattr(fallback_criterion, "label_smoothing", label_smoothing))
        ignore_index = int(getattr(fallback_criterion, "ignore_index", -100))
        weight = getattr(fallback_criterion, "weight", None)
        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=label_smoothing,
        )

    return nn.CrossEntropyLoss(
        reduction="sum",
        label_smoothing=label_smoothing,
    )


def _infer_model_device(model: nn.Module) -> torch.device:
    """从模型参数推断 device。"""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("模型没有参数，无法推断 device。") from exc


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的读取。

    支持：
      cfg.get("kfac.include_bias", True)
      cfg["kfac"]["include_bias"]
      cfg.kfac.include_bias
    """
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
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        return default

    return current