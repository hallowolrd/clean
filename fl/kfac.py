from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import build_train_transform
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
    在本地训练完成后的 local_model 上，额外跑一遍数据来采集 expert Linear 层的 K-FAC 因子。

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
    5. 这里只支持 after_train 采集时机，和 FedFisher 的“先得到本地模型再算 Fisher”流程对齐。
    6. 如果 kfac.disable_augmentation_for_collect=true，则 K-FAC 采集时临时关闭训练集随机增强。
       注意：model_mode=eval 只能控制模型模式，不能关闭 Dataset transform 里的 RandomCrop/Flip。
    """
    if device is None:
        device = _infer_model_device(model)

    device = torch.device(device)

    include_bias = bool(_cfg_get(cfg, "kfac.include_bias", True))
    min_count = int(_cfg_get(cfg, "kfac.min_count", 1))
    max_batches = int(_cfg_get(cfg, "kfac.max_batches", 0))
    expert_name_pattern = str(_cfg_get(cfg, "kfac.expert_name_pattern", "experts."))
    model_mode = str(_cfg_get(cfg, "kfac.model_mode", "eval")).lower().strip()
    disable_augmentation_for_collect = _cfg_bool(
        cfg,
        "kfac.disable_augmentation_for_collect",
        False,
    )

    fisher_timing = str(
        _cfg_get(
            cfg,
            "kfac.fisher_timing",
            _cfg_get(cfg, "kfac.collect_timing", "after_train"),
        )
    ).lower().strip()

    if fisher_timing != "after_train":
        raise ValueError(
            "当前 collect_expert_kfac 只支持 kfac.fisher_timing=after_train。"
            f"当前值：{fisher_timing}。"
            "请在客户端本地训练完成后再单独采集 K-FAC。"
        )

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

    sum_criterion = _build_sum_criterion(
        cfg=cfg,
        fallback_criterion=criterion,
    )
    sum_criterion.to(device)

    kfac_loader, restore_data_transform, augmentation_disabled = _prepare_kfac_loader(
        train_loader=train_loader,
        cfg=cfg,
        disable_augmentation_for_collect=disable_augmentation_for_collect,
    )

    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            for batch_idx, batch in enumerate(kfac_loader):
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

                # 只采集 Fisher/K-FAC，不更新参数。
                model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()

        restore_data_transform()

        model.zero_grad(set_to_none=True)
        model.train(was_training)

    payload: ExpertKFACPayload = {}

    for module_name, module_buffer in buffers.items():
        layer_payload = module_buffer.to_payload(min_count=min_count)
        if layer_payload is None:
            continue

        layer_payload["fisher_timing"] = fisher_timing
        layer_payload["collect_timing"] = fisher_timing
        layer_payload["model_mode"] = model_mode
        layer_payload["max_batches"] = int(max_batches)
        layer_payload["expert_name_pattern"] = expert_name_pattern

        # 记录这次 K-FAC 采集是否尝试关闭随机增强，方便日志诊断。
        layer_payload["disable_augmentation_for_collect"] = bool(
            disable_augmentation_for_collect
        )
        layer_payload["augmentation_disabled"] = bool(augmentation_disabled)

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
            "fisher_timing": "",
            "model_mode": "",
            "disable_augmentation_for_collect": False,
            "augmentation_disabled": False,
        }

    counts = [int(item["count"]) for item in payload.values()]
    trace_A = [float(item["trace_A"]) for item in payload.values()]
    trace_B = [float(item["trace_B"]) for item in payload.values()]

    fisher_timings = sorted(
        {
            str(item.get("fisher_timing", item.get("collect_timing", "")))
            for item in payload.values()
            if str(item.get("fisher_timing", item.get("collect_timing", ""))) != ""
        }
    )
    model_modes = sorted(
        {
            str(item.get("model_mode", ""))
            for item in payload.values()
            if str(item.get("model_mode", "")) != ""
        }
    )

    disable_aug_values = [
        bool(item.get("disable_augmentation_for_collect", False))
        for item in payload.values()
    ]
    aug_disabled_values = [
        bool(item.get("augmentation_disabled", False))
        for item in payload.values()
    ]

    return {
        "num_layers": int(len(payload)),
        "total_count": int(sum(counts)),
        "mean_count": float(sum(counts) / max(len(counts), 1)),
        "mean_trace_A": float(sum(trace_A) / max(len(trace_A), 1)),
        "mean_trace_B": float(sum(trace_B) / max(len(trace_B), 1)),
        "max_trace_A": float(max(trace_A)),
        "max_trace_B": float(max(trace_B)),
        "fisher_timing": (
            fisher_timings[0] if len(fisher_timings) == 1 else ",".join(fisher_timings)
        ),
        "model_mode": (
            model_modes[0] if len(model_modes) == 1 else ",".join(model_modes)
        ),
        "disable_augmentation_for_collect": bool(any(disable_aug_values)),
        "augmentation_disabled": bool(any(aug_disabled_values)),
    }


def _prepare_kfac_loader(
    train_loader: DataLoader,
    cfg: Any,
    disable_augmentation_for_collect: bool,
) -> Tuple[DataLoader, Any, bool]:
    """
    准备 K-FAC 采集用 DataLoader。

    为什么不直接用 model.eval()：
    - model.eval() 只会关闭 Dropout / BN 的训练行为；
    - Dataset transform 里的 RandomCrop / RandomHorizontalFlip 仍然会执行。

    为什么这里新建一个 num_workers=0 的 DataLoader：
    - 当前项目训练 DataLoader 可能使用 persistent_workers；
    - 训练阶段的 worker 进程里已经拷贝了带随机增强的 dataset；
    - 直接修改主进程 dataset.transform 不一定能影响已有 worker；
    - 新建 num_workers=0 的 loader 可以确保这次 K-FAC 采集使用主进程中的 clean transform。
    """
    if not disable_augmentation_for_collect:
        return train_loader, _noop, False

    dataset_name = str(_cfg_get(cfg, "dataset", "")).lower().strip()
    if dataset_name == "":
        return train_loader, _noop, False

    clean_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=False,
    )

    restore_transform, patched = _temporarily_replace_dataset_transform(
        dataset=train_loader.dataset,
        new_transform=clean_transform,
    )

    if not patched:
        return train_loader, restore_transform, False

    # K-FAC 采集不需要 shuffle。关闭 shuffle 能减少额外随机性。
    kfac_loader = DataLoader(
        train_loader.dataset,
        batch_size=getattr(train_loader, "batch_size", None),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(train_loader, "pin_memory", False)),
        drop_last=bool(getattr(train_loader, "drop_last", False)),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )

    return kfac_loader, restore_transform, True


def _temporarily_replace_dataset_transform(
    dataset: Any,
    new_transform: Any,
) -> Tuple[Any, bool]:
    """
    临时替换 dataset.transform。

    支持常见结构：
    - CIFAR10 / CIFAR100 dataset 本身有 transform
    - torch.utils.data.Subset 包着原始 dataset
    - 多层 Subset 嵌套

    返回：
        restore_fn: 调用后恢复原 transform
        patched: 是否真的替换成功
    """
    transform_datasets = []
    _collect_datasets_with_transform(
        dataset=dataset,
        output=transform_datasets,
        visited=set(),
    )

    if len(transform_datasets) == 0:
        return _noop, False

    old_transforms = []
    for item in transform_datasets:
        old_transforms.append((item, getattr(item, "transform")))
        setattr(item, "transform", new_transform)

    def restore() -> None:
        for item, old_transform in old_transforms:
            setattr(item, "transform", old_transform)

    return restore, True


def _collect_datasets_with_transform(
    dataset: Any,
    output: list[Any],
    visited: set[int],
) -> None:
    """
    递归找到带 transform 属性的底层 dataset。
    """
    if dataset is None:
        return

    dataset_id = id(dataset)
    if dataset_id in visited:
        return
    visited.add(dataset_id)

    if hasattr(dataset, "transform"):
        output.append(dataset)

    # torch.utils.data.Subset 常见字段是 dataset。
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None:
        _collect_datasets_with_transform(
            dataset=child_dataset,
            output=output,
            visited=visited,
        )


def _noop() -> None:
    """空恢复函数，用于不需要恢复 transform 的情况。"""
    return None


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
        label_smoothing = float(
            getattr(fallback_criterion, "label_smoothing", label_smoothing)
        )
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


def _cfg_bool(
    cfg: Any,
    key: str,
    default: bool,
) -> bool:
    """读取 bool 配置，兼容 true/false 字符串。"""
    value = _cfg_get(cfg, key, default)

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.lower().strip() in {"1", "true", "yes", "y", "on"}

    return bool(value)