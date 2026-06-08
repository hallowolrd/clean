from __future__ import annotations

import gc
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.eval import extract_logits, unpack_batch


# 参数名 / module 名中只要出现 experts.<id>，就认为它属于某个 expert。
# 例如：
#   moe_head.experts.0.fc1 -> expert 0
#   moe_head.experts.3.fc2 -> expert 3
_EXPERT_ID_PATTERN = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


@dataclass
class _LayerKFACState:
    """
    单个 expert Linear 层的 K-FAC 统计状态。

    A_sum:
        输入激活二阶统计之和：
            A_sum = sum_r a_r a_r^T

    B_sum:
        反向梯度二阶统计之和：
            B_sum = sum_r delta_r delta_r^T

    a_count:
        A 统计中实际参与的 token / activation 行数。

    b_count:
        B 统计中实际参与的 token / activation 行数。

    注意：
        这里先保存 sum，不在 hook 里直接除以 count。
        evidence pass 结束后统一导出 mean_A / mean_B。
    """

    module_name: str
    expert_id: int
    layer_name: str
    A_sum: Optional[torch.Tensor] = None
    B_sum: Optional[torch.Tensor] = None
    a_count: int = 0
    b_count: int = 0

    def add_A(self, A_batch: torch.Tensor, count: int) -> None:
        """累加输入激活二阶统计。"""
        if count <= 0:
            return

        A_batch = A_batch.detach().to(device="cpu", dtype=torch.float32)

        if self.A_sum is None:
            self.A_sum = torch.zeros_like(A_batch, device="cpu", dtype=torch.float32)

        self.A_sum.add_(A_batch)
        self.a_count += int(count)

    def add_B(self, B_batch: torch.Tensor, count: int) -> None:
        """累加反向梯度二阶统计。"""
        if count <= 0:
            return

        B_batch = B_batch.detach().to(device="cpu", dtype=torch.float32)

        if self.B_sum is None:
            self.B_sum = torch.zeros_like(B_batch, device="cpu", dtype=torch.float32)

        self.B_sum.add_(B_batch)
        self.b_count += int(count)


class ExpertKFACCollector:
    """
    只针对 expert Linear 层的 K-FAC evidence 采集器。

    设计目标：
        1. 只统计 expert 参数，不统计 backbone / router / norm。
        2. 本地训练完成后额外跑一轮 evidence forward + backward。
        3. 不做 optimizer.step，不修改模型参数。
        4. 使用 forward hook 统计 A，backward hook 统计 B。
        5. 不使用 module.weight.grad ** 2，避免 batch 梯度抵消。

    K-FAC 统计：
        对 Linear 层 z = W a + b：

            A = 1/N sum_r a_r a_r^T
            B = 1/N sum_r delta_r delta_r^T

        其中：
            a_r      是该 expert Linear 层的第 r 个输入 token / activation
            delta_r  是该 Linear 输出端的第 r 个反向梯度
            N        是该层实际收到的 token / activation 行数

    为什么不用 weight.grad：
        module.weight.grad 是 batch 内所有 token 梯度先求和后的结果。
        如果不同 token 的梯度方向相反，会先抵消再平方，导致 Fisher 被低估。
        这里直接统计每个 token 的二阶量，所以不会出现这个问题。
    """

    def __init__(
        self,
        model: nn.Module,
        include_bias: bool = True,
    ) -> None:
        self.model = model
        self.include_bias = bool(include_bias)

        self.layer_states: Dict[str, _LayerKFACState] = {}
        self.handles: List[Any] = []

    def register(self) -> None:
        """
        注册 expert Linear 层的 forward / backward hook。

        只 hook 名字中包含 experts.<id> 的 nn.Linear。
        """
        self.remove()

        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            expert_id = get_expert_id_from_name(module_name)
            if expert_id is None:
                continue

            layer_state = _LayerKFACState(
                module_name=module_name,
                expert_id=int(expert_id),
                layer_name=module_name,
            )
            self.layer_states[module_name] = layer_state

            self.handles.append(
                module.register_forward_hook(
                    self._make_forward_hook(module_name=module_name)
                )
            )
            self.handles.append(
                module.register_full_backward_hook(
                    self._make_backward_hook(module_name=module_name)
                )
            )

        if len(self.layer_states) == 0:
            raise ValueError(
                "没有找到任何 expert Linear 层。"
                "请确认模型中的 expert 参数名是否包含 experts.<id>，"
                "并且 expert 内部使用 nn.Linear。"
            )

    def remove(self) -> None:
        """移除所有 hook，防止下一轮重复注册导致统计翻倍。"""
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def _make_forward_hook(self, module_name: str):
        """构造 forward hook，用于统计 A。"""

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if len(inputs) == 0:
                return

            x = inputs[0]
            if not torch.is_tensor(x):
                return

            if x.numel() == 0:
                return

            # Linear 输入通常是 [N, d_in]。
            # 如果未来变成 [B, T, d_in]，这里也会统一展平成 [B*T, d_in]。
            a = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32)

            if a.size(0) <= 0:
                return

            if self.include_bias and isinstance(module, nn.Linear):
                if module.bias is not None:
                    ones = torch.ones(
                        a.size(0),
                        1,
                        device=a.device,
                        dtype=a.dtype,
                    )
                    a = torch.cat([a, ones], dim=1)

            # A_batch = sum_r a_r a_r^T
            A_batch = a.transpose(0, 1).matmul(a)

            layer_state = self.layer_states[module_name]
            layer_state.add_A(A_batch=A_batch, count=int(a.size(0)))

        return hook

    def _make_backward_hook(self, module_name: str):
        """构造 backward hook，用于统计 B。"""

        def hook(
            module: nn.Module,
            grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
        ) -> None:
            if len(grad_output) == 0:
                return

            g = grad_output[0]
            if not torch.is_tensor(g):
                return

            if g.numel() == 0:
                return

            # Linear 输出端梯度通常是 [N, d_out]。
            # 如果未来变成 [B, T, d_out]，这里统一展平成 [B*T, d_out]。
            delta = g.detach().reshape(-1, g.shape[-1]).to(dtype=torch.float32)

            if delta.size(0) <= 0:
                return

            # B_batch = sum_r delta_r delta_r^T
            B_batch = delta.transpose(0, 1).matmul(delta)

            layer_state = self.layer_states[module_name]
            layer_state.add_B(B_batch=B_batch, count=int(delta.size(0)))

        return hook

    def export(self, eps: float = 1.0e-8) -> Dict[str, Any]:
        """
        导出 expert K-FAC 统计结果。

        返回格式：
            {
                "experts": {
                    expert_id: {
                        "active_count": int,
                        "mean_A": float,
                        "mean_B": float,
                        "fisher_strength": float,
                        "score": float,
                        "layers": {...},
                    }
                },
                "meta": {...}
            }

        其中：
            active_count:
                当前 expert 在 evidence pass 中实际处理的 token 数。
                如果 expert 有多层 Linear，不把多层 count 相加，而是取 max。
                因为 fc1/fc2 处理的是同一批 routed token，相加会重复计数。

            mean_A / mean_B:
                多层 expert 的 count 加权平均。

            fisher_strength:
                多层 expert 的 count 加权平均 mean_A * mean_B。

            score:
                fisher_only 第一版推荐使用：
                    score = active_count * fisher_strength
        """
        eps = float(eps)

        experts: Dict[int, Dict[str, Any]] = {}
        layer_payloads: Dict[str, Dict[str, Any]] = {}

        for module_name, state in sorted(self.layer_states.items()):
            payload = _export_layer_state(state=state, eps=eps)
            layer_payloads[module_name] = payload

            expert_id = int(state.expert_id)
            expert_bucket = experts.setdefault(
                expert_id,
                {
                    "active_count": 0,
                    "mean_A_weighted_sum": 0.0,
                    "mean_B_weighted_sum": 0.0,
                    "strength_weighted_sum": 0.0,
                    "weight_count_sum": 0.0,
                    "layers": {},
                },
            )

            layer_count = int(payload["count"])
            mean_A = float(payload["mean_A"])
            mean_B = float(payload["mean_B"])
            layer_strength = float(mean_A * mean_B)

            # active_count 表示该 expert 的 routed token 数。
            # expert 多层时不能把 fc1/fc2 的 count 相加，否则会重复使用 active_count。
            expert_bucket["active_count"] = max(
                int(expert_bucket["active_count"]),
                layer_count,
            )

            if layer_count > 0:
                expert_bucket["mean_A_weighted_sum"] += layer_count * mean_A
                expert_bucket["mean_B_weighted_sum"] += layer_count * mean_B
                expert_bucket["strength_weighted_sum"] += layer_count * layer_strength
                expert_bucket["weight_count_sum"] += layer_count

            expert_bucket["layers"][module_name] = payload

        final_experts: Dict[int, Dict[str, Any]] = {}

        for expert_id, bucket in sorted(experts.items()):
            total_count = float(bucket["weight_count_sum"])
            active_count = int(bucket["active_count"])

            if total_count > 0:
                mean_A = float(bucket["mean_A_weighted_sum"] / (total_count + eps))
                mean_B = float(bucket["mean_B_weighted_sum"] / (total_count + eps))
                fisher_strength = float(
                    bucket["strength_weighted_sum"] / (total_count + eps)
                )
            else:
                mean_A = 0.0
                mean_B = 0.0
                fisher_strength = 0.0

            # fisher_only 推荐直接使用这个 score：
            #   score = active_count * fisher_strength
            #
            # 如果服务端为了保持公式简单，使用：
            #   active_count * mean_A * mean_B
            # 也可以跑，但多层 expert 时会和 count 加权 layer_strength 略有差别。
            score = float(active_count) * float(fisher_strength)

            final_experts[int(expert_id)] = {
                "expert_id": int(expert_id),
                "active_count": int(active_count),
                "mean_A": _safe_float(mean_A),
                "mean_B": _safe_float(mean_B),
                "fisher_strength": _safe_float(fisher_strength),
                "score": _safe_float(score),
                "num_layers": int(len(bucket["layers"])),
                "layers": bucket["layers"],
            }

        num_active_experts = sum(
            1
            for expert_payload in final_experts.values()
            if int(expert_payload["active_count"]) > 0
        )

        return {
            "experts": final_experts,
            "layers": layer_payloads,
            "meta": {
                "num_expert_layers": int(len(self.layer_states)),
                "num_experts": int(len(final_experts)),
                "num_active_experts": int(num_active_experts),
                "include_bias": bool(self.include_bias),
            },
        }


def collect_expert_kfac_stats(
    model: nn.Module,
    evidence_loader: DataLoader,
    device: torch.device | str,
    cfg: Any,
) -> Dict[str, Any]:
    """
    客户端本地训练完成后，额外进行一轮 expert K-FAC evidence 统计。

    这个函数只统计，不训练：
        - 会执行 forward + backward
        - 不会执行 optimizer.step
        - 不会修改模型参数
        - 不会使用 torch.no_grad，因为需要 backward
        - 推荐使用无随机增强的 evidence_loader
        - 推荐在 eval 模式下执行

    参数：
        model:
            客户端本地训练完成后的 local_model。

        evidence_loader:
            无随机数据增强的客户端 evidence DataLoader。
            它应该和训练 DataLoader 使用同一个客户端样本划分。

        device:
            统计使用的设备。

        cfg:
            全局配置。
            可选配置位置：
                cfg.expert_fisher.include_bias
                cfg.expert_fisher.model_mode
                cfg.expert_fisher.loss_reduction
                cfg.expert_fisher.max_batches
                cfg.expert_fisher.eps
                cfg.label_smooth

    返回：
        expert_kfac payload，建议放入：
            ClientUpdate.extra["expert_kfac"]
    """
    device = torch.device(device)

    expert_fisher_cfg = _cfg_get(cfg, "expert_fisher", {})
    include_bias = bool(_cfg_get(expert_fisher_cfg, "include_bias", True))
    model_mode = str(_cfg_get(expert_fisher_cfg, "model_mode", "eval")).lower()
    loss_reduction = str(_cfg_get(expert_fisher_cfg, "loss_reduction", "sum")).lower()
    max_batches = _cfg_get(expert_fisher_cfg, "max_batches", None)
    eps = float(_cfg_get(expert_fisher_cfg, "eps", 1.0e-8))

    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            max_batches = None

    if loss_reduction not in {"sum", "mean"}:
        raise ValueError(
            f"expert_fisher.loss_reduction 只支持 sum / mean，当前值：{loss_reduction}"
        )

    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))

    criterion = nn.CrossEntropyLoss(
        reduction=loss_reduction,
        label_smoothing=label_smoothing,
    )

    model.to(device)

    was_training = bool(model.training)
    if model_mode == "eval":
        model.eval()
    elif model_mode == "train":
        model.train()
    else:
        raise ValueError(
            f"expert_fisher.model_mode 只支持 eval / train，当前值：{model_mode}"
        )

    collector = ExpertKFACCollector(
        model=model,
        include_bias=include_bias,
    )

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    try:
        collector.register()

        # evidence pass 不做 optimizer.step。
        # 每个 batch 前后都清梯度，避免本地训练残留梯度或 batch 间累积影响统计。
        model.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(evidence_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, targets = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)

            # 不使用 autocast，尽量用 float32 统计 Fisher，避免半精度导致 trace 太小或溢出。
            # 注意：不能使用 torch.no_grad()，因为这里需要 backward hook 统计 B。
            outputs = model(images)
            logits = extract_logits(outputs)
            loss = criterion(logits, targets)

            loss.backward()

            batch_size = int(targets.size(0))
            if loss_reduction == "sum":
                total_loss += float(loss.item())
            else:
                total_loss += float(loss.item()) * batch_size

            total_samples += batch_size
            total_batches += 1

            model.zero_grad(set_to_none=True)

        if total_batches <= 0 or total_samples <= 0:
            raise ValueError("expert K-FAC evidence pass 没有处理任何 batch。")

        payload = collector.export(eps=eps)
        payload["meta"].update(
            {
                "total_batches": int(total_batches),
                "total_samples": int(total_samples),
                "avg_loss": float(total_loss / max(total_samples, 1)),
                "model_mode": model_mode,
                "loss_reduction": loss_reduction,
            }
        )

        return payload

    finally:
        collector.remove()

        model.zero_grad(set_to_none=True)

        if was_training:
            model.train()
        else:
            model.eval()

        del criterion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_expert_id_from_name(name: str) -> Optional[int]:
    """
    从 module / 参数名中解析 expert id。

    示例：
        moe_head.experts.0.fc1 -> 0
        experts.3.fc2 -> 3
        backbone.layer1.0.conv1 -> None
    """
    match = _EXPERT_ID_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _export_layer_state(
    state: _LayerKFACState,
    eps: float,
) -> Dict[str, Any]:
    """
    把单层 K-FAC sum 统计导出成 mean_A / mean_B 标量。
    """
    a_count = int(state.a_count)
    b_count = int(state.b_count)

    count = int(min(a_count, b_count)) if b_count > 0 else int(a_count)

    if state.A_sum is None or a_count <= 0:
        trace_A = 0.0
        dim_A = 0
        mean_A = 0.0
    else:
        dim_A = int(state.A_sum.shape[0])
        A_mean = state.A_sum / float(max(a_count, 1))
        trace_A = float(torch.trace(A_mean).item())
        mean_A = trace_A / float(max(dim_A, 1))

    if state.B_sum is None or b_count <= 0:
        trace_B = 0.0
        dim_B = 0
        mean_B = 0.0
    else:
        dim_B = int(state.B_sum.shape[0])
        B_mean = state.B_sum / float(max(b_count, 1))
        trace_B = float(torch.trace(B_mean).item())
        mean_B = trace_B / float(max(dim_B, 1))

    mean_A = _safe_float(mean_A)
    mean_B = _safe_float(mean_B)
    trace_A = _safe_float(trace_A)
    trace_B = _safe_float(trace_B)
    layer_strength = _safe_float(mean_A * mean_B)

    return {
        "module_name": str(state.module_name),
        "expert_id": int(state.expert_id),
        "layer_name": str(state.layer_name),
        "count": int(count),
        "a_count": int(a_count),
        "b_count": int(b_count),
        "dim_A": int(dim_A),
        "dim_B": int(dim_B),
        "trace_A": float(trace_A),
        "trace_B": float(trace_B),
        "mean_A": float(mean_A),
        "mean_B": float(mean_B),
        "fisher_strength": float(layer_strength),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    把数值安全转成 float。

    如果出现 NaN / Inf，则返回 default。
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(result):
        return float(default)

    return result


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