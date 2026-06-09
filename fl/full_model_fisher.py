from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.eval import extract_logits, unpack_batch


@dataclass(frozen=True)
class FullModelFisherStats:
    """
    纯 FL 场景下的整模型 Fisher evidence 统计结果。

    说明：
    - 这里统计的是 full-model 级别的 Fisher strength。
    - 它不是 expert-wise Fisher。
    - 它不依赖 router、expert_id、active_count、K-FAC hook。
    - 后续 pure-FL 聚合器可以用 fisher_strength 给客户端分配聚合权重。

    字段：
        fisher_strength:
            当前客户端整模型的 Fisher 强度标量。
            第一版采用经验 Fisher 的简化近似：mean(grad^2)。

        avg_loss:
            evidence pass 上的平均 CrossEntropy loss。

        total_samples:
            evidence pass 实际处理的样本数。

        total_batches:
            evidence pass 实际处理的 batch 数。

        num_tensors:
            至少产生过一次梯度的参数张量数量。

        num_grad_elements:
            参与 Fisher 统计的梯度元素数量。

        max_batches:
            evidence pass 最多使用多少个 batch。
            None 表示使用完整 evidence_loader。

        model_mode:
            evidence pass 时模型使用的模式。
            默认 eval，避免 BatchNorm running stats 被 evidence pass 改动。
    """

    fisher_strength: float
    avg_loss: float
    total_samples: int
    total_batches: int
    num_tensors: int
    num_grad_elements: int
    max_batches: Optional[int]
    model_mode: str

    def to_payload(self) -> Dict[str, Any]:
        """
        转成 ClientUpdate.extra["global_fisher"] 可以直接保存的普通 dict。

        推荐客户端侧写法：
            full_model_fisher = collect_full_model_fisher_stats(...)
            extra["global_fisher"] = full_model_fisher.to_payload()
        """
        score = float(self.total_samples) * float(self.fisher_strength)

        return {
            "fisher_strength": float(self.fisher_strength),
            "num_samples": int(self.total_samples),
            "score": float(score),
            "meta": {
                "avg_loss": float(self.avg_loss),
                "total_samples": int(self.total_samples),
                "total_batches": int(self.total_batches),
                "num_tensors": int(self.num_tensors),
                "num_grad_elements": int(self.num_grad_elements),
                "max_batches": self.max_batches,
                "model_mode": str(self.model_mode),
            },
        }


def collect_full_model_fisher_stats(
    model: nn.Module,
    evidence_loader: DataLoader,
    device: torch.device | str,
    cfg: Any = None,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
    model_mode: Optional[str] = None,
) -> FullModelFisherStats:
    """
    统计纯 FL 场景下的整模型 Fisher evidence。

    核心思想：
        对客户端本地训练后的模型，额外跑一小段 evidence data。
        只做 forward + backward，不做 optimizer.step。
        然后用所有可训练参数的梯度平方均值作为 Fisher strength。

    近似公式：
        fisher_strength_i = mean_p( grad_p^2 )

    其中：
        i 表示客户端；
        p 表示整模型中所有可训练参数元素。

    参数：
        model:
            客户端本地训练完成后的模型。

        evidence_loader:
            用于统计 Fisher 的 DataLoader。
            推荐使用无随机增强的数据版本。
            如果暂时没有单独 evidence_loader，也可以先传 train_loader。

        device:
            统计使用的设备。

        cfg:
            全局配置。
            支持从 cfg.full_model_fisher 读取：
                full_model_fisher:
                  max_batches: 10
                  model_mode: eval

        criterion:
            loss 函数。
            如果为 None，默认使用 CrossEntropyLoss。

        max_batches:
            最多统计多少个 batch。
            如果显式传入，则优先于 cfg.full_model_fisher.max_batches。

        model_mode:
            evidence pass 使用 "eval" 还是 "train"。
            如果显式传入，则优先于 cfg.full_model_fisher.model_mode。

    返回：
        FullModelFisherStats
    """
    if evidence_loader is None:
        raise ValueError("evidence_loader 不能为空。")

    device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    fisher_cfg = _cfg_get(cfg, "full_model_fisher", {})
    resolved_max_batches = _resolve_max_batches(
        explicit_value=max_batches,
        cfg_value=_cfg_get(fisher_cfg, "max_batches", 10),
    )
    resolved_model_mode = _resolve_model_mode(
        explicit_value=model_mode,
        cfg_value=_cfg_get(fisher_cfg, "model_mode", "eval"),
    )

    was_training = bool(model.training)

    model.to(device)

    # 默认使用 eval：
    # 1. 保留 dropout 关闭；
    # 2. 避免 BatchNorm running_mean / running_var 被 evidence pass 改动；
    # 3. eval 模式下仍然可以正常 backward 统计参数梯度。
    if resolved_model_mode == "eval":
        model.eval()
    elif resolved_model_mode == "train":
        model.train()
    else:
        raise ValueError(
            "full_model_fisher.model_mode 只支持 eval 或 train，"
            f"当前值：{resolved_model_mode}"
        )

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    fisher_weighted_sum = 0.0
    num_tensors_total = 0
    num_grad_elements_total = 0

    try:
        for batch_idx, batch in enumerate(evidence_loader):
            if (
                resolved_max_batches is not None
                and batch_idx >= int(resolved_max_batches)
            ):
                break

            images, targets = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)

            outputs = model(images)
            logits = extract_logits(outputs)
            loss = criterion(logits, targets)
            loss.backward()

            batch_size = int(targets.size(0))
            batch_grad_sq_sum = 0.0
            batch_grad_elements = 0
            batch_num_tensors = 0

            for param in model.parameters():
                if not param.requires_grad:
                    continue

                if param.grad is None:
                    continue

                grad = param.grad.detach()

                if not torch.is_floating_point(grad):
                    continue

                if not torch.isfinite(grad).all():
                    raise ValueError(
                        "full_model_fisher 统计到非有限梯度，"
                        "请检查学习率、loss 或输入数据。"
                    )

                grad_float = grad.float()
                batch_grad_sq_sum += float(grad_float.pow(2).sum().item())
                batch_grad_elements += int(grad_float.numel())
                batch_num_tensors += 1

            if batch_grad_elements <= 0:
                raise ValueError(
                    "full_model_fisher 没有统计到任何有效梯度。"
                    "请检查模型参数 requires_grad 是否为 True。"
                )

            batch_fisher_strength = batch_grad_sq_sum / float(batch_grad_elements)

            # batch 之间按样本数加权，避免最后一个小 batch 权重过大。
            fisher_weighted_sum += batch_fisher_strength * batch_size

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            total_batches += 1
            num_tensors_total += batch_num_tensors
            num_grad_elements_total += batch_grad_elements

        if total_samples <= 0 or total_batches <= 0:
            raise ValueError(
                "full_model_fisher evidence pass 没有处理任何样本。"
                "请检查 evidence_loader 是否为空，或 max_batches 是否设置为 0。"
            )

        fisher_strength = fisher_weighted_sum / float(total_samples)
        avg_loss = total_loss / float(total_samples)

        if not _is_finite_number(fisher_strength):
            raise ValueError(
                f"full_model_fisher 得到非有限 fisher_strength：{fisher_strength}"
            )

        return FullModelFisherStats(
            fisher_strength=float(fisher_strength),
            avg_loss=float(avg_loss),
            total_samples=int(total_samples),
            total_batches=int(total_batches),
            num_tensors=int(num_tensors_total),
            num_grad_elements=int(num_grad_elements_total),
            max_batches=resolved_max_batches,
            model_mode=str(resolved_model_mode),
        )

    finally:
        model.zero_grad(set_to_none=True)

        # 恢复进入函数前的 train/eval 状态，避免影响后续流程。
        if was_training:
            model.train()
        else:
            model.eval()


def collect_full_model_fisher_strength(
    model: nn.Module,
    evidence_loader: DataLoader,
    device: torch.device | str,
    cfg: Any = None,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
    model_mode: Optional[str] = None,
) -> float:
    """
    只返回 fisher_strength 的轻量接口。

    如果客户端只需要一个标量，可以调用这个函数。
    如果还需要日志诊断，建议调用 collect_full_model_fisher_stats()。
    """
    stats = collect_full_model_fisher_stats(
        model=model,
        evidence_loader=evidence_loader,
        device=device,
        cfg=cfg,
        criterion=criterion,
        max_batches=max_batches,
        model_mode=model_mode,
    )
    return float(stats.fisher_strength)


def is_full_model_fisher_enabled(cfg: Any) -> bool:
    """
    判断是否启用纯 FL 整模型 Fisher evidence 统计。

    推荐配置：
        full_model_fisher:
          enabled: true
          max_batches: 10
          model_mode: eval

    注意：
    - 这个开关和 expert_fisher.enabled 是分开的。
    - expert_fisher 是 FL+MoE expert-wise K-FAC。
    - full_model_fisher 是 pure-FL client-wise Fisher。
    """
    fisher_cfg = _cfg_get(cfg, "full_model_fisher", {})
    return bool(_cfg_get(fisher_cfg, "enabled", False))


def _resolve_max_batches(
    explicit_value: Optional[int],
    cfg_value: Any,
) -> Optional[int]:
    """
    解析 max_batches。

    规则：
    - 显式参数优先；
    - 其次使用 cfg.full_model_fisher.max_batches；
    - None 表示使用完整 evidence_loader；
    - 正整数表示最多使用多少个 batch。
    """
    value = explicit_value if explicit_value is not None else cfg_value

    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"none", "null", "all", "full"}:
        return None

    value = int(value)

    if value <= 0:
        raise ValueError(f"max_batches 必须为正整数或 None，当前值：{value}")

    return value


def _resolve_model_mode(
    explicit_value: Optional[str],
    cfg_value: Any,
) -> str:
    """
    解析 evidence pass 的模型模式。

    支持：
        eval
        train

    默认 eval。
    """
    value = explicit_value if explicit_value is not None else cfg_value
    value = str(value).lower().strip()

    if value not in {"eval", "train"}:
        raise ValueError(
            "full_model_fisher.model_mode 只支持 eval 或 train，"
            f"当前值：{value}"
        )

    return value


def _is_finite_number(value: float) -> bool:
    """
    判断 Python float 是否为有限数。
    """
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return bool(torch.isfinite(tensor).item())


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的配置读取。

    支持：
    - dict.get(key, default)
    - ConfigNode.get(key, default)
    - getattr(cfg, key, default)
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)