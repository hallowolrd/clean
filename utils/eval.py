from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalResult:
    """
    评估结果。

    loss:
        测试集平均 loss。

    acc:
        Top-1 准确率，百分比形式。
        例如 63.25 表示 63.25%。

    correct:
        预测正确的样本数。

    total:
        总样本数。

    extra:
        预留额外评估指标。
        例如后面可以放 top5_acc、router_usage 等。
    """

    loss: float
    acc: float
    correct: int
    total: int
    extra: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写日志 / csv / json。
        """
        return {
            "loss": float(self.loss),
            "acc": float(self.acc),
            "correct": int(self.correct),
            "total": int(self.total),
            "extra": dict(self.extra),
        }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    criterion: Optional[nn.Module] = None,
) -> EvalResult:
    """
    在测试集上评估模型。

    这个函数默认只使用分类 logits。
    不会使用 aux_loss。
    不会加入 router balance。
    不会加入 entropy。
    不会加入 diversity。
    不会加入 consistency。

    参数：
        model:
            待评估模型。

        data_loader:
            测试集 DataLoader。

        device:
            评估设备，例如 "cuda" 或 "cpu"。

        criterion:
            loss 函数。
            如果为 None，则默认使用 CrossEntropyLoss。

    返回：
        EvalResult
    """
    device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        images, targets = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = extract_logits(outputs)

        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct(logits, targets)
        total_samples += batch_size

    if total_samples <= 0:
        raise ValueError("评估集为空，无法计算指标。")

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples

    return EvalResult(
        loss=avg_loss,
        acc=acc,
        correct=total_correct,
        total=total_samples,
        extra={},
    )


@torch.inference_mode()
def evaluate_topk(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    topk: Sequence[int] = (1, 5),
    criterion: Optional[nn.Module] = None,
) -> EvalResult:
    """
    支持 Top-k 的评估函数。

    第一版主流程可以先用 evaluate()。
    这个函数主要给 CIFAR100 或后续诊断预留。

    返回：
        EvalResult.extra 里会包含：
            top1_acc
            top5_acc
            ...
    """
    device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    topk = tuple(sorted(set(int(k) for k in topk)))
    max_k = max(topk)

    total_loss = 0.0
    total_samples = 0
    topk_correct = {
        k: 0
        for k in topk
    }

    for batch in data_loader:
        images, targets = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = extract_logits(outputs)

        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        batch_topk_correct = count_topk_correct(
            logits=logits,
            targets=targets,
            topk=topk,
            max_k=max_k,
        )

        for k, value in batch_topk_correct.items():
            topk_correct[k] += value

    if total_samples <= 0:
        raise ValueError("评估集为空，无法计算指标。")

    avg_loss = total_loss / total_samples

    extra = {
        f"top{k}_acc": 100.0 * correct / total_samples
        for k, correct in topk_correct.items()
    }

    top1_acc = extra.get("top1_acc", 0.0)

    return EvalResult(
        loss=avg_loss,
        acc=top1_acc,
        correct=topk_correct.get(1, 0),
        total=total_samples,
        extra=extra,
    )


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 DataLoader batch 中取出 images 和 targets。

    支持常见格式：
        1. (images, targets)
        2. [images, targets]
        3. {"image": images, "label": targets}
        4. {"x": images, "y": targets}

    当前 CIFAR 默认是第一种。
    """
    if isinstance(batch, Mapping):
        if "image" in batch and "label" in batch:
            return batch["image"], batch["label"]

        if "images" in batch and "labels" in batch:
            return batch["images"], batch["labels"]

        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]

        raise KeyError(
            "不支持的 batch dict 格式。"
            "需要包含 image/label、images/labels 或 x/y。"
        )

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(
        f"不支持的 batch 类型：{type(batch)}。"
        "期望 batch 是 tuple/list 或 dict。"
    )


def extract_logits(outputs: Any) -> torch.Tensor:
    """
    从模型输出中提取 logits。

    支持：
        1. 直接返回 logits tensor
        2. 返回对象，且对象有 .logits
        3. 返回 dict，且 dict["logits"]
        4. 返回 tuple/list，默认第一个元素是 logits

    这样可以兼容：
        logits = model(x)

    也可以兼容：
        output = model(x, return_router_info=True)
        logits = output.logits
    """
    if torch.is_tensor(outputs):
        return outputs

    if hasattr(outputs, "logits"):
        logits = outputs.logits
        if not torch.is_tensor(logits):
            raise TypeError("outputs.logits 不是 torch.Tensor。")
        return logits

    if isinstance(outputs, Mapping):
        if "logits" not in outputs:
            raise KeyError("模型输出 dict 中缺少 logits。")

        logits = outputs["logits"]

        if not torch.is_tensor(logits):
            raise TypeError('outputs["logits"] 不是 torch.Tensor。')

        return logits

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        logits = outputs[0]

        if not torch.is_tensor(logits):
            raise TypeError("模型输出 tuple/list 的第一个元素不是 torch.Tensor。")

        return logits

    raise TypeError(
        f"无法从模型输出中提取 logits，输出类型：{type(outputs)}"
    )


def count_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    """
    统计 Top-1 预测正确数量。
    """
    preds = logits.argmax(dim=1)
    correct = preds.eq(targets).sum().item()

    return int(correct)


def count_topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int],
    max_k: Optional[int] = None,
) -> Dict[int, int]:
    """
    统计 Top-k 预测正确数量。

    返回：
        {
            1: top1_correct,
            5: top5_correct,
            ...
        }
    """
    topk = tuple(sorted(set(int(k) for k in topk)))

    if len(topk) == 0:
        raise ValueError("topk 不能为空。")

    if max_k is None:
        max_k = max(topk)

    max_k = int(max_k)

    if max_k <= 0:
        raise ValueError(f"max_k 必须大于 0，当前值：{max_k}")

    if max_k > logits.size(1):
        max_k = int(logits.size(1))

    _, pred = logits.topk(
        k=max_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

    result: Dict[int, int] = {}

    for k in topk:
        actual_k = min(k, logits.size(1))
        correct_k = correct[:actual_k].reshape(-1).float().sum().item()
        result[k] = int(correct_k)

    return result