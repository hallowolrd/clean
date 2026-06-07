from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fl.kfac import collect_expert_kfac, summarize_expert_kfac
from fl.types import ClientUpdate
from utils.eval import extract_logits, unpack_batch
from utils.state_dict_ops import (
    check_finite_state_dict,
    state_dict_to,
    subtract_state_dict,
)


@dataclass(frozen=True)
class ClientTrainStats:
    """
    客户端本地训练统计结果。

    avg_loss: 本地训练平均 loss。
    train_acc: 本地训练准确率，百分比形式。
    num_samples: 本地训练样本数。
    num_batches: 本地训练 batch 数。
    """

    avg_loss: float
    train_acc: float
    num_samples: int
    num_batches: int

    def to_metrics(self) -> Dict[str, float]:
        """
        转成 ClientUpdate.metrics 使用的普通 dict。
        """
        return {
            "train_loss": float(self.avg_loss),
            "train_acc": float(self.train_acc),
            "num_batches": float(self.num_batches),
        }


class FLClient:
    """
    联邦学习客户端。

    职责：
    1. 接收 server 下发的 global_model
    2. 在自己的 train_loader 上本地训练
    3. 计算 local_model 相对 global_model 的参数变化量
    4. 返回 ClientUpdate

    不负责：
    1. 选择客户端
    2. 聚合参数
    3. 测试集评估
    4. 保存 checkpoint
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        cfg: Any,
        device: torch.device | str,
    ) -> None:
        self.client_id = int(client_id)
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = torch.device(device)

        if len(self.train_loader.dataset) <= 0:
            raise ValueError(f"客户端 {self.client_id} 的数据集为空。")

    @property
    def num_samples(self) -> int:
        """
        当前客户端本地样本数。
        """
        return int(len(self.train_loader.dataset))

    def train(
        self,
        global_model: nn.Module,
        round_id: int,
    ) -> ClientUpdate:
        """
        执行本地训练，并返回客户端更新。

        参数：
            global_model: server 当前轮下发的全局模型。
            round_id: 当前联邦训练轮数。

        返回：
            ClientUpdate: 包含 model_delta、num_samples、metrics、extra 等信息。
        """
        global_state_cpu = state_dict_to(
            global_model.state_dict(),
            device="cpu",
        )

        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        criterion = build_criterion(self.cfg)
        optimizer = build_optimizer(
            model=local_model,
            cfg=self.cfg,
        )

        local_epochs = int(_cfg_get(self.cfg, "local_epochs", 1))
        grad_clip = _get_grad_clip(self.cfg)

        stats = train_local_model(
            model=local_model,
            train_loader=self.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            local_epochs=local_epochs,
            grad_clip=grad_clip,
        )

        # ------------------------------------------------------------
        # 可选：采集当前客户端本地模型的 expert usage。
        #
        # 统计含义：
        #   本地训练结束后，local_model 在该客户端自己的 train_loader 上，
        #   每个 expert 被 top-k router 选中了多少次。
        #
        # 注意：
        #   topk=2 时，一个样本会贡献 2 次 expert 激活。
        #   所以 expert_counts 的总和通常约等于 num_samples * topk。
        # ------------------------------------------------------------
        expert_usage = None
        if bool(_cfg_get(self.cfg, "logging.collect_expert_usage", False)):
            expert_usage = collect_expert_usage(
                model=local_model,
                train_loader=self.train_loader,
                device=self.device,
                cfg=self.cfg,
            )

        expert_kfac = None
        expert_kfac_summary = None
        expert_kfac_timing = None

        should_collect_expert_kfac = (
            str(_cfg_get(self.cfg, "agg.expert.method", "")).lower().strip()
            == "fisher_kfac_expert"
            or bool(_cfg_get(self.cfg, "kfac.collect", False))
        )

        if should_collect_expert_kfac:
            expert_kfac_timing = str(
                _cfg_get(
                    self.cfg,
                    "kfac.fisher_timing",
                    _cfg_get(self.cfg, "kfac.collect_timing", "after_train"),
                )
            ).lower().strip()

            if expert_kfac_timing != "after_train":
                raise ValueError(
                    "当前 K-FAC 采集只支持 kfac.fisher_timing=after_train。"
                    f"当前值：{expert_kfac_timing}。"
                    "请不要在本地训练过程中混合统计 K-FAC。"
                )

            expert_kfac = collect_expert_kfac(
                model=local_model,
                train_loader=self.train_loader,
                criterion=criterion,
                device=self.device,
                cfg=self.cfg,
            )
            expert_kfac_summary = summarize_expert_kfac(expert_kfac)

        local_state_cpu = state_dict_to(
            local_model.state_dict(),
            device="cpu",
        )

        model_delta = subtract_state_dict(
            local_state=local_state_cpu,
            global_state=global_state_cpu,
            strict=True,
        )

        check_finite_state_dict(model_delta)

        update = ClientUpdate(
            client_id=self.client_id,
            round_id=int(round_id),
            num_samples=self.num_samples,
            model_delta=model_delta,
            metrics=stats.to_metrics(),
            extra={
                "optimizer": get_optimizer_type(self.cfg),
                "local_epochs": int(local_epochs),
                "grad_clip": float(grad_clip) if grad_clip is not None else None,
                "expert_usage": expert_usage,
                "expert_kfac": expert_kfac,
                "expert_kfac_summary": expert_kfac_summary,
                "expert_kfac_timing": expert_kfac_timing,
            },
        )

        del local_model
        del optimizer
        del criterion

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return update


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    grad_clip: Optional[float] = None,
) -> ClientTrainStats:
    """
    训练一个客户端本地模型。

    这里的训练 loss 只有 CrossEntropyLoss。

    明确不加入：
    1. aux_loss
    2. router balance
    3. entropy regularization
    4. expert diversity
    5. router consistency
    6. proximal loss
    """
    if local_epochs <= 0:
        raise ValueError(f"local_epochs 必须大于 0，当前值：{local_epochs}")

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    for _ in range(local_epochs):
        for batch in train_loader:
            images, targets = unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            logits = extract_logits(outputs)
            loss = criterion(logits, targets)

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_clip),
                )

            optimizer.step()

            batch_size = int(targets.size(0))

            total_loss += float(loss.item()) * batch_size
            total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total_samples += batch_size
            total_batches += 1

    if total_samples <= 0:
        raise ValueError("客户端本地训练没有处理任何样本。")

    avg_loss = total_loss / total_samples
    train_acc = 100.0 * total_correct / total_samples

    return ClientTrainStats(
        avg_loss=avg_loss,
        train_acc=train_acc,
        num_samples=total_samples,
        num_batches=total_batches,
    )


@torch.inference_mode()
def collect_expert_usage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    cfg: Any,
) -> Dict[str, Any]:
    """
    统计一个客户端本地模型的 expert 使用情况。

    统计时机：
        本地训练结束后。

    统计数据：
        当前客户端自己的 train_loader。

    输出字段：
        num_samples:
            实际用于统计的样本数。
        num_batches:
            实际用于统计的 batch 数。
        num_experts:
            expert 总数。
        topk:
            每个样本激活的 expert 数。
        total_activations:
            expert 总激活次数。
            通常约等于 num_samples * topk。
        expert_counts:
            每个 expert 被选中的次数。
        expert_fraction:
            每个 expert 被选中的比例。
        active_experts:
            至少被选中过一次的 expert 数。
        dead_experts:
            本次统计中完全没有被选中的 expert id。
        supported:
            当前模型是否支持 return_router_info=True。

    注意：
        这个函数只做前向统计，不更新模型参数。
    """
    max_batches = int(_cfg_get(cfg, "logging.expert_usage_max_batches", 0))
    num_experts = int(_cfg_get(cfg, "num_experts", 0))
    topk = int(_cfg_get(cfg, "topk", 1))

    if num_experts <= 0:
        return {
            "supported": False,
            "reason": "num_experts <= 0",
        }

    old_training = bool(model.training)
    model.eval()

    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.float64,
        device="cpu",
    )

    total_samples = 0
    total_batches = 0
    supported = True
    unsupported_reason = ""

    try:
        for batch_index, batch in enumerate(train_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            images, targets = unpack_batch(batch)
            images = images.to(device, non_blocking=True)

            try:
                outputs = model(
                    images,
                    return_router_info=True,
                )
            except TypeError as exc:
                supported = False
                unsupported_reason = (
                    "model does not support return_router_info=True: "
                    f"{exc}"
                )
                break

            router_info = extract_router_info(outputs)

            if router_info is None:
                supported = False
                unsupported_reason = "model output does not contain router_info"
                break

            batch_expert_counts = router_info.get("expert_counts", None)

            if batch_expert_counts is None:
                supported = False
                unsupported_reason = "router_info does not contain expert_counts"
                break

            batch_expert_counts = batch_expert_counts.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

            if batch_expert_counts.numel() != num_experts:
                supported = False
                unsupported_reason = (
                    "expert_counts length mismatch: "
                    f"expected={num_experts}, actual={batch_expert_counts.numel()}"
                )
                break

            expert_counts += batch_expert_counts.reshape(-1)

            total_samples += int(images.size(0))
            total_batches += 1

    finally:
        if old_training:
            model.train()
        else:
            model.eval()

    if not supported:
        return {
            "supported": False,
            "reason": unsupported_reason,
        }

    total_activations = float(expert_counts.sum().item())

    if total_activations > 0:
        expert_fraction_tensor = expert_counts / total_activations
    else:
        expert_fraction_tensor = torch.zeros_like(expert_counts)

    expert_counts_dict = {
        int(expert_id): int(expert_counts[expert_id].item())
        for expert_id in range(num_experts)
    }

    expert_fraction_dict = {
        int(expert_id): float(expert_fraction_tensor[expert_id].item())
        for expert_id in range(num_experts)
    }

    dead_experts = [
        int(expert_id)
        for expert_id, count in expert_counts_dict.items()
        if count <= 0
    ]

    active_experts = int(num_experts - len(dead_experts))

    return {
        "supported": True,
        "num_samples": int(total_samples),
        "num_batches": int(total_batches),
        "max_batches": int(max_batches),
        "num_experts": int(num_experts),
        "topk": int(topk),
        "total_activations": int(total_activations),
        "expert_counts": expert_counts_dict,
        "expert_fraction": expert_fraction_dict,
        "active_experts": int(active_experts),
        "dead_experts": dead_experts,
    }


def extract_router_info(outputs: Any) -> Optional[Mapping[str, Any]]:
    """
    从模型输出中提取 router_info。

    兼容几种常见输出：
    1. dataclass / object:
        outputs.router_info
    2. dict:
        outputs["router_info"]
    3. tuple/list:
        outputs[1] 是 router_info

    当前 resnet_sparse_moe_head 在 return_router_info=True 时，
    返回对象里包含 .router_info。
    """
    if hasattr(outputs, "router_info"):
        router_info = outputs.router_info
        if isinstance(router_info, Mapping):
            return router_info
        return None

    if isinstance(outputs, Mapping):
        router_info = outputs.get("router_info", None)
        if isinstance(router_info, Mapping):
            return router_info
        return None

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        router_info = outputs[1]
        if isinstance(router_info, Mapping):
            return router_info
        return None

    return None


def build_criterion(cfg: Any) -> nn.Module:
    """
    构建本地训练 loss 函数。

    第一版只使用 CrossEntropyLoss。
    """
    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))

    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )


def build_optimizer(
    model: nn.Module,
    cfg: Any,
) -> optim.Optimizer:
    """
    根据 cfg.optimizer 构建优化器。

    当前支持：
    sgd
    adam
    adamw
    """
    optimizer_type = get_optimizer_type(cfg)
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})

    lr = float(_cfg_get(optimizer_cfg, "lr", 0.01))
    weight_decay = float(_cfg_get(optimizer_cfg, "weight_decay", 0.0))

    params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    if len(params) == 0:
        raise ValueError("模型没有可训练参数。")

    if optimizer_type == "sgd":
        momentum = float(_cfg_get(optimizer_cfg, "momentum", 0.9))
        nesterov = bool(_cfg_get(optimizer_cfg, "nesterov", False))

        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if optimizer_type == "adam":
        betas = _cfg_get(optimizer_cfg, "betas", (0.9, 0.999))
        eps = float(_cfg_get(optimizer_cfg, "eps", 1e-8))

        return optim.Adam(
            params,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )

    if optimizer_type == "adamw":
        betas = _cfg_get(optimizer_cfg, "betas", (0.9, 0.999))
        eps = float(_cfg_get(optimizer_cfg, "eps", 1e-8))

        return optim.AdamW(
            params,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"不支持的优化器类型：{optimizer_type}。"
        "当前支持：sgd, adam, adamw"
    )


def get_optimizer_type(cfg: Any) -> str:
    """
    从配置中读取优化器类型。
    """
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})
    optimizer_type = _cfg_get(optimizer_cfg, "type", "sgd")

    return str(optimizer_type).lower().strip()


def build_clients(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    device: torch.device | str,
) -> List[FLClient]:
    """
    根据客户端 DataLoader 列表创建 FLClient 列表。

    参数：
        cfg: 全局配置。
        client_loaders: 每个客户端对应一个 DataLoader。
        device: 本地训练使用的设备。
    """
    clients: List[FLClient] = []

    for client_id, train_loader in enumerate(client_loaders):
        clients.append(
            FLClient(
                client_id=client_id,
                train_loader=train_loader,
                cfg=cfg,
                device=device,
            )
        )

    return clients


def select_clients(
    clients: Sequence[FLClient],
    frac: float,
    round_id: int,
    seed: int,
) -> List[FLClient]:
    """
    按比例选择本轮参与训练的客户端。

    选择逻辑：
        每一轮使用 seed + round_id 生成随机数。
        这样同一个 seed 下实验可复现。
    """
    if len(clients) == 0:
        raise ValueError("clients 不能为空。")

    if frac <= 0:
        raise ValueError(f"frac 必须大于 0，当前值：{frac}")

    num_clients = len(clients)
    num_selected = max(1, int(num_clients * float(frac)))
    num_selected = min(num_selected, num_clients)

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(round_id))

    perm = torch.randperm(
        num_clients,
        generator=generator,
    ).tolist()

    selected_indices = perm[:num_selected]

    return [
        clients[index]
        for index in selected_indices
    ]


def train_selected_clients(
    clients: Sequence[FLClient],
    global_model: nn.Module,
    round_id: int,
) -> List[ClientUpdate]:
    """
    训练本轮选中的客户端。

    server.py 后面可以直接调用这个函数。
    """
    updates: List[ClientUpdate] = []

    for client in clients:
        update = client.train(
            global_model=global_model,
            round_id=round_id,
        )
        updates.append(update)

    return updates


def _get_grad_clip(cfg: Any) -> Optional[float]:
    """
    读取梯度裁剪配置。

    支持两种写法：

    optimizer:
      grad_clip: 5.0

    或者：

    grad_clip: 5.0

    如果没有配置，则返回 None。
    """
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})

    value = _cfg_get(
        optimizer_cfg,
        "grad_clip",
        None,
    )

    if value is None:
        value = _cfg_get(
            cfg,
            "grad_clip",
            None,
        )

    if value is None:
        return None

    value = float(value)

    if value <= 0:
        return None

    return value


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