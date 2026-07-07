from __future__ import annotations

import copy
import gc
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fl.fisher_diag import (
    collect_expert_diag_fisher,
    summarize_expert_diag_fisher,
)
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

    avg_loss:
        本地训练平均分类损失，即 CrossEntropyLoss。
        保持原有 train_loss 字段的语义不变。
    train_acc:
        本地训练准确率，百分比形式。
    num_samples:
        本地训练实际处理的累计样本数。
        当 local_epochs > 1 时，该值包含所有 local epoch。
    num_batches:
        本地训练实际处理的累计 batch 数。
    avg_aux_loss:
        未乘 load_balance_loss_weight 的平均 Switch 风格负载均衡损失。
    avg_objective_loss:
        实际执行 backward 的平均总损失：
        CrossEntropyLoss + load_balance_loss_weight * aux_loss。
    train_expert_usage:
        本地训练过程中实时累计的 expert 路由统计。
        该统计与本轮 local model delta 的生成过程对齐。
    """

    avg_loss: float
    train_acc: float
    num_samples: int
    num_batches: int
    avg_aux_loss: float = 0.0
    avg_objective_loss: float = 0.0
    train_expert_usage: Optional[Dict[str, Any]] = None

    def to_metrics(self) -> Dict[str, float]:
        """
        转成 ClientUpdate.metrics 使用的普通 dict。
        """

        return {
            # 保持原有字段与含义不变：train_loss 仍然表示分类损失。
            "train_loss": float(self.avg_loss),
            "train_acc": float(self.train_acc),
            "num_batches": float(self.num_batches),
            # 新增负载均衡相关诊断值。
            "train_aux_loss": float(self.avg_aux_loss),
            "train_objective_loss": float(self.avg_objective_loss),
        }


class FLClient:
    """
    联邦学习客户端。

    职责：
    1. 接收 server 下发的 global_model
    2. 在自己的 train_loader 上本地训练
    3. 在自己的 evidence_loader 上统计 Fisher / K-FAC evidence
    4. 计算 local_model 相对 global_model 的参数变化量
    5. 返回 ClientUpdate

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
        evidence_loader: Optional[DataLoader] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.train_loader = train_loader

        # Fisher / K-FAC evidence 专用 loader。
        # 正常情况下由 data/loaders.py 基于 train_evidence_dataset 构建，
        # 其 transform 已经关闭 RandomCrop / RandomHorizontalFlip。
        #
        # 如果旧代码路径没有传入 evidence_loader，则回退到 train_loader，
        # 这样可以兼容旧配置，但推荐新流程始终显式传入 evidence_loader。
        self.evidence_loader = (
            evidence_loader
            if evidence_loader is not None
            else train_loader
        )

        self.cfg = cfg
        self.device = torch.device(device)

        if len(self.train_loader.dataset) <= 0:
            raise ValueError(f"客户端 {self.client_id} 的数据集为空。")

        if len(self.evidence_loader.dataset) <= 0:
            raise ValueError(
                f"客户端 {self.client_id} 的 evidence 数据集为空。"
            )

    @property
    def num_samples(self) -> int:
        """
        当前客户端本地训练样本数。

        注意：
        聚合时的样本数仍然按训练集 train_loader 统计。
        evidence_loader 只是 Fisher / K-FAC 统计用，不改变客户端样本权重定义。
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
        global_model:
            server 当前轮下发的全局模型。
        round_id:
            当前联邦训练轮数。

        返回：
        ClientUpdate:
            包含 model_delta、num_samples、metrics、extra 等信息。
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

        # 从 base.yaml 根级字段读取统一的 Switch 风格负载均衡系数。
        # 设为 0.0 时完全保持原来的纯 CrossEntropyLoss 训练流程。
        load_balance_loss_weight = float(
            _cfg_get(
                self.cfg,
                "load_balance_loss_weight",
                0.0,
            )
        )

        if not math.isfinite(load_balance_loss_weight):
            raise ValueError(
                "load_balance_loss_weight 必须是有限数值，"
                f"当前值：{load_balance_loss_weight}"
            )

        if load_balance_loss_weight < 0.0:
            raise ValueError(
                "load_balance_loss_weight 不能小于 0，"
                f"当前值：{load_balance_loss_weight}"
            )

        expert_agg_method = str(
            _cfg_get(self.cfg, "agg.expert.method", "")
        ).lower().strip()

        # diagonal Fisher shrinkage 的 expert delta 将改为使用本地训练阶段的
        # 真实路由 usage 聚合，因此必须在产生 local delta 的同一训练 forward 中
        # 实时累计 expert_counts，不能再依赖训练结束后的额外前向统计。
        collect_train_expert_usage = (
            expert_agg_method == "fisher_diag_shrinkage_expert"
        )

        stats = train_local_model(
            model=local_model,
            train_loader=self.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            local_epochs=local_epochs,
            grad_clip=grad_clip,
            load_balance_loss_weight=load_balance_loss_weight,
            collect_train_expert_usage=collect_train_expert_usage,
            num_experts=int(_cfg_get(self.cfg, "num_experts", 0)),
            topk=int(_cfg_get(self.cfg, "topk", 1)),
        )

        # ------------------------------------------------------------
        # 可选：采集当前客户端本地模型的 expert usage。
        #
        # diagonal Fisher shrinkage：
        #   直接使用本地训练循环中实时累计的 train_expert_usage，
        #   不再在训练结束后额外前向一遍 train_loader。
        #
        # 其他聚合方法：
        #   如果 logging.collect_expert_usage=true，仍保留原来的训练后
        #   额外前向统计，仅作为日志诊断，保证旧实验路径不受影响。
        # ------------------------------------------------------------
        expert_usage = stats.train_expert_usage

        if (
            expert_usage is None
            and bool(
                _cfg_get(
                    self.cfg,
                    "logging.collect_expert_usage",
                    False,
                )
            )
        ):
            expert_usage = collect_expert_usage(
                model=local_model,
                train_loader=self.train_loader,
                device=self.device,
                cfg=self.cfg,
            )

        # ------------------------------------------------------------
        # 可选：采集严格逐样本的 expert diagonal empirical Fisher。
        #
        # 只在以下任一条件满足时执行：
        # 1. expert 聚合方法为 fisher_diag_shrinkage_expert；
        # 2. 配置中显式设置 fisher_diag.collect=true。
        #
        # 采集时使用 evidence_loader，而不是带随机数据增强的 train_loader。
        # collect_expert_diag_fisher 内部会：
        # - 在本地训练结束后的 local_model 上采集；
        # - 使用 Linear hook 逐样本累计梯度平方；
        # - 同时统计 Fisher evidence pass 的 expert routed count；
        # - 只返回 expert 参数的 diagonal Fisher。
        #
        # 新逻辑中：
        # - train_expert_usage 只负责后续 expert delta 聚合；
        # - Fisher payload 中的 expert_routed_samples 只负责 Fisher 聚合；
        # - 两类 count 分开保存，不相加、不混用。
        #
        # 这里只负责采集和上传，不在客户端执行任何 Fisher 聚合或收缩。
        # ------------------------------------------------------------
        expert_fisher_diag = None
        expert_fisher_diag_summary = None
        expert_fisher_diag_timing = None

        should_collect_expert_fisher_diag = (
            expert_agg_method == "fisher_diag_shrinkage_expert"
            or bool(_cfg_get(self.cfg, "fisher_diag.collect", False))
        )

        if should_collect_expert_fisher_diag:
            expert_fisher_diag_timing = str(
                _cfg_get(
                    self.cfg,
                    "fisher_diag.fisher_timing",
                    _cfg_get(
                        self.cfg,
                        "fisher_diag.collect_timing",
                        "after_train",
                    ),
                )
            ).lower().strip()

            if expert_fisher_diag_timing != "after_train":
                raise ValueError(
                    "当前 diagonal Fisher 采集只支持 "
                    "fisher_diag.fisher_timing=after_train。"
                    f"当前值：{expert_fisher_diag_timing}。"
                    "请不要在本地训练过程中混合统计 diagonal Fisher。"
                )

            expert_fisher_diag = collect_expert_diag_fisher(
                model=local_model,
                train_loader=self.evidence_loader,
                criterion=criterion,
                device=self.device,
                cfg=self.cfg,
            )

            expert_fisher_diag_summary = summarize_expert_diag_fisher(
                expert_fisher_diag
            )

        expert_kfac = None
        expert_kfac_summary = None
        expert_kfac_timing = None

        should_collect_expert_kfac = (
            expert_agg_method == "fisher_kfac_expert"
            or bool(_cfg_get(self.cfg, "kfac.collect", False))
        )

        if should_collect_expert_kfac:
            expert_kfac_timing = str(
                _cfg_get(
                    self.cfg,
                    "kfac.fisher_timing",
                    _cfg_get(
                        self.cfg,
                        "kfac.collect_timing",
                        "after_train",
                    ),
                )
            ).lower().strip()

            if expert_kfac_timing != "after_train":
                raise ValueError(
                    "当前 K-FAC 采集只支持 kfac.fisher_timing=after_train。"
                    f"当前值：{expert_kfac_timing}。"
                    "请不要在本地训练过程中混合统计 K-FAC。"
                )

            # ------------------------------------------------------------
            # Fisher / K-FAC evidence 统计使用 evidence_loader。
            #
            # 这一步是此前修改的关键：
            # 原代码这里使用 self.train_loader，
            # 如果 train_dataset 开启了 RandomCrop / RandomHorizontalFlip，
            # 那么统计 Fisher 时也会触发随机数据增强。
            #
            # 现在改为 self.evidence_loader。
            # 新的 evidence_loader 来自 train_evidence_dataset，
            # transform 强制关闭随机数据增强，只保留 ToTensor + Normalize。
            #
            # 注意：
            # collect_expert_kfac 内部仍然会根据 kfac.model_mode 切换 eval/train。
            # model.eval() 只能关闭 Dropout / BN 训练态行为，
            # 不能关闭 torchvision transform 的随机增强。
            # 所以必须在 loader / dataset 层面单独处理。
            # ------------------------------------------------------------
            expert_kfac = collect_expert_kfac(
                model=local_model,
                train_loader=self.evidence_loader,
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
                "grad_clip": (
                    float(grad_clip)
                    if grad_clip is not None
                    else None
                ),
                "load_balance_loss_weight": float(
                    load_balance_loss_weight
                ),
                # 新字段：后续 diagonal Fisher shrinkage 聚合器应读取这里，
                # 用训练阶段 usage 计算 expert delta 的 client-expert 权重。
                "train_expert_usage": stats.train_expert_usage,
                # 保留原字段给 server 日志使用。
                # diagonal Fisher shrinkage 下它与 train_expert_usage 指向
                # 同一份训练期统计；其他方法仍可能是训练后的日志统计。
                "expert_usage": expert_usage,
                "expert_fisher_diag": expert_fisher_diag,
                "expert_fisher_diag_summary": expert_fisher_diag_summary,
                "expert_fisher_diag_timing": expert_fisher_diag_timing,
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
    load_balance_loss_weight: float = 0.0,
    collect_train_expert_usage: bool = False,
    num_experts: int = 0,
    topk: int = 1,
) -> ClientTrainStats:
    """
    训练一个客户端本地模型。

    当 load_balance_loss_weight > 0 时，本地训练目标为：

        objective_loss
            = task_loss
            + load_balance_loss_weight * aux_loss

    其中：
    - task_loss 是原有的 CrossEntropyLoss；
    - aux_loss 是模型返回的 Switch Transformer 风格负载均衡损失；
    - load_balance_loss_weight=0 时，完全退化为原来的纯分类训练流程。

    当 collect_train_expert_usage=True 时：
    - 每个训练 batch 使用同一次 forward 返回的 expert_counts；
    - 在整个本地训练轨迹中累计每个 expert 的真实路由次数；
    - 该统计用于解释并聚合本轮 local expert delta。

    注意：
    负载均衡损失只用于客户端正常本地训练。
    本地训练结束后的 Fisher / K-FAC evidence 统计仍然只使用 criterion，
    不会把 aux_loss 混入 Fisher 定义。
    """

    if local_epochs <= 0:
        raise ValueError(f"local_epochs 必须大于 0，当前值：{local_epochs}")

    if not math.isfinite(load_balance_loss_weight):
        raise ValueError(
            "load_balance_loss_weight 必须是有限数值，"
            f"当前值：{load_balance_loss_weight}"
        )

    if load_balance_loss_weight < 0.0:
        raise ValueError(
            "load_balance_loss_weight 不能小于 0，"
            f"当前值：{load_balance_loss_weight}"
        )

    if collect_train_expert_usage:
        if num_experts <= 0:
            raise ValueError(
                "collect_train_expert_usage=True 时，"
                f"num_experts 必须大于 0，当前值：{num_experts}。"
            )

        if topk <= 0:
            raise ValueError(
                "collect_train_expert_usage=True 时，"
                f"topk 必须大于 0，当前值：{topk}。"
            )

        if topk > num_experts:
            raise ValueError(
                "topk 不能大于 num_experts，"
                f"当前 topk={topk}, num_experts={num_experts}。"
            )

    use_load_balance = load_balance_loss_weight > 0.0

    # 即使关闭负载均衡，只要要统计训练阶段 usage，也必须请求 router_info。
    need_router_info = use_load_balance or collect_train_expert_usage

    model.train()

    # 分类损失、辅助损失和实际反向传播目标分别统计，
    # 避免改变原有 train_loss 的含义。
    total_task_loss = 0.0
    total_aux_loss = 0.0
    total_objective_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    train_expert_counts: Optional[torch.Tensor]
    if collect_train_expert_usage:
        # expert 数量通常很小，每个 batch 将 count 拷到 CPU 后累计，
        # 避免长期占用额外 GPU buffer。
        train_expert_counts = torch.zeros(
            num_experts,
            dtype=torch.float64,
            device="cpu",
        )
    else:
        train_expert_counts = None

    for _ in range(local_epochs):
        for batch in train_loader:
            images, targets = unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if need_router_info:
                # 启用负载均衡或收集训练 usage 时必须请求 router_info。
                try:
                    outputs = model(
                        images,
                        return_router_info=True,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "当前本地训练需要 router_info，但模型不支持 "
                        "model(images, return_router_info=True)。"
                    ) from exc
            else:
                # 不需要 router_info 时保持原来的前向路径。
                outputs = model(images)

            logits = extract_logits(outputs)
            task_loss = criterion(logits, targets)

            router_info = None
            if need_router_info:
                router_info = extract_router_info(outputs)

                if router_info is None:
                    raise RuntimeError(
                        "当前本地训练需要 router_info，"
                        "但模型输出中没有 router_info。"
                    )

            if collect_train_expert_usage:
                assert train_expert_counts is not None
                assert router_info is not None

                batch_expert_counts = router_info.get(
                    "expert_counts",
                    None,
                )

                if batch_expert_counts is None:
                    raise RuntimeError(
                        "收集训练 expert usage 时，"
                        "router_info 中没有 expert_counts。"
                    )

                if not torch.is_tensor(batch_expert_counts):
                    raise TypeError(
                        "router_info['expert_counts'] 必须是 Tensor，"
                        f"当前类型：{type(batch_expert_counts).__name__}"
                    )

                batch_expert_counts = (
                    batch_expert_counts
                    .detach()
                    .reshape(-1)
                    .to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )

                if batch_expert_counts.numel() != num_experts:
                    raise ValueError(
                        "expert_counts 长度与 num_experts 不一致："
                        f"expected={num_experts}, "
                        f"actual={batch_expert_counts.numel()}。"
                    )

                if not torch.isfinite(batch_expert_counts).all():
                    raise FloatingPointError(
                        "训练阶段 expert_counts 出现 NaN 或 Inf。"
                    )

                if torch.any(batch_expert_counts < 0):
                    raise ValueError(
                        "训练阶段 expert_counts 不能包含负数。"
                    )

                train_expert_counts += batch_expert_counts

            if use_load_balance:
                assert router_info is not None

                aux_loss = router_info.get("aux_loss", None)

                if aux_loss is None:
                    raise RuntimeError(
                        "已启用负载均衡，但 router_info 中没有 aux_loss。"
                    )

                if not torch.is_tensor(aux_loss):
                    raise TypeError(
                        "router_info['aux_loss'] 必须是 Tensor，"
                        f"当前类型：{type(aux_loss).__name__}"
                    )

                if aux_loss.numel() != 1:
                    raise ValueError(
                        "router_info['aux_loss'] 必须是标量 Tensor，"
                        f"当前 shape={tuple(aux_loss.shape)}"
                    )

                if not torch.isfinite(aux_loss).all():
                    raise FloatingPointError(
                        "router aux_loss 出现 NaN 或 Inf。"
                    )

                if not aux_loss.requires_grad:
                    raise RuntimeError(
                        "router aux_loss 不包含梯度。请检查模型中是否对 "
                        "router_probs 或 aux_loss 调用了 detach()。"
                    )

                objective_loss = (
                    task_loss
                    + load_balance_loss_weight * aux_loss
                )
            else:
                # 使用同设备、同 dtype 的零标量，便于统一统计。
                aux_loss = task_loss.new_zeros(())
                objective_loss = task_loss

            if not torch.isfinite(objective_loss).all():
                raise FloatingPointError(
                    "客户端本地训练 objective_loss 出现 NaN 或 Inf。"
                )

            objective_loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_clip),
                )

            optimizer.step()

            batch_size = int(targets.size(0))

            total_task_loss += (
                float(task_loss.detach().item()) * batch_size
            )
            total_aux_loss += (
                float(aux_loss.detach().item()) * batch_size
            )
            total_objective_loss += (
                float(objective_loss.detach().item()) * batch_size
            )
            total_correct += int(
                logits.argmax(dim=1).eq(targets).sum().item()
            )
            total_samples += batch_size
            total_batches += 1

    if total_samples <= 0:
        raise ValueError("客户端本地训练没有处理任何样本。")

    avg_task_loss = total_task_loss / total_samples
    avg_aux_loss = total_aux_loss / total_samples
    avg_objective_loss = total_objective_loss / total_samples
    train_acc = 100.0 * total_correct / total_samples

    train_expert_usage = None
    if train_expert_counts is not None:
        train_expert_usage = _build_train_expert_usage(
            expert_counts=train_expert_counts,
            num_processed_samples=total_samples,
            num_batches=total_batches,
            num_experts=num_experts,
            topk=topk,
        )

    return ClientTrainStats(
        avg_loss=avg_task_loss,
        train_acc=train_acc,
        num_samples=total_samples,
        num_batches=total_batches,
        avg_aux_loss=avg_aux_loss,
        avg_objective_loss=avg_objective_loss,
        train_expert_usage=train_expert_usage,
    )


def _build_train_expert_usage(
    expert_counts: torch.Tensor,
    num_processed_samples: int,
    num_batches: int,
    num_experts: int,
    topk: int,
) -> Dict[str, Any]:
    """
    将本地训练期间累计的 expert count 转成上传用统计结构。

    两种比例必须区分：

    expert_usage[e]
        = expert_counts[e] / num_processed_samples
        与 diagonal Fisher payload 中 usage 的定义一致。
        top-k 路由下所有 expert_usage 的和应等于 topk。
        后续算法使用该字段构造 expert delta 权重。

    expert_fraction[e]
        = expert_counts[e] / total_activations
        所有 expert_fraction 的和为 1，仅用于日志展示。
    """

    if num_processed_samples <= 0:
        raise ValueError(
            "构建 train_expert_usage 时 num_processed_samples 必须大于 0。"
        )

    if num_batches <= 0:
        raise ValueError(
            "构建 train_expert_usage 时 num_batches 必须大于 0。"
        )

    counts = expert_counts.detach().reshape(-1).to(
        device="cpu",
        dtype=torch.float64,
    )

    if counts.numel() != num_experts:
        raise ValueError(
            "训练 expert count 数量与 num_experts 不一致："
            f"expected={num_experts}, actual={counts.numel()}。"
        )

    if not torch.isfinite(counts).all():
        raise FloatingPointError(
            "训练阶段累计 expert_counts 出现 NaN 或 Inf。"
        )

    if torch.any(counts < 0):
        raise ValueError(
            "训练阶段累计 expert_counts 不能包含负数。"
        )

    total_activations = int(counts.sum().item())
    expected_activations = int(num_processed_samples) * int(topk)

    # 当前 SparseMoEHead 的每个样本严格选择 topk 个不同 expert，
    # 因此训练路径中的累计激活数应与 processed_samples * topk 完全一致。
    if total_activations != expected_activations:
        raise RuntimeError(
            "训练阶段 expert 路由总次数不一致："
            f"actual={total_activations}, "
            f"expected={expected_activations} "
            f"(num_processed_samples={num_processed_samples}, topk={topk})。"
        )

    usage_tensor = counts / float(num_processed_samples)

    if total_activations > 0:
        fraction_tensor = counts / float(total_activations)
    else:
        fraction_tensor = torch.zeros_like(counts)

    expert_counts_dict = {
        int(expert_id): int(counts[expert_id].item())
        for expert_id in range(num_experts)
    }

    expert_usage_dict = {
        int(expert_id): float(usage_tensor[expert_id].item())
        for expert_id in range(num_experts)
    }

    expert_fraction_dict = {
        int(expert_id): float(fraction_tensor[expert_id].item())
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
        "source": "local_training",
        # 保留 num_samples 字段，兼容旧日志或下游代码。
        "num_samples": int(num_processed_samples),
        "num_processed_samples": int(num_processed_samples),
        "num_batches": int(num_batches),
        "num_experts": int(num_experts),
        "topk": int(topk),
        "total_activations": int(total_activations),
        "expert_counts": expert_counts_dict,
        # 算法字段：sum(expert_usage.values()) == topk。
        "expert_usage": expert_usage_dict,
        # 日志字段：sum(expert_fraction.values()) == 1。
        "expert_fraction": expert_fraction_dict,
        "active_experts": int(active_experts),
        "dead_experts": dead_experts,
    }


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
    新的 diagonal Fisher shrinkage 路径不再依赖这里的训练后统计；
    该函数仅为其他方法和旧日志路径保留。
    """

    max_batches = int(
        _cfg_get(cfg, "logging.expert_usage_max_batches", 0)
    )
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
                unsupported_reason = (
                    "model output does not contain router_info"
                )
                break

            batch_expert_counts = router_info.get(
                "expert_counts",
                None,
            )

            if batch_expert_counts is None:
                supported = False
                unsupported_reason = (
                    "router_info does not contain expert_counts"
                )
                break

            batch_expert_counts = batch_expert_counts.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

            if batch_expert_counts.numel() != num_experts:
                supported = False
                unsupported_reason = (
                    "expert_counts length mismatch: "
                    f"expected={num_experts}, "
                    f"actual={batch_expert_counts.numel()}"
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
    1. dataclass / object: outputs.router_info
    2. dict: outputs["router_info"]
    3. tuple/list: outputs[1] 是 router_info

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
    weight_decay = float(
        _cfg_get(optimizer_cfg, "weight_decay", 0.0)
    )

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
    client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
) -> List[FLClient]:
    """
    根据客户端 DataLoader 列表创建 FLClient 列表。

    参数：
    cfg:
        全局配置。
    client_loaders:
        每个客户端对应一个训练 DataLoader。
    device:
        本地训练使用的设备。
    client_evidence_loaders:
        每个客户端对应一个 Fisher / K-FAC evidence DataLoader。
        如果为 None，则每个客户端回退使用自己的 train_loader。
    """

    if (
        client_evidence_loaders is not None
        and len(client_evidence_loaders) != len(client_loaders)
    ):
        raise ValueError(
            "client_evidence_loaders 数量必须和 client_loaders 一致。"
            f"当前 client_loaders={len(client_loaders)}, "
            f"client_evidence_loaders={len(client_evidence_loaders)}。"
        )

    clients: List[FLClient] = []

    for client_id, train_loader in enumerate(client_loaders):
        if client_evidence_loaders is None:
            evidence_loader = None
        else:
            evidence_loader = client_evidence_loaders[client_id]

        clients.append(
            FLClient(
                client_id=client_id,
                train_loader=train_loader,
                evidence_loader=evidence_loader,
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

    dict 或 ConfigNode：
        cfg.get(key, default)

    普通对象：
        getattr(cfg, key, default)
    """

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)
