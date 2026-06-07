from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResNetSparseMoEHeadOutput:
    """
    ResNetSparseMoEHead 的可选输出结构。

    默认训练时不需要这个结构，直接返回 logits 即可。
    当需要分析 router / expert usage 时，可以设置 return_router_info=True。
    """

    logits: torch.Tensor
    router_info: Dict[str, Any]


class BasicBlock(nn.Module):
    """
    CIFAR 风格 ResNet BasicBlock。

    结构：
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Residual -> ReLU

    这个 block 比 torchvision 默认 ResNet 更适合 CIFAR 小图，
    因为前面的 stem 不会过早大幅下采样。
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 当通道数或空间尺寸变化时，用 1x1 Conv 对齐残差分支。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """
    ResNet 图像特征提取器。

    输入：
        x: [B, C, H, W]

    输出：
        feat: [B, 512]

    说明：
    - 对 CIFAR10 / CIFAR100 这类 32x32 小图，stem 使用 stride=1。
    - 对 TinyImageNet 这类更大图，stem 使用 stride=2。
    - 最后通过 AdaptiveAvgPool2d(1) 得到单个全局特征向量。
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        stem_stride = 1 if int(image_size) <= 32 else 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        """
        每个 stage 使用两个 BasicBlock。
        第一个 block 负责必要的下采样，第二个 block 保持尺寸。
        """
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        return x


class ExpertFFN(nn.Module):
    """
    单个 expert。

    这里 expert 内部直接输出分类 logits 的一部分：
        feature -> Linear -> ReLU -> Linear -> num_classes

    因此这个模型里“分类头”是在 expert 内部的。
    聚合 expert 参数时，会同时聚合每个 expert 的 fc1 和 fc2。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        return x


class TopKGating(nn.Module):
    """
    标准 Top-K 路由器。

    输入：
        x: [B, in_dim]

    输出：
        weights: [B, num_experts]
            只有 top-k expert 位置非零。
            默认保持原始 softmax 概率，不重新归一化。
        topk_indices: [B, topk]
            每个样本选中的 expert id。
        router_probs: [B, num_experts]
            softmax 后的完整路由概率。
        router_logits: [B, num_experts]
            router 原始 logits。

    注意：
    - 不加乘法噪声。
    - 不加负载均衡 loss。
    - 不加 router entropy / diversity / consistency 正则。
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        topk: int,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts 必须大于 0，当前值：{num_experts}")
        if topk <= 0:
            raise ValueError(f"topk 必须大于 0，当前值：{topk}")
        if topk > num_experts:
            raise ValueError(
                f"topk 不能大于 num_experts，当前 topk={topk}, "
                f"num_experts={num_experts}"
            )

        self.in_dim = int(in_dim)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.renormalize_topk_probs = bool(renormalize_topk_probs)

        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"TopKGating 期望输入为 [B, D]，当前 shape={tuple(x.shape)}")

        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits.float(), dim=-1)

        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.topk,
            dim=-1,
        )

        if self.renormalize_topk_probs:
            topk_probs = topk_probs / topk_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        weights = torch.zeros_like(router_probs)
        weights.scatter_(dim=1, index=topk_indices, src=topk_probs)
        weights = weights.to(dtype=x.dtype)

        return weights, topk_indices, router_probs, router_logits


class SparseMoEHead(nn.Module):
    """
    稀疏 MoE 分类头。

    输入：
        x: [B, feat_dim]

    输出：
        logits: [B, num_classes]

    计算方式：
    1. router 为每个样本选择 top-k 个 expert。
    2. 遍历 expert。
    3. 只把被该 expert 选中的样本送入该 expert。
    4. 用 router 权重加权 expert 输出。
    5. 所有被选中的 expert 输出累加成最终 logits。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_experts: int,
        topk: int,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)

        self.gating = TopKGating(
            in_dim=in_dim,
            num_experts=num_experts,
            topk=topk,
            renormalize_topk_probs=renormalize_topk_probs,
        )

        # 这个命名很重要：
        # 参数名会包含 moe_head.experts.<expert_id>....
        # 这样现有 param_groups / K-FAC 逻辑更容易识别 expert 参数。
        self.experts = nn.ModuleList(
            [
                ExpertFFN(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=num_classes,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError(f"SparseMoEHead 期望输入为 [B, D]，当前 shape={tuple(x.shape)}")

        weights, topk_indices, router_probs, router_logits = self.gating(x)

        batch_size = x.size(0)
        logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=x.device,
            dtype=x.dtype,
        )

        # 真稀疏计算：每个 expert 只处理路由到自己的样本。
        for expert_id, expert in enumerate(self.experts):
            selected_mask = topk_indices == expert_id
            token_mask = selected_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x[token_mask]
            expert_output = expert(expert_input)

            selected_weights = weights[token_mask, expert_id]
            logits[token_mask] = logits[token_mask] + (
                expert_output * selected_weights.unsqueeze(-1)
            )

        if not return_router_info:
            return logits

        # 以下信息只用于诊断，不参与训练 loss。
        expert_one_hot = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).to(dtype=torch.float32)

        expert_counts = expert_one_hot.sum(dim=(0, 1))
        sample_expert_counts = expert_one_hot.sum(dim=1)

        density = expert_counts / max(float(batch_size * self.topk), 1.0)
        density_proxy = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(
            density.to(router_probs.device) * density_proxy
        )

        router_info = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts.to(x.device),
            "sample_expert_counts": sample_expert_counts.to(x.device),
            "selected_experts": topk_indices,
            "topk_probs": torch.gather(router_probs, dim=1, index=topk_indices).to(
                dtype=x.dtype
            ),
            "router_probs": router_probs.to(dtype=x.dtype),
            "router_logits": router_logits.to(dtype=x.dtype),
        }

        return logits, router_info


class ResNetSparseMoEHead(nn.Module):
    """
    ResNet + Sparse MoE Head 分类模型。

    整体结构：
        image
          -> ResNetBackbone
          -> global feature [B, 512]
          -> SparseMoEHead
          -> logits [B, num_classes]

    和当前 resnet_switch_moe 的主要区别：
    - 这个模型没有 Transformer block。
    - 这个模型没有单独 classifier。
    - 每个 expert 自己输出 num_classes 维 logits。
    - router / backbone 是 non-expert 参数。
    - experts 是 expert 参数。
    """

    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        topk: int = 2,
        in_channels: int = 3,
        image_size: int = 32,
        moe_hidden_dim: int = 512,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，当前值：{num_classes}")

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.moe_hidden_dim = int(moe_hidden_dim)

        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            image_size=image_size,
        )

        self.moe_head = SparseMoEHead(
            in_dim=self.backbone.feat_dim,
            hidden_dim=moe_hidden_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            topk=topk,
            renormalize_topk_probs=renormalize_topk_probs,
        )

        self._init_extra_weights()

    def _init_extra_weights(self) -> None:
        """
        初始化 Linear 层。

        Conv / BN 使用 PyTorch 默认初始化也能跑；
        这里额外对 Linear 做 Xavier 初始化，让 router 和 expert 更稳定一点。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | ResNetSparseMoEHeadOutput:
        feat = self.backbone(x)

        if not return_router_info:
            logits = self.moe_head(
                feat,
                return_router_info=False,
            )
            return logits

        logits, router_info = self.moe_head(
            feat,
            return_router_info=True,
        )

        return ResNetSparseMoEHeadOutput(
            logits=logits,
            router_info=router_info,
        )


def build_resnet_sparse_moe_head_from_cfg(cfg: Any) -> ResNetSparseMoEHead:
    """
    根据 cfg 构建 ResNetSparseMoEHead。

    推荐配置示例：

    model: resnet_sparse_moe_head
    num_experts: 4
    topk: 2

    model_cfg:
      in_channels: 3
      image_size: 32
      moe_hidden_dim: 512
      renormalize_topk_probs: false

    说明：
    - num_classes 优先从 cfg.num_classes 读取。
    - in_channels / image_size 会优先从 cfg.input_shape 推断。
    - model_cfg 里的 in_channels / image_size 可以覆盖默认值。
    """

    model_cfg = _cfg_get(cfg, "model_cfg", {})

    input_shape = _cfg_get(cfg, "input_shape", (3, 32, 32))
    if input_shape is None:
        input_shape = (3, 32, 32)

    default_in_channels = int(input_shape[0])
    default_image_size = int(input_shape[1])

    # 兼容两种写法：
    # 1. model_cfg.moe_hidden_dim
    # 2. model_cfg.hidden_dim
    # 如果都没写，默认使用原始 moefedavg.py 里的 512。
    default_moe_hidden_dim = _cfg_get(
        model_cfg,
        "hidden_dim",
        512,
    )

    return ResNetSparseMoEHead(
        num_classes=int(_cfg_get(cfg, "num_classes")),
        num_experts=int(_cfg_get(cfg, "num_experts", 4)),
        topk=int(_cfg_get(cfg, "topk", 2)),
        in_channels=int(
            _cfg_get(
                model_cfg,
                "in_channels",
                default_in_channels,
            )
        ),
        image_size=int(
            _cfg_get(
                model_cfg,
                "image_size",
                default_image_size,
            )
        ),
        moe_hidden_dim=int(
            _cfg_get(
                model_cfg,
                "moe_hidden_dim",
                default_moe_hidden_dim,
            )
        ),
        renormalize_topk_probs=bool(
            _cfg_get(
                model_cfg,
                "renormalize_topk_probs",
                False,
            )
        ),
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