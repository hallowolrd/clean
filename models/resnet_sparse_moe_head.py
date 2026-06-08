from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    CIFAR 适配版 ResNet BasicBlock。

    说明：
    - 结构来自你提供的 moefedavg.py。
    - conv 默认不使用 bias，因为后面接 BatchNorm。
    - 当 stride != 1 或通道数变化时，用 1x1 shortcut 对齐维度。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]

        输出：
            out: [B, out_ch, H', W']
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """
    简化版 ResNet backbone。

    整体流程：
        image
        -> stem
        -> layer1
        -> layer2
        -> layer3
        -> layer4
        -> AdaptiveAvgPool2d(1)
        -> flatten feature

    输出：
        [B, 512]

    注意：
    - 对 CIFAR10 / CIFAR100 这种 32x32 小图，stem_stride=1。
    - 对 TinyImageNet 这种更大图像，stem_stride=2。
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
    ) -> None:
        super().__init__()

        stem_stride = 1 if int(image_size) <= 32 else 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
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
        in_ch: int,
        out_ch: int,
        stride: int,
    ) -> nn.Sequential:
        """
        每个 stage 使用两个 BasicBlock。
        第一个 block 可以负责下采样，第二个 block 保持分辨率。
        """
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]

        输出：
            feat: [B, 512]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ExpertFFN(nn.Module):
    """
    单个专家。

    结构：
        Linear(in_dim -> hidden_dim)
        -> ReLU
        -> Linear(hidden_dim -> out_dim)

    在这个模型里，out_dim = num_classes。
    也就是说，每个 expert 直接输出分类 logits。
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
        """
        输入：
            x: [N, in_dim]

        输出：
            logits: [N, out_dim]
        """
        return self.fc2(F.relu(self.fc1(x)))


class TopKGating(nn.Module):
    """
    标准 Top-K 路由器。

    特点：
    - 不加乘法噪声。
    - 不加负载均衡损失。
    - 不加 entropy / diversity / consistency 正则。
    - router 只负责给每个样本选择 top-k expert。

    返回：
    - weights: [B, num_experts]，非 top-k expert 的权重为 0。
    - topk_idx: [B, topk]，每个样本选中的 expert id。
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        topk: int,
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

        # 和你提供的 moefedavg.py 保持一致：router 不使用 bias。
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
            x: [B, in_dim]

        输出：
            weights: [B, num_experts]
            topk_idx: [B, topk]
        """
        logits = self.gate(x)

        # softmax 用 float32 计算更稳，最后再转回输入 dtype。
        probs = torch.softmax(logits.float(), dim=-1)

        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)

        weights = torch.zeros_like(probs)
        weights.scatter_(dim=1, index=topk_idx, src=topk_vals)
        weights = weights.to(dtype=x.dtype)

        return weights, topk_idx


class MoELayer(nn.Module):
    """
    稀疏 MoE 分类头。

    输入：
        x: [B, in_dim]

    输出：
        out: [B, out_dim]

    参数命名重点：
        self.experts = nn.ModuleList(...)

    这样 state_dict 里专家参数名会包含：
        moe_head.experts.0.fc1.weight
        moe_head.experts.0.fc1.bias
        moe_head.experts.0.fc2.weight
        moe_head.experts.0.fc2.bias

    因此可以直接复用当前 models/param_groups.py 里基于 experts.<id>
    的专家参数识别规则，不需要额外改 param_groups。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_experts: int,
        topk: int,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_experts = int(num_experts)
        self.topk = int(topk)

        self.gating = TopKGating(
            in_dim=in_dim,
            num_experts=num_experts,
            topk=topk,
        )

        self.experts = nn.ModuleList(
            [
                ExpertFFN(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, in_dim]

        输出：
            out: [B, out_dim]

        计算方式：
        - 每个样本只送入自己 top-k 选中的专家。
        - 每个专家只处理分配给自己的样本。
        - expert 输出按 router 权重加权累加。
        """
        weights, topk_idx = self.gating(x)

        batch_size = x.size(0)
        out = torch.zeros(
            batch_size,
            self.out_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for expert_id, expert in enumerate(self.experts):
            # expert_mask: [B, topk]
            # token_mask: [B]
            # token_mask[b] = True 表示第 b 个样本选择了当前 expert。
            expert_mask = topk_idx == expert_id
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x[token_mask]
            expert_output = expert(expert_input)

            selected_weights = weights[token_mask, expert_id]
            out[token_mask] = out[token_mask] + expert_output * selected_weights.unsqueeze(-1)

        return out


class ResNetSparseMoEHead(nn.Module):
    """
    ResNet + Sparse MoE Head 分类模型。

    整体结构：
        image
        -> ResNetBackbone
        -> pooled feature
        -> MoE Head
        -> logits

    和当前 resnet_switch_moe 的区别：
    - 当前模型没有 Transformer token 序列。
    - 每个样本经过 ResNet 后只有一个全局 feature。
    - MoE 位于分类 head，expert 直接输出 num_classes 维 logits。
    - forward 默认只返回 logits，训练代码仍然只用 CrossEntropyLoss。
    """

    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        topk: int = 2,
        in_channels: int = 3,
        image_size: int = 32,
        expert_hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，当前值：{num_classes}")

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.expert_hidden_dim = int(expert_hidden_dim)

        self.backbone = ResNetBackbone(
            in_channels=self.in_channels,
            image_size=self.image_size,
        )

        feat_dim = int(self.backbone.feat_dim)

        self.moe_head = MoELayer(
            in_dim=feat_dim,
            hidden_dim=self.expert_hidden_dim,
            out_dim=self.num_classes,
            num_experts=self.num_experts,
            topk=self.topk,
        )

        self._init_head_weights()

    def _init_head_weights(self) -> None:
        """
        初始化 MoE head 里的 Linear。

        ResNet 里的 Conv / BatchNorm 使用 PyTorch 默认初始化。
        这里单独初始化 Linear，避免新 head 初始化过于随意。
        """
        for module in self.moe_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]

        输出：
            logits: [B, num_classes]

        注意：
        - 默认只返回 logits。
        - 不返回 aux_loss。
        - 不把 router 诊断信息混进训练流程。
        """
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits


def build_resnet_sparse_moe_head_from_cfg(cfg: Any) -> ResNetSparseMoEHead:
    """
    根据 cfg 构建 ResNetSparseMoEHead。

    推荐配置示例：

    model: resnet_sparse_moe_head
    num_classes: 10
    num_experts: 4
    topk: 2
    input_shape: [3, 32, 32]

    model_cfg:
      in_channels: 3
      image_size: 32
      expert_hidden_dim: 512
    """
    model_cfg = _cfg_get(cfg, "model_cfg", {})
    input_shape = _cfg_get(cfg, "input_shape", (3, 32, 32))

    if input_shape is None:
        default_in_channels = 3
        default_image_size = 32
    else:
        default_in_channels = int(input_shape[0])
        default_image_size = int(input_shape[1])

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
        expert_hidden_dim=int(
            _cfg_get(
                model_cfg,
                "expert_hidden_dim",
                512,
            )
        ),
    )


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的配置读取。

    支持：
    - cfg.get(key, default)
    - getattr(cfg, key, default)
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)