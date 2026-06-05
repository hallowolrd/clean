from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


@dataclass(frozen=True)
class SwitchFeedForwardOutput:
    """
    Switch FFN 的输出。

    hidden_states:
        MoE FFN 输出后的 token 表示。

    router_info:
        路由诊断信息。
        注意：这里会计算 aux_loss，但模型不会把它加进训练 loss。
        训练代码默认仍然只使用 CrossEntropyLoss。
    """

    hidden_states: torch.Tensor
    router_info: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ResNetSwitchMoEOutput:
    """
    ResNetSwitchMoE 的输出。

    默认训练时不需要这个结构，直接返回 logits 即可。
    当需要分析 router / expert usage 时，可以设置 return_router_info=True。
    """

    logits: torch.Tensor
    router_info: Dict[str, Any]


class ResNetTokenizer(nn.Module):
    """
    ResNet 图像 tokenizer。

    作用：
        把输入图片变成 token 序列。

    对 CIFAR10 / CIFAR100：
        输入图片形状：
            [B, 3, 32, 32]

        CIFAR-style ResNet18 输出特征图：
            [B, 512, 4, 4]

        经过 1x1 Conv 投影到 hidden_dim：
            [B, hidden_dim, 4, 4]

        flatten 后变成 token：
            [B, 16, hidden_dim]

    这里借鉴 vsmc 的结构：
        1. 使用 ResNet 作为图像 backbone/tokenizer
        2. 对 CIFAR 小图，把 conv1 改成 3x3 stride=1
        3. 去掉 maxpool，避免过早下采样
        4. 最后使用 1x1 Conv 投影到 Transformer hidden_dim
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        in_channels: int = 3,
        hidden_dim: int = 128,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        self.backbone_name = str(backbone_name).lower()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)

        backbone, backbone_out_dim = self._build_resnet_backbone(
            backbone_name=self.backbone_name,
            in_channels=self.in_channels,
            image_size=self.image_size,
        )

        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Conv2d(
                backbone_out_dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.num_image_tokens = self._infer_num_image_tokens()

    @staticmethod
    def _build_resnet_backbone(
        backbone_name: str,
        in_channels: int,
        image_size: int,
    ) -> Tuple[nn.Module, int]:
        """
        构建 ResNet backbone。

        当前支持：
            resnet18
            resnet34
            resnet50

        对 CIFAR 小图：
            conv1 = 3x3, stride=1, padding=1
            maxpool = Identity

        这样 32x32 输入经过 layer2/layer3/layer4 三次下采样后，
        空间分辨率变成 4x4。
        """
        if backbone_name == "resnet18":
            model = resnet18(weights=None)
            out_dim = 512
        elif backbone_name == "resnet34":
            model = resnet34(weights=None)
            out_dim = 512
        elif backbone_name == "resnet50":
            model = resnet50(weights=None)
            out_dim = 2048
        else:
            raise ValueError(
                f"不支持的 backbone_name：{backbone_name}。"
                "当前支持：resnet18, resnet34, resnet50"
            )

        if image_size <= 32:
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.maxpool = nn.Identity()
        else:
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        return backbone, out_dim

    def _infer_num_image_tokens(self) -> int:
        """
        用一次 dummy forward 推断图像 token 数量。

        CIFAR10 默认是：
            [1, 3, 32, 32] -> [1, hidden_dim, 4, 4]
            token 数量 = 4 * 4 = 16
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = torch.zeros(
                1,
                self.in_channels,
                self.image_size,
                self.image_size,
            )
            feat = self.proj(self.backbone(x))
            num_tokens = int(feat.shape[-2] * feat.shape[-1])

        if was_training:
            self.train()

        return num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
            x:
                [B, C, H, W]

        输出：
            tokens:
                [B, num_image_tokens, hidden_dim]
        """
        feat = self.backbone(x)
        feat = self.proj(feat)

        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens


class SwitchFeedForward(nn.Module):
    """
    Top-k Switch Feed-Forward MoE。

    这是 Transformer block 中替代普通 FFN 的 MoE 模块。

    输入：
        x: [B, N, hidden_dim]

    路由流程：
        1. router 得到每个 token 到每个 expert 的 logits
        2. softmax 得到 router_probs
        3. top-k 选择 expert
        4. 被选中的 expert 处理对应 token
        5. 用 router 权重加权 expert 输出

    注意：
        这里支持 topk=1 / topk=2 / topk>2。
        topk=1 时就退化成标准 Switch Transformer 风格的 top-1 routing。

    本文件只实现模型结构。
    不在这里把 aux_loss 加入训练 loss。
    不实现 router balance / entropy / diversity / consistency。
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        topk: int = 1,
        dropout: float = 0.1,
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

        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.renormalize_topk_probs = bool(renormalize_topk_probs)

        self.router = nn.Linear(
            hidden_dim,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | SwitchFeedForwardOutput:
        """
        前向传播。

        输入：
            x:
                [B, N, hidden_dim]

        输出：
            hidden_states:
                [B, N, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = x.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"输入 hidden_dim 不匹配：当前输入={hidden_dim}, "
                f"模型期望={self.hidden_dim}"
            )

        x_flat = x.reshape(batch_size * num_tokens, hidden_dim)

        router_logits = self.router(x_flat)
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

        topk_probs = topk_probs.to(dtype=x.dtype)

        output_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            selected_mask = topk_indices == expert_id
            token_mask = selected_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x_flat[token_mask]
            expert_output = expert(expert_input)

            selected_weights = (
                topk_probs[token_mask]
                * selected_mask[token_mask].to(dtype=topk_probs.dtype)
            ).sum(dim=-1)

            output_flat[token_mask] = (
                output_flat[token_mask]
                + expert_output * selected_weights.unsqueeze(-1)
            )

        output = output_flat.reshape(batch_size, num_tokens, hidden_dim)

        if not return_router_info:
            return output

        router_probs_view = router_probs.reshape(
            batch_size,
            num_tokens,
            self.num_experts,
        )
        router_logits_view = router_logits.reshape(
            batch_size,
            num_tokens,
            self.num_experts,
        )
        topk_indices_view = topk_indices.reshape(
            batch_size,
            num_tokens,
            self.topk,
        )
        topk_probs_view = topk_probs.reshape(
            batch_size,
            num_tokens,
            self.topk,
        )

        expert_one_hot = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).to(dtype=torch.float32)

        expert_counts = expert_one_hot.sum(dim=(0, 1))

        sample_expert_counts = expert_one_hot.sum(dim=1).reshape(
            batch_size,
            num_tokens,
            self.num_experts,
        ).sum(dim=1)

        density = expert_counts / max(float(batch_size * num_tokens * self.topk), 1.0)
        density_proxy = router_probs.mean(dim=0)

        aux_loss = self.num_experts * torch.sum(
            density.to(router_probs.device) * density_proxy
        )

        router_info = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts.to(x.device),
            "sample_expert_counts": sample_expert_counts.to(x.device),
            "selected_experts": topk_indices_view,
            "topk_probs": topk_probs_view,
            "router_probs": router_probs_view.to(dtype=x.dtype),
            "router_logits": router_logits_view.to(dtype=x.dtype),
        }

        return SwitchFeedForwardOutput(
            hidden_states=output,
            router_info=router_info,
        )


class SwitchTransformerBlock(nn.Module):
    """
    Switch Transformer Block。

    结构：
        x
          ↓
        LayerNorm
          ↓
        Multihead Self-Attention
          ↓
        Residual
          ↓
        LayerNorm
          ↓
        SwitchFeedForward MoE
          ↓
        Residual

    这个 block 里没有额外的 router balance / entropy / diversity /
    consistency 正则项。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        topk: int,
        dropout: float = 0.1,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.switch_ffn = SwitchFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            topk=topk,
            dropout=dropout,
            renormalize_topk_probs=renormalize_topk_probs,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        """
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_output)

        ffn_input = self.ffn_norm(x)

        if not return_router_info:
            ffn_output = self.switch_ffn(
                ffn_input,
                return_router_info=False,
            )
            x = x + ffn_output
            return x

        ffn_result = self.switch_ffn(
            ffn_input,
            return_router_info=True,
        )
        x = x + ffn_result.hidden_states

        return x, ffn_result.router_info


class ResNetSwitchMoE(nn.Module):
    """
    ResNet + Switch Transformer MoE 分类模型。

    整体结构：
        image
          ↓
        ResNetTokenizer
          ↓
        image tokens
          ↓
        拼接 CLS token
          ↓
        加 position embedding
          ↓
        SwitchTransformerBlock × switch_layers
          ↓
        LayerNorm
          ↓
        取 CLS token
          ↓
        Linear classifier
          ↓
        logits

    重要说明：
        1. 模型内部会计算 aux_loss 作为 router 诊断信息。
        2. 本文件不会把 aux_loss 加进训练 loss。
        3. 本文件不实现 router balance。
        4. 本文件不实现 entropy regularization。
        5. 本文件不实现 expert diversity regularization。
        6. 本文件不实现 router consistency regularization。
        7. 默认训练时 model(x) 只返回 logits。
    """

    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        topk: int = 1,
        backbone_name: str = "resnet18",
        in_channels: int = 3,
        image_size: int = 32,
        hidden_dim: int = 128,
        switch_layers: int = 2,
        switch_heads: int = 4,
        switch_ffn_dim: int = 256,
        dropout: float = 0.1,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，当前值：{num_classes}")

        if switch_layers <= 0:
            raise ValueError(f"switch_layers 必须大于 0，当前值：{switch_layers}")

        if hidden_dim % switch_heads != 0:
            raise ValueError(
                f"hidden_dim 必须能被 switch_heads 整除，"
                f"当前 hidden_dim={hidden_dim}, switch_heads={switch_heads}"
            )

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.hidden_dim = int(hidden_dim)
        self.switch_layers = int(switch_layers)
        self.switch_heads = int(switch_heads)
        self.switch_ffn_dim = int(switch_ffn_dim)

        self.tokenizer = ResNetTokenizer(
            backbone_name=backbone_name,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
        )

        self.num_image_tokens = int(self.tokenizer.num_image_tokens)
        self.num_tokens = self.num_image_tokens + 1

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, hidden_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                SwitchTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=switch_heads,
                    ffn_dim=switch_ffn_dim,
                    num_experts=num_experts,
                    topk=topk,
                    dropout=dropout,
                    renormalize_topk_probs=renormalize_topk_probs,
                )
                for _ in range(switch_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_extra_weights()

    def _init_extra_weights(self) -> None:
        """
        初始化 Transformer 额外参数。

        ResNet backbone 使用 torchvision 默认初始化。
        这里主要初始化：
            cls_token
            pos_embed
            Linear
            LayerNorm
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | ResNetSwitchMoEOutput:
        """
        前向传播。

        默认：
            logits = model(x)

        调试 router 时：
            output = model(x, return_router_info=True)
            logits = output.logits
            router_info = output.router_info

        注意：
            训练代码里如果只写：
                logits = model(x)
                loss = CrossEntropyLoss(logits, y)

            那么 aux_loss 不会参与训练。
        """
        tokens = self.tokenizer(x)

        batch_size = tokens.shape[0]

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            [cls_tokens, tokens],
            dim=1,
        )

        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"token 数量不匹配：当前输入 token 数={tokens.shape[1]}, "
                f"模型初始化 token 数={self.num_tokens}"
            )

        tokens = tokens + self.pos_embed
        tokens = self.pos_dropout(tokens)

        if not return_router_info:
            for block in self.blocks:
                tokens = block(
                    tokens,
                    return_router_info=False,
                )

            tokens = self.norm(tokens)
            cls_feature = tokens[:, 0]
            logits = self.classifier(cls_feature)

            return logits

        router_info_by_layer: List[Dict[str, torch.Tensor]] = []
        aux_losses = []

        for layer_id, block in enumerate(self.blocks):
            tokens, layer_router_info = block(
                tokens,
                return_router_info=True,
            )

            layer_router_info = dict(layer_router_info)
            layer_router_info["layer_id"] = torch.tensor(
                layer_id,
                device=x.device,
            )

            router_info_by_layer.append(layer_router_info)
            aux_losses.append(layer_router_info["aux_loss"])

        tokens = self.norm(tokens)
        cls_feature = tokens[:, 0]
        logits = self.classifier(cls_feature)

        aux_loss = torch.stack(aux_losses).sum() if aux_losses else torch.tensor(
            0.0,
            device=x.device,
        )

        router_info = {
            "aux_loss": aux_loss,
            "router_info_by_layer": router_info_by_layer,
            "expert_counts_by_layer": [
                item["expert_counts"]
                for item in router_info_by_layer
            ],
            "sample_expert_counts_by_layer": [
                item["sample_expert_counts"]
                for item in router_info_by_layer
            ],
            "selected_experts_by_layer": [
                item["selected_experts"]
                for item in router_info_by_layer
            ],
            "topk_probs_by_layer": [
                item["topk_probs"]
                for item in router_info_by_layer
            ],
            "router_probs_by_layer": [
                item["router_probs"]
                for item in router_info_by_layer
            ],
            "router_logits_by_layer": [
                item["router_logits"]
                for item in router_info_by_layer
            ],
        }

        return ResNetSwitchMoEOutput(
            logits=logits,
            router_info=router_info,
        )


def build_resnet_switch_moe_from_cfg(cfg: Any) -> ResNetSwitchMoE:
    """
    根据 cfg 构建 ResNetSwitchMoE。

    推荐配置：

        model: resnet_switch_moe
        num_experts: 4
        topk: 2

        model_cfg:
          backbone_name: resnet18
          hidden_dim: 128
          switch_layers: 2
          switch_heads: 4
          switch_ffn_dim: 256
          dropout: 0.1
          renormalize_topk_probs: false

    注意：
        aux_loss 不会在模型里加入训练 loss。
        是否使用 aux_loss 由 client/trainer 决定。
        第一版我们不使用它。
    """
    model_cfg = _cfg_get(cfg, "model_cfg", {})

    input_shape = _cfg_get(cfg, "input_shape", (3, 32, 32))
    default_in_channels = int(input_shape[0]) if input_shape is not None else 3
    default_image_size = int(input_shape[1]) if input_shape is not None else 32

    return ResNetSwitchMoE(
        num_classes=int(_cfg_get(cfg, "num_classes")),
        num_experts=int(_cfg_get(cfg, "num_experts", 4)),
        topk=int(_cfg_get(cfg, "topk", 1)),
        backbone_name=str(
            _cfg_get(
                model_cfg,
                "backbone_name",
                "resnet18",
            )
        ),
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
        hidden_dim=int(
            _cfg_get(
                model_cfg,
                "hidden_dim",
                128,
            )
        ),
        switch_layers=int(
            _cfg_get(
                model_cfg,
                "switch_layers",
                2,
            )
        ),
        switch_heads=int(
            _cfg_get(
                model_cfg,
                "switch_heads",
                4,
            )
        ),
        switch_ffn_dim=int(
            _cfg_get(
                model_cfg,
                "switch_ffn_dim",
                256,
            )
        ),
        dropout=float(
            _cfg_get(
                model_cfg,
                "dropout",
                0.1,
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