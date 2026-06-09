from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ResNet18ClientSide(nn.Module):
    """
    纯 FL 版 ResNet18 的前半部分。

    说明：
    - 这部分来自 fl 仓库 fedavg.py 里的 ResNet18_client_side。
    - 虽然名字叫 client_side，但在当前纯 FL 场景中，客户端本地训练的是完整模型。
    - 这里保留这个拆分，只是为了最大程度保持原始模型结构不变。
    """

    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        保持原始 fedavg.py 的初始化方式。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, 3, 32, 32]

        输出：
            residual3: [B, 64, 32, 32]
        """
        residual1 = self.layer1(x)

        out1 = self.layer2(residual1)
        out1 = out1 + residual1
        residual2 = F.relu(out1)

        out2 = self.layer3(residual2)
        out2 = out2 + residual2
        residual3 = F.relu(out2)

        return residual3


class BaseBlock(nn.Module):
    """
    ResNet BasicBlock。

    说明：
    - 对应 fl 仓库 fedavg.py 里的 Baseblock。
    - 为了命名更规范，这里改成 BaseBlock。
    - 结构和原实现保持一致。
    """

    expansion = 1

    def __init__(
        self,
        input_planes: int,
        planes: int,
        stride: int = 1,
        dim_change: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_planes,
            out_channels=planes,
            stride=stride,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            stride=1,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.dim_change = dim_change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]

        输出：
            output: [B, planes, H', W']
        """
        residual = x

        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            residual = self.dim_change(residual)

        output = output + residual
        output = F.relu(output)

        return output


class ResNet18ServerSide(nn.Module):
    """
    纯 FL 版 ResNet18 的后半部分。

    说明：
    - 这部分来自 fl 仓库 fedavg.py 里的 ResNet18_server_side。
    - 输入是 client_side 输出的 [B, 64, 32, 32] 特征图。
    - 输出是分类 logits。
    """

    def __init__(
        self,
        block: type[BaseBlock],
        num_layers: list[int],
        classes: int,
    ) -> None:
        super().__init__()

        self.input_planes = 64

        self.layer4 = self._make_layer(
            block=block,
            planes=128,
            num_layers=num_layers[0],
            stride=2,
        )
        self.layer5 = self._make_layer(
            block=block,
            planes=256,
            num_layers=num_layers[1],
            stride=2,
        )
        self.layer6 = self._make_layer(
            block=block,
            planes=512,
            num_layers=num_layers[2],
            stride=2,
        )

        self.fc = nn.Linear(512 * block.expansion, classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        保持原始 fedavg.py 的初始化方式。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BaseBlock],
        planes: int,
        num_layers: int,
        stride: int = 2,
    ) -> nn.Sequential:
        """
        构建一个 ResNet stage。

        参数：
            block: BasicBlock 类型。
            planes: 当前 stage 的输出通道数。
            num_layers: 当前 stage 中 block 数量。
            stride: 第一个 block 的 stride。
        """
        dim_change = None

        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.input_planes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                input_planes=self.input_planes,
                planes=planes,
                stride=stride,
                dim_change=dim_change,
            )
        )

        self.input_planes = planes * block.expansion

        for _ in range(1, num_layers):
            layers.append(
                block(
                    input_planes=self.input_planes,
                    planes=planes,
                )
            )
            self.input_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, 64, 32, 32]

        输出：
            logits: [B, num_classes]

        注意：
            这里保留原始 fedavg.py 的 F.avg_pool2d(x, 4)。
            因此该模型默认用于 CIFAR10 / CIFAR100 这种 32x32 输入。
        """
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class ResNet18FedAvg(nn.Module):
    """
    纯 FL 的 ResNet18 FedAvg baseline 模型。

    整体结构：
        image
        -> ResNet18ClientSide
        -> ResNet18ServerSide
        -> logits

    重要说明：
    - 这个模型没有 MoE。
    - 这个模型没有 expert。
    - 这个模型没有 router。
    - 后续做 pure-FL fisher_only_global / fisher_history_wolf_global 时，
      应该把整个模型作为一个整体参数组聚合。
    """

    def __init__(self, classes: int = 10) -> None:
        super().__init__()

        if classes <= 0:
            raise ValueError(f"classes 必须大于 0，当前值：{classes}")

        self.classes = int(classes)

        self.client_side = ResNet18ClientSide()
        self.server_side = ResNet18ServerSide(
            block=BaseBlock,
            num_layers=[3, 3, 3],
            classes=self.classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, 3, 32, 32]

        输出：
            logits: [B, num_classes]
        """
        x = self.client_side(x)
        logits = self.server_side(x)
        return logits


def build_resnet18_fedavg_from_cfg(cfg: Any) -> ResNet18FedAvg:
    """
    根据 clean_ablation 的 cfg 构建 ResNet18FedAvg。

    推荐配置示例：
        model: resnet18_fedavg
        num_classes: 10
        input_shape: [3, 32, 32]

    注意：
    - 当前模型结构默认适配 32x32 图像。
    - input_shape 这里只做检查和提示，不改变模型结构。
    """
    num_classes = int(_cfg_get(cfg, "num_classes", 10))

    input_shape = _cfg_get(cfg, "input_shape", (3, 32, 32))
    if input_shape is not None:
        in_channels = int(input_shape[0])
        image_size = int(input_shape[1])

        if in_channels != 3:
            raise ValueError(
                "ResNet18FedAvg 当前按原始 fedavg.py 设计，只支持 3 通道输入，"
                f"当前 input_shape={input_shape}"
            )

        if image_size != 32:
            raise ValueError(
                "ResNet18FedAvg 当前保留原始 F.avg_pool2d(x, 4) 结构，"
                "默认只支持 32x32 输入。"
                f"当前 input_shape={input_shape}"
            )

    return ResNet18FedAvg(classes=num_classes)


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