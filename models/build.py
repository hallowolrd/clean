from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

from models.resnet_switch_moe import (
    ResNetSwitchMoE,
    build_resnet_switch_moe_from_cfg,
)


ModelBuilder = Callable[[Any], nn.Module]


MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "resnet_switch_moe": build_resnet_switch_moe_from_cfg,
}


def build_model(cfg: Any) -> nn.Module:
    """
    根据配置创建模型。

    配置示例：
        model: resnet_switch_moe

    当前支持：
        resnet_switch_moe

    后续扩展其他模型时，只需要：
        1. 新增模型文件
        2. 写一个 build_xxx_from_cfg(cfg)
        3. 在 MODEL_BUILDERS 里注册
    """
    model_name = get_model_name(cfg)

    if model_name not in MODEL_BUILDERS:
        supported = ", ".join(list_supported_models())
        raise ValueError(
            f"不支持的模型名称：{model_name}。"
            f"当前支持：{supported}"
        )

    model = MODEL_BUILDERS[model_name](cfg)

    return model


def get_model_name(cfg: Any) -> str:
    """
    从配置中读取模型名称。

    支持：
        cfg.model
        cfg.get("model")
    """
    model_name = _cfg_get(cfg, "model", None)

    if model_name is None:
        raise ValueError("配置中缺少 model 字段。")

    model_name = str(model_name).lower().strip()

    if not model_name:
        raise ValueError("配置中的 model 字段不能为空。")

    return model_name


def list_supported_models() -> List[str]:
    """
    返回当前支持的模型名称列表。
    """
    return sorted(MODEL_BUILDERS.keys())


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    统计模型参数量。

    参数：
        trainable_only:
            如果为 True，只统计 requires_grad=True 的参数。
            如果为 False，统计所有参数。
    """
    total = 0

    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue

        total += int(param.numel())

    return total


def summarize_model(model: nn.Module) -> Dict[str, int]:
    """
    返回模型参数量摘要。

    输出：
        {
            "total_params": ...,
            "trainable_params": ...
        }
    """
    return {
        "total_params": count_parameters(
            model=model,
            trainable_only=False,
        ),
        "trainable_params": count_parameters(
            model=model,
            trainable_only=True,
        ),
    }


def print_model_summary(model: nn.Module) -> None:
    """
    打印模型参数量摘要。

    这个函数主要用于调试。
    训练主流程里后面可以选择是否调用。
    """
    summary = summarize_model(model)

    print(
        "[Model] "
        f"total_params={summary['total_params']:,} | "
        f"trainable_params={summary['trainable_params']:,}"
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