from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def disable_tf32() -> None:
    """
    关闭 TF32，减少 Ampere 及以后 NVIDIA GPU 上的数值差异。

    说明：
        1. torch.backends.cuda.matmul.allow_tf32 控制 Linear / matmul / bmm 等矩阵乘法是否允许 TF32。
        2. torch.backends.cudnn.allow_tf32 控制 cuDNN 卷积是否允许 TF32。
        3. torch.set_float32_matmul_precision("highest") 是 PyTorch 2.x 的补充设置，
           表示 float32 矩阵乘法尽量使用更高精度路径。
    """
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: Optional[bool] = None,
) -> None:
    """
    设置全局随机种子。

    作用范围：
        1. Python random
        2. NumPy
        3. PyTorch CPU
        4. PyTorch CUDA
        5. Python hash seed

    参数：
        seed:
            随机种子。

        deterministic:
            是否开启 PyTorch 确定性模式。
            如果为 True，实验更容易复现，但训练速度可能变慢。

        benchmark:
            是否开启 cudnn.benchmark。
            如果为 None，则根据 deterministic 自动决定。
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int，当前类型：{type(seed)}")

    if seed < 0:
        raise ValueError(f"seed 必须是非负整数，当前值：{seed}")

    # 固定 Python hash 随机性
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 固定 Python / NumPy 随机性
    random.seed(seed)
    np.random.seed(seed)

    # 固定 PyTorch 随机性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭 TF32，避免 Ampere 及以后 GPU 上使用低精度 Tensor Core 路径。
    # 这会让训练稍慢一点，但数值更稳定，更适合做可复现实验。
    disable_tf32()

    # 控制 cudnn 行为
    torch.backends.cudnn.deterministic = deterministic

    if benchmark is None:
        torch.backends.cudnn.benchmark = not deterministic
    else:
        torch.backends.cudnn.benchmark = benchmark

    # PyTorch 确定性算法开关
    # warn_only=True 可以避免部分算子不支持确定性时直接崩掉。
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker 的随机种子初始化函数。

    用法：
        DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            generator=build_torch_generator(seed),
        )
    """
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    """
    创建带固定随机种子的 torch.Generator。

    主要用于 DataLoader，保证 shuffle 更可复现。
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int，当前类型：{type(seed)}")

    if seed < 0:
        raise ValueError(f"seed 必须是非负整数，当前值：{seed}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator