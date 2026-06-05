from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.seed import build_torch_generator, seed_worker


@dataclass(frozen=True)
class DataLoaderBundle:
    """
    DataLoader 打包结果。

    client_loaders:
        每个客户端对应一个训练 DataLoader。

    test_loader:
        服务端测试集 DataLoader。

    client_datasets:
        每个客户端对应的 Subset 数据集。

    client_sample_counts:
        每个客户端的样本数量。
    """

    client_loaders: List[DataLoader]
    test_loader: DataLoader
    client_datasets: List[Subset]
    client_sample_counts: Dict[int, int]


def build_dataloaders(
    cfg: Any,
    train_dataset: Dataset,
    test_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
) -> DataLoaderBundle:
    """
    根据客户端样本索引构建 DataLoader。

    这个函数只负责：
        1. 把 train_dataset 切成多个客户端 Subset
        2. 为每个客户端创建训练 DataLoader
        3. 为服务端创建测试 DataLoader

    不负责：
        1. 加载原始数据集
        2. 生成 Dirichlet 划分
        3. 本地训练
        4. 参数聚合
    """
    batch_size = int(cfg.batch_size)
    test_batch_size = int(cfg.test_batch_size)
    num_workers = int(cfg.num_workers)
    seed = int(cfg.seed)

    pin_memory = _infer_pin_memory(cfg)

    client_datasets = build_client_datasets(
        train_dataset=train_dataset,
        client_indices=client_indices,
    )

    client_loaders = build_client_train_loaders(
        cfg=cfg,
        client_datasets=client_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    test_loader = build_test_loader(
        test_dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    client_sample_counts = {
        client_id: len(client_dataset)
        for client_id, client_dataset in enumerate(client_datasets)
    }

    return DataLoaderBundle(
        client_loaders=client_loaders,
        test_loader=test_loader,
        client_datasets=client_datasets,
        client_sample_counts=client_sample_counts,
    )


def build_client_datasets(
    train_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
) -> List[Subset]:
    """
    根据 client_indices 创建客户端 Subset。

    每个客户端只看到自己对应的训练样本。
    """
    client_datasets: List[Subset] = []

    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            raise ValueError(
                f"客户端 {client_id} 没有训练样本，"
                "请检查数据划分结果。"
            )

        client_dataset = Subset(
            train_dataset,
            list(indices),
        )
        client_datasets.append(client_dataset)

    return client_datasets


def build_client_train_loaders(
    cfg: Any,
    client_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> List[DataLoader]:
    """
    为每个客户端创建训练 DataLoader。

    训练集默认 shuffle=True。
    每个客户端使用不同的 generator seed，避免所有客户端 shuffle 顺序完全一致。
    """
    client_loaders: List[DataLoader] = []

    drop_last = bool(_cfg_get(cfg, "drop_last", False))
    persistent_workers = num_workers > 0

    for client_id, client_dataset in enumerate(client_datasets):
        generator = build_torch_generator(seed + client_id)

        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=persistent_workers,
        )

        client_loaders.append(loader)

    return client_loaders


def build_test_loader(
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """
    创建服务端测试 DataLoader。

    测试集必须 shuffle=False，保证每次评估顺序稳定。
    """
    persistent_workers = num_workers > 0

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(seed),
        persistent_workers=persistent_workers,
    )


def _infer_pin_memory(cfg: Any) -> bool:
    """
    判断 DataLoader 是否启用 pin_memory。

    规则：
        1. 如果配置里显式写了 pin_memory，就使用配置值
        2. 如果 device 是 cuda 或 auto，就默认启用
        3. 如果 device 是 cpu，就默认关闭
    """
    explicit_pin_memory = _cfg_get(cfg, "pin_memory", None)

    if explicit_pin_memory is not None:
        return bool(explicit_pin_memory)

    device = str(_cfg_get(cfg, "device", "auto")).lower()

    if device == "cpu":
        return False

    return True


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """
    兼容 ConfigNode 和普通对象的配置读取。

    支持：
        cfg.get("xxx", default)
        cfg.xxx
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)