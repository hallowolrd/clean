from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    """
    数据划分结果。

    client_indices:
        每个客户端对应的样本索引列表。

    client_sample_counts:
        每个客户端的样本数量。

    client_class_counts:
        每个客户端的类别分布。
        形状大致是：
            {
                client_id: {
                    class_id: count
                }
            }

    num_classes:
        数据集类别数。

    partition_method:
        数据划分方法，例如 iid / dirichlet。
    """

    client_indices: List[List[int]]
    client_sample_counts: Dict[int, int]
    client_class_counts: Dict[int, Dict[int, int]]
    num_classes: int
    partition_method: str


def partition_dataset(cfg: Any, dataset: Any) -> PartitionResult:
    """
    根据配置划分训练集。

    第一版支持：
        1. iid
        2. dirichlet

    默认使用 dirichlet，因为你的 FL 实验主要是 non-IID 场景。

    需要的配置字段：
        cfg.num_clients
        cfg.alpha
        cfg.seed

    可选配置字段：
        cfg.partition_method，默认 dirichlet
        cfg.min_samples_per_client，默认 1
        cfg.partition_max_retries，默认 100
    """
    num_clients = int(cfg.num_clients)
    seed = int(cfg.seed)

    partition_method = str(_cfg_get(cfg, "partition_method", "dirichlet")).lower()

    targets = get_dataset_targets(dataset)
    num_classes = infer_num_classes(targets)

    if partition_method == "iid":
        client_indices = build_iid_partition(
            targets=targets,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition_method == "dirichlet":
        alpha = float(cfg.alpha)
        min_samples_per_client = int(_cfg_get(cfg, "min_samples_per_client", 1))
        max_retries = int(_cfg_get(cfg, "partition_max_retries", 100))

        client_indices = build_dirichlet_partition(
            targets=targets,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_client=min_samples_per_client,
            max_retries=max_retries,
        )

    else:
        raise ValueError(
            f"不支持的数据划分方法：{partition_method}。"
            f"当前支持：iid, dirichlet"
        )

    validate_partition(
        client_indices=client_indices,
        dataset_size=len(dataset),
        num_clients=num_clients,
    )

    client_sample_counts = compute_client_sample_counts(client_indices)
    client_class_counts = compute_client_class_counts(
        client_indices=client_indices,
        targets=targets,
        num_classes=num_classes,
    )

    return PartitionResult(
        client_indices=client_indices,
        client_sample_counts=client_sample_counts,
        client_class_counts=client_class_counts,
        num_classes=num_classes,
        partition_method=partition_method,
    )


def build_iid_partition(
    targets: Sequence[int],
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    """
    IID 划分。

    做法：
        1. 打乱全部样本索引
        2. 均匀切成 num_clients 份

    注意：
        这里不会保证每个客户端类别完全均衡。
        它只是从整体数据集中随机均分。
    """
    _validate_num_clients(num_clients)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(targets))
    rng.shuffle(all_indices)

    split_indices = np.array_split(all_indices, num_clients)

    return [
        split.astype(int).tolist()
        for split in split_indices
    ]


def build_dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> List[List[int]]:
    """
    Dirichlet non-IID 划分。

    核心思想：
        对每个类别 c：
            1. 找出所有属于类别 c 的样本
            2. 从 Dirichlet(alpha) 采样一个客户端比例
            3. 按这个比例把该类别样本分给不同客户端

    alpha 越小：
        客户端类别分布越不均衡。

    alpha 越大：
        客户端类别分布越接近 IID。
    """
    _validate_num_clients(num_clients)

    if alpha <= 0:
        raise ValueError(f"alpha 必须大于 0，当前值：{alpha}")

    if min_samples_per_client < 0:
        raise ValueError(
            f"min_samples_per_client 不能小于 0，当前值：{min_samples_per_client}"
        )

    if max_retries <= 0:
        raise ValueError(f"max_retries 必须大于 0，当前值：{max_retries}")

    targets_array = np.asarray(targets, dtype=np.int64)
    num_classes = infer_num_classes(targets_array)

    for retry_id in range(max_retries):
        rng = np.random.default_rng(seed + retry_id)
        client_indices: List[List[int]] = [
            []
            for _ in range(num_clients)
        ]

        for class_id in range(num_classes):
            class_indices = np.where(targets_array == class_id)[0]
            rng.shuffle(class_indices)

            # 为当前类别采样每个客户端的分配比例
            proportions = rng.dirichlet(
                alpha=np.full(num_clients, alpha, dtype=np.float64)
            )

            # 根据比例切分该类别样本
            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.astype(int).tolist())

        # 每个客户端内部再打乱一次，避免类别块顺序过于明显
        for client_id in range(num_clients):
            rng.shuffle(client_indices[client_id])

        sample_counts = [
            len(indices)
            for indices in client_indices
        ]

        if min(sample_counts) >= min_samples_per_client:
            return client_indices

    raise RuntimeError(
        "Dirichlet 数据划分失败："
        f"尝试 {max_retries} 次后，仍存在客户端样本数小于 "
        f"{min_samples_per_client}。"
        "可以尝试增大 alpha，或降低 min_samples_per_client。"
    )


def get_dataset_targets(dataset: Any) -> List[int]:
    """
    从 torchvision dataset 中取出标签。

    CIFAR10 / CIFAR100 通常有 dataset.targets。
    为了更通用，也兼容 dataset.labels。
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets

    elif hasattr(dataset, "labels"):
        targets = dataset.labels

    else:
        raise AttributeError(
            "无法从 dataset 中读取标签。"
            "当前只支持包含 targets 或 labels 属性的数据集。"
        )

    return [
        int(label)
        for label in targets
    ]


def infer_num_classes(targets: Sequence[int]) -> int:
    """
    根据标签推断类别数。

    假设标签是从 0 开始的整数类别。
    """
    if len(targets) == 0:
        raise ValueError("targets 为空，无法推断类别数。")

    return int(max(targets)) + 1


def compute_client_sample_counts(
    client_indices: Sequence[Sequence[int]],
) -> Dict[int, int]:
    """
    统计每个客户端的样本数量。
    """
    return {
        client_id: len(indices)
        for client_id, indices in enumerate(client_indices)
    }


def compute_client_class_counts(
    client_indices: Sequence[Sequence[int]],
    targets: Sequence[int],
    num_classes: int,
) -> Dict[int, Dict[int, int]]:
    """
    统计每个客户端的类别分布。
    """
    targets_array = np.asarray(targets, dtype=np.int64)

    result: Dict[int, Dict[int, int]] = {}

    for client_id, indices in enumerate(client_indices):
        class_counts = {
            class_id: 0
            for class_id in range(num_classes)
        }

        if len(indices) > 0:
            client_targets = targets_array[np.asarray(indices, dtype=np.int64)]
            unique_classes, counts = np.unique(client_targets, return_counts=True)

            for class_id, count in zip(unique_classes, counts):
                class_counts[int(class_id)] = int(count)

        result[client_id] = class_counts

    return result


def validate_partition(
    client_indices: Sequence[Sequence[int]],
    dataset_size: int,
    num_clients: int,
) -> None:
    """
    检查数据划分是否合法。

    检查内容：
        1. 客户端数量是否正确
        2. 所有样本是否都被分配
        3. 是否存在重复样本
        4. 是否存在越界索引
    """
    if len(client_indices) != num_clients:
        raise ValueError(
            f"客户端数量不匹配：期望 {num_clients}，"
            f"实际 {len(client_indices)}"
        )

    all_indices: List[int] = []

    for client_id, indices in enumerate(client_indices):
        for index in indices:
            if index < 0 or index >= dataset_size:
                raise ValueError(
                    f"客户端 {client_id} 存在越界样本索引：{index}，"
                    f"数据集大小：{dataset_size}"
                )

            all_indices.append(int(index))

    if len(all_indices) != dataset_size:
        raise ValueError(
            f"划分后样本总数不等于数据集大小："
            f"划分后 {len(all_indices)}，数据集 {dataset_size}"
        )

    unique_indices = set(all_indices)

    if len(unique_indices) != dataset_size:
        raise ValueError(
            f"数据划分存在重复或遗漏："
            f"唯一索引数 {len(unique_indices)}，数据集大小 {dataset_size}"
        )


def partition_summary_to_dict(partition: PartitionResult) -> Dict[str, Any]:
    """
    把划分结果中的摘要信息转成普通 dict。

    用于后续写日志、保存 json。
    注意：
        不保存完整 client_indices，避免日志文件太大。
    """
    return {
        "partition_method": partition.partition_method,
        "num_classes": partition.num_classes,
        "num_clients": len(partition.client_indices),
        "client_sample_counts": partition.client_sample_counts,
        "client_class_counts": partition.client_class_counts,
    }


def _validate_num_clients(num_clients: int) -> None:
    """检查客户端数量是否合法。"""
    if not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError(f"num_clients 必须是正整数，当前值：{num_clients}")


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