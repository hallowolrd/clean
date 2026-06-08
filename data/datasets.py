from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from torchvision import datasets, transforms


# =========================
# 数据集元信息注册区
# =========================
# 后续新增数据集时，优先在这里注册。
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "cifar10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "input_shape": (3, 32, 32),
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


@dataclass(frozen=True)
class DatasetBundle:
    """
    数据集打包结果。

    这里只保存原始 train / evidence / test dataset。
    客户端划分和 DataLoader 构建不要放在这里。

    train_dataset:
        正常本地训练使用的数据集，可以带随机数据增强。

    train_evidence_dataset:
        Fisher / K-FAC evidence pass 使用的数据集。
        它和 train_dataset 使用同一份官方训练集，但 transform 不包含随机增强。
        这样客户端本地训练完成后额外做 forward + backward 统计 Fisher 时，
        不会因为 RandomCrop / RandomHorizontalFlip 导致 evidence 统计不稳定。

    test_dataset:
        服务端测试集，不使用随机增强。
    """

    name: str
    train_dataset: Any
    train_evidence_dataset: Any
    test_dataset: Any
    num_classes: int
    input_shape: Tuple[int, int, int]


def build_datasets(cfg: Any) -> DatasetBundle:
    """
    根据配置构建数据集。

    输入：
        cfg: 全局配置对象，需要至少包含：
            cfg.dataset
            cfg.data_root

    输出：
        DatasetBundle:
            train_dataset:
                原始训练集，后续会交给 data/partition.py 划分给客户端。
                这个数据集用于正常本地训练，可以使用随机数据增强。

            train_evidence_dataset:
                原始训练集的无随机增强版本。
                后续使用同一份 client_indices 划分给客户端，用于本地训练完成后的
                Fisher / K-FAC evidence pass。

            test_dataset:
                服务端测试集。

            num_classes:
                类别数。

            input_shape:
                输入图片形状。
    """
    dataset_name = str(cfg.dataset).lower()
    data_root = Path(cfg.data_root)

    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"不支持的数据集：{dataset_name}。"
            f"当前支持：{sorted(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[dataset_name]

    train_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=_cfg_get(cfg, "data_augmentation", True),
    )

    # evidence transform 固定不使用随机增强。
    # 目的：客户端本地训练完成后额外做一轮 Fisher/K-FAC 统计时，
    # 输入数据保持确定，避免随机裁剪/翻转污染 evidence。
    train_evidence_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=False,
    )

    test_transform = build_test_transform(dataset_name=dataset_name)

    download = bool(_cfg_get(cfg, "download_data", True))

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            transform=train_transform,
            download=download,
        )

        train_evidence_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            transform=train_evidence_transform,
            download=download,
        )

        test_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=False,
            transform=test_transform,
            download=download,
        )

    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            transform=train_transform,
            download=download,
        )

        train_evidence_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            transform=train_evidence_transform,
            download=download,
        )

        test_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=False,
            transform=test_transform,
            download=download,
        )

    else:
        # 理论上前面已经拦住了，这里只是防御式写法。
        raise ValueError(f"未实现的数据集加载逻辑：{dataset_name}")

    return DatasetBundle(
        name=dataset_name,
        train_dataset=train_dataset,
        train_evidence_dataset=train_evidence_dataset,
        test_dataset=test_dataset,
        num_classes=int(info["num_classes"]),
        input_shape=tuple(info["input_shape"]),
    )


def build_train_transform(
    dataset_name: str,
    use_augmentation: bool = True,
) -> Callable:
    """
    构建训练集 transform。

    CIFAR 训练集默认使用：
        RandomCrop
        RandomHorizontalFlip
        ToTensor
        Normalize

    如果 use_augmentation=False，则只使用：
        ToTensor
        Normalize

    注意：
        train_evidence_dataset 会调用 use_augmentation=False。
        这样 Fisher / K-FAC evidence pass 不会使用随机数据增强。
    """
    dataset_name = dataset_name.lower()
    mean, std = get_normalization_stats(dataset_name)

    transform_list = []

    if use_augmentation:
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(transform_list)


def build_test_transform(dataset_name: str) -> Callable:
    """
    构建测试集 transform。

    测试集不使用随机增强，保证评估稳定。
    """
    dataset_name = dataset_name.lower()
    mean, std = get_normalization_stats(dataset_name)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    获取数据集元信息。

    后续模型构建时可以用：
        num_classes
        input_shape
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"不支持的数据集：{dataset_name}。"
            f"当前支持：{sorted(DATASET_INFO.keys())}"
        )

    return dict(DATASET_INFO[dataset_name])


def get_num_classes(dataset_name: str) -> int:
    """获取数据集类别数。"""
    return int(get_dataset_info(dataset_name)["num_classes"])


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    """获取输入图片形状。"""
    return tuple(get_dataset_info(dataset_name)["input_shape"])


def get_normalization_stats(
    dataset_name: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    获取数据集归一化均值和标准差。
    """
    info = get_dataset_info(dataset_name)
    return tuple(info["mean"]), tuple(info["std"])


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