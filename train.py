from __future__ import annotations

# ============================================================
# 注意：
# 下面这一段必须放在 import torch 之前。
#
# 原因：
# 1. CUBLAS_WORKSPACE_CONFIG 必须在 CUDA / cuBLAS 初始化前设置。
# 2. PYTHONHASHSEED 必须在 Python 解释器启动前生效。
#    所以如果当前进程的 PYTHONHASHSEED 和配置 seed 不一致，
#    这里会自动重新执行一次当前 Python 命令。
# ============================================================

import os
import sys
from pathlib import Path


def _get_cli_arg_value(name: str) -> str | None:
    """
    从命令行参数中读取指定参数值。

    支持两种写法：
    1. --config configs/uniform.yaml
    2. --config=configs/uniform.yaml
    """
    prefix = name + "="
    argv = sys.argv

    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            return argv[idx + 1]

        if arg.startswith(prefix):
            return arg[len(prefix):]

    return None


def _clean_simple_yaml_value(value: str) -> str:
    """
    清理简单 YAML 标量值。

    这里只服务于启动早期读取 seed / include，
    不替代项目里的正式 load_config。
    """
    value = value.strip()

    # 去掉简单引号
    if len(value) >= 2:
        if (value[0] == value[-1]) and value[0] in {"'", '"'}:
            value = value[1:-1]

    return value.strip()


def _read_top_level_scalar_from_yaml_like_file(
    path: Path,
    key: str,
    visited: set[Path] | None = None,
) -> str | None:
    """
    在 import torch 之前，轻量读取 YAML 文件里的顶层简单标量。

    目的：
    - 提前读取 seed，让 PYTHONHASHSEED 可以跟随训练 seed。
    - 支持你的配置风格：uniform.yaml 里 include: base.yaml。

    注意：
    - 这不是完整 YAML 解析器；
    - 只用于启动阶段读取 seed / include；
    - 正式配置仍然由 utils.config.load_config() 读取。
    """
    if visited is None:
        visited = set()

    path = path.resolve()
    if path in visited:
        return None
    visited.add(path)

    if not path.exists():
        return None

    include_path: Path | None = None
    local_value: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            # 去掉行内注释。这里足够处理当前配置里的 seed / include。
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            name = left.strip()
            value = _clean_simple_yaml_value(right)

            if name == "include":
                include_path = (path.parent / value).resolve()
                continue

            if name == key:
                local_value = value
                continue

    # 当前配置里的 seed 优先级高于 include 里的 seed。
    if local_value is not None:
        return local_value

    if include_path is not None:
        return _read_top_level_scalar_from_yaml_like_file(
            path=include_path,
            key=key,
            visited=visited,
        )

    return None


def _prepare_deterministic_env_before_torch() -> None:
    """
    在 torch / CUDA 初始化前准备确定性相关环境变量。

    CUBLAS_WORKSPACE_CONFIG:
        控制 cuBLAS 矩阵乘法的确定性。
        如果用户已经手动设置了 :16:8 或其他合法值，这里不覆盖。

    PYTHONHASHSEED:
        让 Python hash 随机种子跟随配置文件里的 seed。
        该变量必须在解释器启动前生效，所以必要时自动 re-exec 一次。
    """
    # cuBLAS 确定性配置。显存不紧张时优先使用 :4096:8。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    config_arg = _get_cli_arg_value("--config")
    config_seed = None

    if config_arg is not None:
        config_seed = _read_top_level_scalar_from_yaml_like_file(
            path=Path(config_arg),
            key="seed",
        )

    # 正常情况下 base.yaml 里有 seed。
    # 如果读取失败，就退回到已有 PYTHONHASHSEED；再没有就用 0。
    target_hash_seed = str(
        config_seed
        if config_seed is not None
        else os.environ.get("PYTHONHASHSEED", "0")
    )

    current_hash_seed = os.environ.get("PYTHONHASHSEED")
    already_reexec = os.environ.get("CLEAN_REEXEC_FOR_PYTHONHASHSEED") == "1"

    if current_hash_seed != target_hash_seed:
        os.environ["PYTHONHASHSEED"] = target_hash_seed

        # PYTHONHASHSEED 必须在 Python 解释器启动前生效。
        # 当前进程已经启动了，所以这里自动重启一次当前命令。
        if not already_reexec:
            os.environ["CLEAN_REEXEC_FOR_PYTHONHASHSEED"] = "1"
            os.execvpe(
                sys.executable,
                [sys.executable] + sys.argv,
                os.environ,
            )


_prepare_deterministic_env_before_torch()


import argparse
import csv
import json
import traceback
from typing import Any, Dict, List

import torch

from data.datasets import build_datasets
from data.loaders import build_dataloaders
from data.partition import partition_dataset, partition_summary_to_dict
from fl.server import build_server, resolve_device
from utils.config import ensure_run_dir, load_config, save_config
from utils.logging import tee_output_to_file
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="FL + MoE training entrypoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径，例如 configs/base.yaml",
    )

    return parser.parse_args()


def main() -> int:
    """
    训练入口。

    这里负责：
    1. 读取配置
    2. 设置随机种子
    3. 创建输出目录
    4. 打开 train.log 双写
    5. 调用 run_training() 执行实际训练
    """
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(
        seed=int(cfg.seed),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    run_dir = Path(ensure_run_dir(cfg))
    log_path = run_dir / "train.log"

    with tee_output_to_file(log_path):
        try:
            return run_training(
                args=args,
                cfg=cfg,
                run_dir=run_dir,
            )
        except Exception:
            print()
            print("=" * 80)
            print("[Train] Failed")
            print("=" * 80)
            traceback.print_exc()
            return 1


def run_training(
    args: argparse.Namespace,
    cfg: Any,
    run_dir: Path,
) -> int:
    """
    实际训练流程。

    这个函数会被 main() 包在 tee_output_to_file() 里面，
    所以这里所有 print 和报错都会同时写入控制台和 train.log。

    总流程：
    1. 保存配置
    2. 解析设备
    3. 加载数据集
    4. 划分客户端数据
    5. 创建 DataLoader
    6. 创建 FLServer
    7. 执行联邦训练
    8. 保存结果
    """
    if bool(cfg.get("logging.save_config", True)):
        save_config(
            cfg=cfg,
            output_path=run_dir / "config_used.yaml",
        )

    device = resolve_device(cfg)

    print()
    print("=" * 80)
    print("[Train] Start")
    print(f"[Train] config: {args.config}")
    print(f"[Train] run_name: {cfg.run_name}")
    print(f"[Train] run_dir: {cfg.run_dir}")
    print(f"[Train] log_file: {run_dir / 'train.log'}")
    print(f"[Train] device: {device}")
    print("=" * 80)
    print()

    dataset_bundle = build_datasets(cfg)

    print(
        "[Data] "
        f"dataset={dataset_bundle.name} | "
        f"num_classes={dataset_bundle.num_classes} | "
        f"input_shape={dataset_bundle.input_shape}"
    )
    print(
        "[Data] "
        f"train_size={len(dataset_bundle.train_dataset)} | "
        f"train_evidence_size={len(dataset_bundle.train_evidence_dataset)} | "
        f"test_size={len(dataset_bundle.test_dataset)}"
    )

    partition = partition_dataset(
        cfg=cfg,
        dataset=dataset_bundle.train_dataset,
    )

    save_partition_summary(
        partition=partition,
        output_path=run_dir / "partition_summary.json",
    )

    # train_dataset：
    # 正常本地训练使用，可以带随机数据增强。
    #
    # train_evidence_dataset：
    # Fisher / K-FAC evidence pass 使用，不带随机数据增强。
    # 它和 train_dataset 使用同一份官方训练集，并复用同一份 client_indices。
    loader_bundle = build_dataloaders(
        cfg=cfg,
        train_dataset=dataset_bundle.train_dataset,
        train_evidence_dataset=dataset_bundle.train_evidence_dataset,
        test_dataset=dataset_bundle.test_dataset,
        client_indices=partition.client_indices,
    )

    # client_evidence_loaders 只负责透传给 client。
    # server 本身不关心 Fisher / K-FAC 的统计细节，保持极致解耦。
    server = build_server(
        cfg=cfg,
        client_loaders=loader_bundle.client_loaders,
        client_evidence_loaders=loader_bundle.client_evidence_loaders,
        test_loader=loader_bundle.test_loader,
        device=device,
    )

    train_result = server.train()

    save_train_outputs(
        train_result=train_result,
        output_dir=run_dir,
        save_csv=bool(cfg.get("logging.save_results_csv", True)),
    )

    print()
    print("=" * 80)
    print("[Train] Done")
    print(f"[Train] best_acc: {train_result.best_acc:.2f}%")
    print(f"[Train] best_round: {train_result.best_round}")
    print(f"[Train] outputs saved to: {cfg.run_dir}")
    print("=" * 80)

    return 0


def save_partition_summary(
    partition: Any,
    output_path: Path,
) -> None:
    """
    保存数据划分摘要。

    注意：
    不保存完整 client_indices。
    这里只保存每个客户端样本数、类别分布等轻量信息。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = partition_summary_to_dict(partition)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_train_outputs(
    train_result: Any,
    output_dir: Path,
    save_csv: bool = True,
) -> None:
    """
    保存训练输出。

    输出文件：
    summary.json: 完整训练摘要。
    results.csv: 每轮核心指标，方便直接画图或导入 Excel。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(train_result.to_dict()),
            f,
            ensure_ascii=False,
            indent=2,
        )

    if save_csv:
        csv_path = output_dir / "results.csv"
        save_round_results_csv(
            round_results=train_result.round_results,
            output_path=csv_path,
        )


def save_round_results_csv(
    round_results: List[Any],
    output_path: Path,
) -> None:
    """
    保存每轮训练结果到 CSV。

    CSV 只保存最常用的核心指标：
    round_id
    selected_clients
    avg_train_loss
    avg_train_acc
    test_loss
    test_acc
    best_acc
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "round_id",
        "selected_clients",
        "avg_train_loss",
        "avg_train_acc",
        "test_loss",
        "test_acc",
        "best_acc",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for item in round_results:
            aggregation_info = item.aggregation_info

            row = {
                "round_id": int(item.round_id),
                "selected_clients": " ".join(
                    str(client_id)
                    for client_id in item.selected_clients
                ),
                "avg_train_loss": aggregation_info.get(
                    "avg_train_loss",
                    "",
                ),
                "avg_train_acc": aggregation_info.get(
                    "avg_train_acc",
                    "",
                ),
                "test_loss": float(item.test_loss),
                "test_acc": float(item.test_acc),
                "best_acc": float(item.best_acc),
            }

            writer.writerow(row)


def make_json_safe(obj: Any) -> Any:
    """
    把对象转换成 JSON 可保存格式。

    主要处理：
    torch.Tensor
    torch.device
    Path
    dict
    list / tuple
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    if hasattr(obj, "to_dict"):
        return make_json_safe(obj.to_dict())

    return obj


if __name__ == "__main__":
    raise SystemExit(main())