from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from data.datasets import build_datasets
from data.loaders import build_dataloaders
from data.partition import partition_dataset, partition_summary_to_dict
from fl.server import build_server, resolve_device
from utils.config import ensure_run_dir, load_config, save_config
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

    总流程：
        1. 读取配置
        2. 设置随机种子
        3. 创建输出目录
        4. 加载数据集
        5. 划分客户端数据
        6. 创建 DataLoader
        7. 创建 FLServer
        8. 执行联邦训练
        9. 保存结果
    """
    args = parse_args()

    cfg = load_config(args.config)

    set_seed(
        seed=int(cfg.seed),
        deterministic=bool(cfg.get("deterministic", False)),
    )

    run_dir = ensure_run_dir(cfg)

    if bool(cfg.get("logging.save_config", True)):
        save_config(
            cfg=cfg,
            output_path=Path(run_dir) / "config_used.yaml",
        )

    device = resolve_device(cfg)

    print()
    print("=" * 80)
    print("[Train] Start")
    print(f"[Train] config: {args.config}")
    print(f"[Train] run_name: {cfg.run_name}")
    print(f"[Train] run_dir: {cfg.run_dir}")
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
        f"test_size={len(dataset_bundle.test_dataset)}"
    )

    partition = partition_dataset(
        cfg=cfg,
        dataset=dataset_bundle.train_dataset,
    )

    save_partition_summary(
        partition=partition,
        output_path=Path(run_dir) / "partition_summary.json",
    )

    loader_bundle = build_dataloaders(
        cfg=cfg,
        train_dataset=dataset_bundle.train_dataset,
        test_dataset=dataset_bundle.test_dataset,
        client_indices=partition.client_indices,
    )

    server = build_server(
        cfg=cfg,
        client_loaders=loader_bundle.client_loaders,
        test_loader=loader_bundle.test_loader,
        device=device,
    )

    train_result = server.train()

    save_train_outputs(
        train_result=train_result,
        output_dir=Path(run_dir),
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
        summary.json:
            完整训练摘要。

        results.csv:
            每轮核心指标，方便直接画图或导入 Excel。
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