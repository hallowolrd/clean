#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test_acc curves from federated-learning results.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="总实验目录，例如 outputs/cifar10_resnet_cifar",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="test_acc 滑动平均窗口，默认 5；设为 1 表示不平滑。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="可选。只比较指定 dataset；默认从 config_used.yaml 自动识别。",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="可选。只比较指定 backbone；默认从 config_used.yaml 自动识别。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选。图片保存目录；默认保存到 --input-dir。",
    )
    return parser.parse_args()


def load_run_metadata(csv_path: Path) -> Tuple[str | None, str | None, str]:
    config_path = csv_path.parent / "config_used.yaml"

    if not config_path.is_file():
        return None, None, csv_path.parent.name

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset = cfg.get("dataset")

    model_cfg = cfg.get("model_cfg", {}) or {}
    backbone = model_cfg.get("backbone") if isinstance(model_cfg, dict) else None

    agg_cfg = cfg.get("agg", {}) or {}
    expert_cfg = agg_cfg.get("expert", {}) if isinstance(agg_cfg, dict) else {}
    method = expert_cfg.get("method") if isinstance(expert_cfg, dict) else None

    if not method:
        method = csv_path.parent.name

    return (
        str(dataset) if dataset is not None else None,
        str(backbone) if backbone is not None else None,
        str(method),
    )


def read_test_acc(csv_path: Path) -> Tuple[List[int], List[float]]:
    rounds: List[int] = []
    test_acc: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        required = {"round_id", "test_acc"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} 缺少必要列：{sorted(missing)}；"
                f"当前列：{reader.fieldnames}"
            )

        for row in reader:
            round_text = str(row.get("round_id", "")).strip()
            acc_text = str(row.get("test_acc", "")).strip()

            if not round_text or not acc_text:
                continue

            rounds.append(int(float(round_text)))
            test_acc.append(float(acc_text))

    if not rounds:
        raise ValueError(f"{csv_path} 中没有有效的 round_id/test_acc 数据。")

    pairs = sorted(zip(rounds, test_acc), key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs]


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError(f"window 必须大于 0，当前值：{window}")

    result: List[float] = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value

        if index >= window:
            running_sum -= values[index - window]

        current_count = min(index + 1, window)
        result.append(running_sum / current_count)

    return result


def friendly_method_name(method: str) -> str:
    name = method.lower().strip()

    if name == "uniform" or "uniform" in name:
        return "Uniform"
    if "fisher" in name or "kfac" in name:
        return "Fisher/K-FAC"
    if "fedmoe_da" in name or "fed_moe_da" in name or "fedmoeda" in name:
        return "FedMoE-DA"
    if "somfed" in name:
        return "SOMFed"
    if "fed_moe" in name or "fedmoe" in name:
        return "Fed-MoE"

    return method


def method_order(method: str) -> Tuple[int, str]:
    label = friendly_method_name(method)
    preferred = {
        "Uniform": 0,
        "Fisher/K-FAC": 1,
        "Fed-MoE": 2,
        "FedMoE-DA": 3,
        "SOMFed": 4,
    }
    return preferred.get(label, 100), label.lower()


def choose_dataset_backbone(
    runs: List[Dict[str, object]],
    dataset_arg: str | None,
    backbone_arg: str | None,
) -> Tuple[str, str]:
    known_pairs = {
        (str(run["dataset"]), str(run["backbone"]))
        for run in runs
        if run["dataset"] is not None and run["backbone"] is not None
    }

    if dataset_arg is not None and backbone_arg is not None:
        return dataset_arg, backbone_arg

    if dataset_arg is not None:
        backbones = sorted({b for d, b in known_pairs if d == dataset_arg})
        if len(backbones) == 1:
            return dataset_arg, backbones[0]
        raise ValueError(
            f"dataset={dataset_arg!r} 对应的 backbone 无法唯一确定：{backbones}。"
            "请同时传 --backbone。"
        )

    if backbone_arg is not None:
        datasets = sorted({d for d, b in known_pairs if b == backbone_arg})
        if len(datasets) == 1:
            return datasets[0], backbone_arg
        raise ValueError(
            f"backbone={backbone_arg!r} 对应的 dataset 无法唯一确定：{datasets}。"
            "请同时传 --dataset。"
        )

    if len(known_pairs) == 1:
        return next(iter(known_pairs))

    if len(known_pairs) == 0:
        raise ValueError(
            "没有从 config_used.yaml 中识别出 dataset/backbone。"
            "请显式传 --dataset 和 --backbone。"
        )

    choices = ", ".join(f"{d}+{b}" for d, b in sorted(known_pairs))
    raise ValueError(
        "输入目录中存在多个 dataset/backbone 组合："
        f"{choices}。请显式传 --dataset 和 --backbone。"
    )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if args.window <= 0:
        raise ValueError(f"--window 必须大于 0，当前值：{args.window}")

    csv_files = sorted(input_dir.rglob("results.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到任何 results.csv。")

    runs: List[Dict[str, object]] = []
    for csv_path in csv_files:
        dataset, backbone, method = load_run_metadata(csv_path)
        runs.append(
            {
                "csv": csv_path,
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
            }
        )

    dataset, backbone = choose_dataset_backbone(
        runs=runs,
        dataset_arg=args.dataset,
        backbone_arg=args.backbone,
    )

    matching_runs = [
        run
        for run in runs
        if run["dataset"] == dataset and run["backbone"] == backbone
    ]

    if not matching_runs:
        raise ValueError(
            f"没有找到 dataset={dataset!r}, backbone={backbone!r} 的 results.csv。"
        )

    # 同一算法如果因为重复实验出现 _v2/_v3，只取最后修改的 results.csv。
    latest_by_method: Dict[str, Path] = {}
    for run in matching_runs:
        method = str(run["method"])
        csv_path = Path(run["csv"])
        previous = latest_by_method.get(method)
        if previous is None or csv_path.stat().st_mtime > previous.stat().st_mtime:
            latest_by_method[method] = csv_path

    selected = sorted(latest_by_method.items(), key=lambda item: method_order(item[0]))

    if len(selected) < 5:
        print(
            f"[Warning] 当前只识别到 {len(selected)} 个算法结果；"
            "将对已找到的结果作图。"
        )
    elif len(selected) > 5:
        print(
            f"[Warning] 当前识别到 {len(selected)} 个算法结果；"
            "图中会全部绘制。"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    print()
    print(f"Dataset : {dataset}")
    print(f"Backbone: {backbone}")
    print(f"Window  : {args.window}")
    print("Results :")

    for method, csv_path in selected:
        rounds, test_acc = read_test_acc(csv_path)
        smoothed = moving_average(test_acc, args.window)
        label = friendly_method_name(method)

        ax.plot(rounds, smoothed, linewidth=2.0, label=label)

        print(
            f"  {label:<14} "
            f"rounds={len(rounds):>3} "
            f"final={test_acc[-1]:>6.2f}% "
            f"best={max(test_acc):>6.2f}% "
            f"<- {csv_path}"
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        f"{dataset} | {backbone} | Test Accuracy "
        f"(Moving Average, window={args.window})"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset}_{backbone}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
