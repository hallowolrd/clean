#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


# =========================
# 论文绘图样式
# =========================

# MATLAB 规范：宽 14.2 cm，高 8.0 cm
FIG_WIDTH = 14.2 / 2.54
FIG_HEIGHT = 8.0 / 2.54

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "font.size": 15,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": True,
    }
)


# =========================
# Server Steps 配置
#
# 这 5 条曲线对应同一个 TKFAC 算法，
# 仅服务器迭代步数不同。
# =========================
SERVER_STEP_ORDER = [
    10,
    30,
    50,
    100,
    200,
]

SERVER_STEP_NAMES = {
    10: "Steps=10",
    30: "Steps=30",
    50: "Steps=50",
    100: "Steps=100",
    200: "Steps=200",
}

SERVER_STEP_STYLES = {
    10: {
        "color": "#1689D8",
        "linestyle": (0, (5.0, 1.3, 1.6, 1.3)),
        "linewidth": 1.8,
    },
    30: {
        "color": "#A500A5",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    50: {
        "color": "#19C4C7",
        "linestyle": (0, (1.0, 1.15)),
        "linewidth": 1.8,
    },
    100: {
        "color": "#F4BE00",
        "linestyle": (0, (5.0, 1.3, 1.4, 1.3)),
        "linewidth": 1.8,
    },
    200: {
        "color": "#8A8A8A",
        "linestyle": (0, (3.2, 1.8)),
        "linewidth": 1.8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TKFAC test_acc curves under different server_steps."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "总实验目录，例如 "
            "outputs_tkfac_server_steps/cifar10_resnet_cifar"
        ),
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
        help="可选。图片保存目录；默认保存到项目根目录 pictures。",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="可选。只绘制到指定通信轮数，例如 40 或 50；默认绘制全部轮数。",
    )
    return parser.parse_args()


def load_run_metadata(
    csv_path: Path,
) -> Tuple[str | None, str | None, int | None]:
    config_path = csv_path.parent / "config_used.yaml"

    if not config_path.is_file():
        return None, None, None

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset = cfg.get("dataset")

    model_cfg = cfg.get("model_cfg", {}) or {}
    backbone = (
        model_cfg.get("backbone")
        if isinstance(model_cfg, dict)
        else None
    )

    tkfac_cfg = cfg.get("tkfac", {}) or {}
    server_steps = (
        tkfac_cfg.get("server_steps")
        if isinstance(tkfac_cfg, dict)
        else None
    )

    return (
        str(dataset) if dataset is not None else None,
        str(backbone) if backbone is not None else None,
        int(server_steps) if server_steps is not None else None,
    )


def read_test_acc(
    csv_path: Path,
) -> Tuple[List[int], List[float]]:
    rounds: List[int] = []
    test_acc: List[float] = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
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
            round_text = str(
                row.get("round_id", "")
            ).strip()
            acc_text = str(
                row.get("test_acc", "")
            ).strip()

            if not round_text or not acc_text:
                continue

            rounds.append(
                int(float(round_text))
            )
            test_acc.append(
                float(acc_text)
            )

    if not rounds:
        raise ValueError(
            f"{csv_path} 中没有有效的 round_id/test_acc 数据。"
        )

    pairs = sorted(
        zip(rounds, test_acc),
        key=lambda item: item[0],
    )

    return (
        [item[0] for item in pairs],
        [item[1] for item in pairs],
    )


def moving_average(
    values: List[float],
    window: int,
) -> List[float]:
    if window <= 0:
        raise ValueError(
            f"window 必须大于 0，当前值：{window}"
        )

    result: List[float] = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value

        if index >= window:
            running_sum -= values[index - window]

        current_count = min(
            index + 1,
            window,
        )
        result.append(
            running_sum / current_count
        )

    return result


def choose_dataset_backbone(
    runs: List[Dict[str, object]],
    dataset_arg: str | None,
    backbone_arg: str | None,
) -> Tuple[str, str]:
    known_pairs = {
        (
            str(run["dataset"]),
            str(run["backbone"]),
        )
        for run in runs
        if run["dataset"] is not None
        and run["backbone"] is not None
    }

    if (
        dataset_arg is not None
        and backbone_arg is not None
    ):
        return dataset_arg, backbone_arg

    if dataset_arg is not None:
        backbones = sorted(
            {
                backbone
                for dataset, backbone
                in known_pairs
                if dataset == dataset_arg
            }
        )

        if len(backbones) == 1:
            return dataset_arg, backbones[0]

        raise ValueError(
            f"dataset={dataset_arg!r} 对应的 backbone "
            f"无法唯一确定：{backbones}。"
            "请同时传 --backbone。"
        )

    if backbone_arg is not None:
        datasets = sorted(
            {
                dataset
                for dataset, backbone
                in known_pairs
                if backbone == backbone_arg
            }
        )

        if len(datasets) == 1:
            return datasets[0], backbone_arg

        raise ValueError(
            f"backbone={backbone_arg!r} 对应的 dataset "
            f"无法唯一确定：{datasets}。"
            "请同时传 --dataset。"
        )

    if len(known_pairs) == 1:
        return next(iter(known_pairs))

    if len(known_pairs) == 0:
        raise ValueError(
            "没有从 config_used.yaml 中识别出 "
            "dataset/backbone。"
            "请显式传 --dataset 和 --backbone。"
        )

    choices = ", ".join(
        f"{dataset}+{backbone}"
        for dataset, backbone
        in sorted(known_pairs)
    )

    raise ValueError(
        "输入目录中存在多个 dataset/backbone 组合："
        f"{choices}。"
        "请显式传 --dataset 和 --backbone。"
    )


def main() -> int:
    args = parse_args()

    input_dir = (
        args.input_dir
        .expanduser()
        .resolve()
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"输入目录不存在：{input_dir}"
        )

    if args.window <= 0:
        raise ValueError(
            f"--window 必须大于 0，当前值：{args.window}"
        )

    if (
        args.max_round is not None
        and args.max_round <= 0
    ):
        raise ValueError(
            f"--max-round 必须大于 0，"
            f"当前值：{args.max_round}"
        )

    csv_files = sorted(
        input_dir.rglob("results.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"在 {input_dir} 下没有找到任何 results.csv。"
        )

    runs: List[Dict[str, object]] = []

    for csv_path in csv_files:
        (
            dataset,
            backbone,
            server_steps,
        ) = load_run_metadata(csv_path)

        runs.append(
            {
                "csv": csv_path,
                "dataset": dataset,
                "backbone": backbone,
                "server_steps": server_steps,
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
        if run["dataset"] == dataset
        and run["backbone"] == backbone
        and run["server_steps"] is not None
    ]

    if not matching_runs:
        raise ValueError(
            f"没有找到 dataset={dataset!r}, "
            f"backbone={backbone!r} 且包含 "
            "tkfac.server_steps 的 results.csv。"
        )

    latest_by_steps: Dict[int, Path] = {}

    for run in matching_runs:
        server_steps = int(
            run["server_steps"]
        )
        csv_path = Path(
            run["csv"]
        )

        previous = latest_by_steps.get(
            server_steps
        )

        if (
            previous is None
            or csv_path.stat().st_mtime
            > previous.stat().st_mtime
        ):
            latest_by_steps[
                server_steps
            ] = csv_path

    selected_steps = [
        step
        for step in SERVER_STEP_ORDER
        if step in latest_by_steps
    ]

    extra_steps = sorted(
        step
        for step in latest_by_steps
        if step not in SERVER_STEP_ORDER
    )

    selected_steps.extend(
        extra_steps
    )

    if len(selected_steps) < len(SERVER_STEP_ORDER):
        missing_steps = [
            step
            for step in SERVER_STEP_ORDER
            if step not in latest_by_steps
        ]
        print(
            f"[Warning] 缺少 server_steps="
            f"{missing_steps} 的实验结果；"
            "将对已找到的结果作图。"
        )

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT,
        )
    )

    print()
    print(f"Dataset : {dataset}")
    print(f"Backbone: {backbone}")
    print(f"Window  : {args.window}")
    print("Results :")

    all_smoothed: List[float] = []
    max_round = 0

    for index, server_steps in enumerate(
        selected_steps
    ):
        csv_path = latest_by_steps[
            server_steps
        ]

        rounds, test_acc = read_test_acc(
            csv_path
        )

        if args.max_round is not None:
            filtered = [
                (
                    round_id,
                    acc,
                )
                for round_id, acc
                in zip(
                    rounds,
                    test_acc,
                )
                if round_id <= args.max_round
            ]

            if not filtered:
                raise ValueError(
                    f"{csv_path} 在 round <= "
                    f"{args.max_round} 范围内没有数据。"
                )

            rounds = [
                item[0]
                for item in filtered
            ]
            test_acc = [
                item[1]
                for item in filtered
            ]

        smoothed = moving_average(
            test_acc,
            args.window,
        )

        if server_steps in SERVER_STEP_STYLES:
            style = SERVER_STEP_STYLES[
                server_steps
            ]
        else:
            fallback_styles = list(
                SERVER_STEP_STYLES.values()
            )
            style = fallback_styles[
                index % len(fallback_styles)
            ]

        label = SERVER_STEP_NAMES.get(
            server_steps,
            f"Steps={server_steps}",
        )

        ax.plot(
            rounds,
            smoothed,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=label,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=3,
        )

        all_smoothed.extend(
            smoothed
        )

        if rounds:
            max_round = max(
                max_round,
                max(rounds),
            )

        print(
            f"  {label:<12} "
            f"rounds={len(rounds):>3} "
            f"final={test_acc[-1]:>6.2f}% "
            f"best={max(test_acc):>6.2f}% "
            f"<- {csv_path}"
        )

    ax.set_xlabel(
        "Training Round",
        fontsize=21,
        labelpad=6,
    )

    ax.set_ylabel(
        "Test Accuracy (%)",
        fontsize=21,
        labelpad=6,
    )

    x_ticks = [
        max_round * i / 5
        for i in range(6)
    ]

    x_tick_labels = [
        (
            f"{int(round(tick))}"
            if abs(
                tick - round(tick)
            ) < 1e-9
            else f"{tick:g}"
        )
        for tick in x_ticks
    ]

    ax.set_xticks(
        x_ticks
    )

    ax.set_xticklabels(
        x_tick_labels
    )

    ax.set_xlim(
        0,
        max_round,
    )

    if not all_smoothed:
        raise ValueError(
            "没有可用于绘图的 test_acc 数据。"
        )

    y_min = min(
        all_smoothed
    )

    y_max = max(
        all_smoothed
    )

    # GAPSL 风格：
    # 纵轴下边界从较整齐的主刻度开始。
    y_lower = math.floor(
        y_min / 5
    ) * 5

    # 顶部只保留少量余量，
    # 并把上边框写成实际的纵坐标值。
    y_upper = math.ceil(
        y_max + 1.0
    )

    if y_upper - y_max < 1.0:
        y_upper += 1

    # 内部刻度尽量控制在约 3 个区间，
    # 最后一个刻度单独用 y_upper。
    raw_step = (
        y_upper - y_lower
    ) / 3

    nice_steps = [
        5,
        10,
        15,
        20,
        25,
        30,
    ]

    y_step = min(
        nice_steps,
        key=lambda step: abs(
            step - raw_step
        ),
    )

    y_ticks = list(
        range(
            int(y_lower),
            int(y_upper),
            int(y_step),
        )
    )

    if not y_ticks:
        y_ticks = [
            int(y_lower)
        ]

    if (
        y_ticks
        and (
            y_upper - y_ticks[-1]
        ) < 0.5 * y_step
    ):
        y_ticks.pop()

    if (
        not y_ticks
        or y_ticks[-1]
        != int(y_upper)
    ):
        y_ticks.append(
            int(y_upper)
        )

    ax.set_ylim(
        y_lower,
        y_upper,
    )

    ax.set_yticks(
        y_ticks
    )

    ax.grid(
        which="major",
        axis="both",
        visible=True,
        linestyle="-",
        linewidth=1.0,
        color="#D9D9D9",
        alpha=0.9,
        zorder=0,
    )

    ax.set_axisbelow(
        True
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            2.0
        )
        spine.set_color(
            "black"
        )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.8,
        length=3.5,
        top=True,
        right=True,
        pad=4,
    )

    legend = ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        borderpad=0.22,
        labelspacing=0.18,
        columnspacing=0.8,
        handlelength=1.65,
        handletextpad=0.25,
        fontsize=14,
    )

    legend.get_frame().set_linewidth(
        1.6
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / "pictures"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / f"{dataset}_{backbone}_server_steps.png"
    )

    pdf_path = (
        output_dir
        / f"{dataset}_{backbone}_server_steps.pdf"
    )

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"Saved PNG: {png_path}"
    )
    print(
        f"Saved PDF: {pdf_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
