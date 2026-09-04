#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ============================================================
# 只需要修改这里：横轴、算法名、数值、颜色
# ============================================================

ALPHAS = [0.1, 0.2, 0.3]

METHODS = [
    {
        "name": "FedAvg-MoE",
        "values": [62.93, 78.15, 82.71],
        "color": "#A000A0",
    },
    {
        "name": "SOMFed",
        "values": [60.94, 77.29, 82.51],
        "color": "#8A8A8A",
    },
    {
        "name": "Fed-MoE",
        "values": [58.26, 78.49, 84.06],
        "color": "#1689D8",
    },
    {
        "name": "FedMoE-DA",
        "values": [55.85, 75.60, 82.32],
        "color": "#38C7CB",
    },
    {
        "name": "TKFAC",
        "values": [69.53, 80.39, 84.89],
        "color": "#FFD200",
    },
]


# ============================================================
# 输出设置
# ============================================================

OUTPUT_DIR = Path("./paper_bar")
OUTPUT_NAME = "dirichlet_best_accuracy"


# ============================================================
# 图像尺寸与字体
# ============================================================

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
        "mathtext.fontset": "stix",

        "font.size": 13,
        "axes.labelsize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 13,

        "axes.linewidth": 1.8,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,

        "legend.frameon": True,
    }
)


def validate_data() -> None:
    if not ALPHAS:
        raise ValueError("ALPHAS 不能为空。")

    if not METHODS:
        raise ValueError("METHODS 不能为空。")

    names = set()

    for method in METHODS:
        name = method["name"]
        values = method["values"]

        if name in names:
            raise ValueError(f"存在重复算法名：{name}")

        names.add(name)

        if len(values) != len(ALPHAS):
            raise ValueError(
                f"{name} 的数据数量为 {len(values)}，"
                f"但 ALPHAS 数量为 {len(ALPHAS)}。"
            )


def main() -> int:
    validate_data()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = np.arange(
        len(ALPHAS),
        dtype=float,
    )

    # 参考图中 5 根柱子较紧凑，
    # 但柱子之间仍保留细小空隙。
    bar_width = 0.115
    bar_step = 0.135

    offsets = (
        np.arange(
            len(METHODS),
            dtype=float,
        )
        - (len(METHODS) - 1) / 2.0
    ) * bar_step

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    # ========================================================
    # 分组柱状图
    # ========================================================

    for index, method in enumerate(METHODS):
        ax.bar(
            x + offsets[index],
            method["values"],
            width=bar_width,
            color=method["color"],
            edgecolor="black",
            linewidth=1.15,
            zorder=3,
        )

    # ========================================================
    # 坐标轴
    # ========================================================

    ax.set_xlabel(
        r"Dirichlet $\alpha$",
        labelpad=2,
    )

    ax.set_ylabel(
        "Best Accuracy (%)",
        labelpad=5,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{alpha:.1f}"
            for alpha in ALPHAS
        ]
    )

    # 与参考图一致
    ax.set_ylim(50, 98)

    ax.set_yticks(
        [50, 60, 70, 80, 90]
    )

    ax.set_xlim(
        -0.48,
        len(ALPHAS) - 1 + 0.48,
    )

    # ========================================================
    # 背景网格
    # 只有横向灰色网格
    # ========================================================

    ax.grid(
        which="major",
        axis="y",
        visible=True,
        linestyle="-",
        linewidth=1.0,
        color="#C8C8C8",
        alpha=0.70,
        zorder=0,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # 黑色坐标轴边框
    # ========================================================

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.6,
        length=4.0,
        top=True,
        right=True,
        pad=5,
    )

    # ========================================================
    # 图例
    # ========================================================

    legend_handles = [
        Patch(
            facecolor=method["color"],
            edgecolor="black",
            linewidth=1.0,
            label=method["name"],
        )
        for method in METHODS
    ]

    legend_labels = [
        method["name"]
        for method in METHODS
    ]

    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,

        loc="upper right",
        ncol=3,

        frameon=True,
        fancybox=False,
        framealpha=1.0,

        edgecolor="black",
        facecolor="white",

        borderpad=0.12,
        labelspacing=0.08,
        columnspacing=0.45,

        handlelength=0.50,
        handleheight=0.95,
        handletextpad=0.12,

        fontsize=11.5,
    )

    legend.get_frame().set_linewidth(1.2)

    # ========================================================
    # 保存
    # ========================================================

    fig.tight_layout(
        pad=0.35
    )

    png_path = (
        OUTPUT_DIR
        / f"{OUTPUT_NAME}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{OUTPUT_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(
        f"Saved PNG: "
        f"{png_path.resolve()}"
    )

    print(
        f"Saved PDF: "
        f"{pdf_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
