from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_run_arg(text: str) -> tuple[str, Path]:
    """
    解析命令行输入的实验路径。

    支持两种写法：
        1. uniform=outputs/xxx/results.csv
        2. uniform=outputs/xxx

    如果传入的是目录，会自动读取目录下的 results.csv。
    """
    if "=" not in text:
        path = Path(text).expanduser()
        label = path.parent.name if path.name == "results.csv" else path.name
        return label, path

    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip()).expanduser()


def resolve_results_csv(path: Path) -> Path:
    """
    如果 path 是目录，则默认读取 path/results.csv。
    """
    if path.is_dir():
        path = path / "results.csv"

    if not path.exists():
        raise FileNotFoundError(f"找不到 results.csv：{path}")

    return path


def read_result_csv(path: Path) -> pd.DataFrame:
    """
    读取单个实验的 results.csv。

    当前项目的 results.csv 关键字段：
        round_id: 联邦训练轮次
        test_acc: 测试集准确率
    """
    path = resolve_results_csv(path)
    df = pd.read_csv(path)

    if "round_id" not in df.columns:
        raise ValueError(f"{path} 中缺少 round_id 列，当前列名：{list(df.columns)}")

    if "test_acc" not in df.columns:
        raise ValueError(f"{path} 中缺少 test_acc 列，当前列名：{list(df.columns)}")

    df = df.copy()

    # 防止 CSV 中混入字符串或百分号
    df["round_id"] = pd.to_numeric(df["round_id"], errors="coerce")
    df["test_acc"] = (
        df["test_acc"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")

    df = df.dropna(subset=["round_id", "test_acc"])
    df = df.sort_values("round_id")

    df["round_id"] = df["round_id"].astype(int)

    return df


def add_sliding_window(
    df: pd.DataFrame,
    window: int,
    center: bool,
) -> pd.DataFrame:
    """
    添加滑动窗口平滑后的 test_acc。

    window=5 表示用 5 轮做滑动平均。
    center=True 表示居中平滑，曲线不会明显向右延迟，更适合画论文图。
    center=False 表示只用当前轮及之前轮次，更严格但曲线会有一点延迟。
    """
    df = df.copy()

    if window <= 1:
        df["test_acc_smooth"] = df["test_acc"]
    else:
        df["test_acc_smooth"] = (
            df["test_acc"]
            .rolling(
                window=window,
                min_periods=1,
                center=center,
            )
            .mean()
        )

    return df


def plot_curves(
    runs: list[tuple[str, Path]],
    window: int,
    center: bool,
    show_raw: bool,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.2))

    for label, path in runs:
        df = read_result_csv(path)
        df = add_sliding_window(
            df=df,
            window=window,
            center=center,
        )

        # 原始曲线：浅一点，用来观察真实波动
        if show_raw:
            plt.plot(
                df["round_id"],
                df["test_acc"],
                linewidth=1.0,
                alpha=0.25,
                label=f"{label} raw",
            )

        # 平滑曲线：主曲线
        plt.plot(
            df["round_id"],
            df["test_acc_smooth"],
            linewidth=2.2,
            label=f"{label} smooth-w{window}",
        )

        best_acc = df["test_acc"].max()
        best_round = int(df.loc[df["test_acc"].idxmax(), "round_id"])
        final_acc = df["test_acc"].iloc[-1]

        print(
            f"{label}: "
            f"best_acc={best_acc:.2f}% @ round {best_round}, "
            f"final_acc={final_acc:.2f}%"
        )

    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n图已保存到：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot test_acc comparison curves from multiple FL results.csv files."
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "实验结果路径，格式：label=path。"
            "path 可以是 results.csv，也可以是包含 results.csv 的实验目录。"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="滑动窗口大小，例如 3、5、7。默认 5。",
    )

    parser.add_argument(
        "--no-center",
        action="store_true",
        help="关闭居中滑动平均，改成只使用当前轮及之前轮次。",
    )

    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="只画滑动平均曲线，不画原始 test_acc 曲线。",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Algorithm Comparison on Test Accuracy",
        help="图标题。",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="outputs/compare_test_acc.png",
        help="输出图片路径。",
    )

    args = parser.parse_args()

    runs = [parse_run_arg(item) for item in args.runs]

    plot_curves(
        runs=runs,
        window=args.window,
        center=not args.no_center,
        show_raw=not args.hide_raw,
        title=args.title,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()