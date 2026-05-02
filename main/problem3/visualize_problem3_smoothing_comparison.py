from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "数据以及可视化" / "附件3.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "问题三"

SMOOTH_SECONDS = 7.5


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[:, :3].copy()
    out.columns = ["time_s", "x_m", "y_m"]
    for col in ["time_s", "x_m", "y_m"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["time_s", "x_m", "y_m"]].isna().any().any():
        raise ValueError("Input contains missing or non-numeric values.")
    out = out.sort_values("time_s")
    if out["time_s"].duplicated().any():
        out = out.groupby("time_s", as_index=False)[["x_m", "y_m"]].mean()
    return out


def savgol_window(t: np.ndarray) -> int:
    dt = float(np.median(np.diff(t)))
    window = max(5, int(round(SMOOTH_SECONDS / dt)))
    if window % 2 == 0:
        window += 1
    window = min(window, len(t) - 1 if (len(t) - 1) % 2 == 1 else len(t) - 2)
    return max(window, 5)


def smooth_xy(df: pd.DataFrame) -> np.ndarray:
    t = df["time_s"].to_numpy(float)
    xy = df[["x_m", "y_m"]].to_numpy(float)
    return savgol_filter(xy, window_length=savgol_window(t), polyorder=3, axis=0, mode="interp")


def axis_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.vstack(arrays)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = max(1.0, 0.04 * (x_max - x_min))
    y_pad = max(1.0, 0.04 * (y_max - y_min))
    return (float(x_min - x_pad), float(x_max + x_pad)), (float(y_min - y_pad), float(y_max + y_pad))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df1 = standardize_frame(pd.read_excel(DATA_PATH, sheet_name=0))
    df2 = standardize_frame(pd.read_excel(DATA_PATH, sheet_name=1))

    xy1_raw = df1[["x_m", "y_m"]].to_numpy(float)
    xy2_raw = df2[["x_m", "y_m"]].to_numpy(float)
    xy1_smooth = smooth_xy(df1)
    xy2_smooth = smooth_xy(df2)

    xlim, ylim = axis_limits(xy1_raw, xy2_raw, xy1_smooth, xy2_smooth)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    plot_items = [
        (axes[0], "方式一：未平滑与平滑后散点对比", xy1_raw, xy1_smooth),
        (axes[1], "方式二：未平滑与平滑后散点对比", xy2_raw, xy2_smooth),
    ]

    for ax, title, raw, smoothed in plot_items:
        ax.scatter(raw[:, 0], raw[:, 1], s=8, c="#2563eb", alpha=0.32, label="未平滑散点")
        ax.scatter(smoothed[:, 0], smoothed[:, 1], s=8, c="#dc2626", alpha=0.72, label="平滑后散点")
        ax.set_title(title)
        ax.set_xlabel("X 坐标 (m)")
        ax.set_ylabel("Y 坐标 (m)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("问题三：未平滑数据与 Savitzky-Golay 平滑数据对比", fontsize=15)
    fig.tight_layout()

    output_path = OUT_DIR / "problem3_raw_vs_savgol_smoothing_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved smoothing comparison figure: {output_path}")


if __name__ == "__main__":
    main()
