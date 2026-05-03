from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "数据以及可视化" / "附件2.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "问题二"
FIG_PATH = OUT_DIR / "problem2_attachment2_aligned_four_panel.png"
ALIGNED_CSV_PATH = OUT_DIR / "problem2_attachment2_aligned_by_delta_bias.csv"

METHOD1_COLOR = "#d62728"
METHOD2_COLOR = "#1f77b4"
DELTA_T2_MINUS_T1_S = 50.474885
BIAS_ADD_TO_METHOD2_M = np.array([-3.475681, 1.835845], dtype=float)
POINT_SIZE = 7
POINT_ALPHA = 0.72


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[:, :3].copy()
    out.columns = ["time_s", "x_m", "y_m"]
    for col in ["time_s", "x_m", "y_m"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["time_s", "x_m", "y_m"]].isna().any().any():
        raise ValueError("附件二中存在缺失值或非数值。")
    out = out.sort_values("time_s")
    if out["time_s"].duplicated().any():
        out = out.groupby("time_s", as_index=False)[["x_m", "y_m"]].mean()
    if not np.all(np.diff(out["time_s"].to_numpy(float)) > 0):
        raise ValueError("附件二时间列不是严格递增。")
    return out


def load_alignment_values() -> tuple[float, np.ndarray]:
    return DELTA_T2_MINUS_T1_S, BIAS_ADD_TO_METHOD2_M.copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df1 = standardize_frame(pd.read_excel(DATA_PATH, sheet_name="方式1(4Hz)"))
    df2 = standardize_frame(pd.read_excel(DATA_PATH, sheet_name="方式2(5Hz)"))
    delta_t, bias = load_alignment_values()

    df1_plot = df1.copy()
    df1_plot["aligned_time_s"] = df1_plot["time_s"]
    df1_plot["x_aligned_m"] = df1_plot["x_m"]
    df1_plot["y_aligned_m"] = df1_plot["y_m"]
    df1_plot["方式"] = "方式1"

    df2_plot = df2.copy()
    df2_plot["aligned_time_s"] = df2_plot["time_s"] - delta_t
    df2_plot["x_aligned_m"] = df2_plot["x_m"] + bias[0]
    df2_plot["y_aligned_m"] = df2_plot["y_m"] + bias[1]
    df2_plot["方式"] = "方式2对齐后"

    overlap_start = max(float(df1_plot["aligned_time_s"].min()), float(df2_plot["aligned_time_s"].min()))
    overlap_end = min(float(df1_plot["aligned_time_s"].max()), float(df2_plot["aligned_time_s"].max()))

    df1_plot = df1_plot[(df1_plot["aligned_time_s"] >= overlap_start) & (df1_plot["aligned_time_s"] <= overlap_end)]
    df2_plot = df2_plot[(df2_plot["aligned_time_s"] >= overlap_start) & (df2_plot["aligned_time_s"] <= overlap_end)]

    aligned = pd.concat(
        [
            df1_plot[["方式", "time_s", "aligned_time_s", "x_aligned_m", "y_aligned_m"]],
            df2_plot[["方式", "time_s", "aligned_time_s", "x_aligned_m", "y_aligned_m"]],
        ],
        ignore_index=True,
    )
    aligned.to_csv(ALIGNED_CSV_PATH, index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xt = fig.add_subplot(2, 2, 3)
    ax_yt = fig.add_subplot(2, 2, 4)

    for frame, color, label in [
        (df1_plot, METHOD1_COLOR, "方式1"),
        (df2_plot, METHOD2_COLOR, "方式2对齐后"),
    ]:
        t = frame["aligned_time_s"].to_numpy(float)
        x = frame["x_aligned_m"].to_numpy(float)
        y = frame["y_aligned_m"].to_numpy(float)

        ax3d.scatter(x, y, t, color=color, s=POINT_SIZE, alpha=POINT_ALPHA, label=label)
        ax_xy.scatter(x, y, color=color, s=POINT_SIZE, alpha=POINT_ALPHA, label=label)
        ax_xt.scatter(t, x, color=color, s=POINT_SIZE, alpha=POINT_ALPHA, label=label)
        ax_yt.scatter(t, y, color=color, s=POINT_SIZE, alpha=POINT_ALPHA, label=label)

    ax3d.set_title("普通三维")
    ax3d.set_xlabel("X坐标 (m)")
    ax3d.set_ylabel("Y坐标 (m)")
    ax3d.set_zlabel("对齐时间 (s)")
    ax3d.legend(loc="upper right")

    ax_xy.set_title("X-Y投影")
    ax_xy.set_xlabel("X坐标 (m)")
    ax_xy.set_ylabel("Y坐标 (m)")
    ax_xy.axis("equal")
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc="upper right")

    ax_xt.set_title("X-时间投影")
    ax_xt.set_xlabel("对齐时间 (s)")
    ax_xt.set_ylabel("X坐标 (m)")
    ax_xt.grid(alpha=0.3)
    ax_xt.legend(loc="upper right")

    ax_yt.set_title("Y-时间投影")
    ax_yt.set_xlabel("对齐时间 (s)")
    ax_yt.set_ylabel("Y坐标 (m)")
    ax_yt.grid(alpha=0.3)
    ax_yt.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存四区域可视化图：{FIG_PATH}")
    print(f"已保存对齐后数据：{ALIGNED_CSV_PATH}")
    print(f"使用参数：Δt = {delta_t:.6f} s，b = ({bias[0]:.6f}, {bias[1]:.6f}) m")


if __name__ == "__main__":
    main()
