from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "outputs" / "问题三" / "problem3_trajectory_10hz_kalman.csv"
OUTPUT_PATH = ROOT / "outputs" / "问题三" / "problem3_trajectory_10hz_kalman_3d_scatter.png"

df = pd.read_csv(DATA_PATH)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["x_method1_interp_m"],
    df["y_method1_interp_m"],
    df["time_s"],
    c="#E53935",
    s=2,
    alpha=0.5,
    label="方式1",
)

ax.scatter(
    df["x_method2_filter_input_interp_m"],
    df["y_method2_filter_input_interp_m"],
    df["time_s"],
    c="#1E88E5",
    s=2,
    alpha=0.5,
    label="方式2(对齐后)",
)

ax.scatter(
    df["x_kalman_m"],
    df["y_kalman_m"],
    df["time_s"],
    c="#43A047",
    s=4,
    alpha=0.85,
    label="卡尔曼融合",
)

ax.set_xlabel("X坐标 (m)")
ax.set_ylabel("Y坐标 (m)")
ax.set_zlabel("时间 (s)")
ax.set_title("问题3 10Hz轨迹三维散点图")
ax.legend(loc="upper right", markerscale=3)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"已保存：{OUTPUT_PATH}")
