from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "outputs" / "问题一" / "problem1_trajectory_10hz_fitted_average.csv"
OUTPUT_PATH = ROOT / "outputs" / "问题一" / "problem1_trajectory_10hz_fitted_average_3d.png"


df = pd.read_csv(DATA_PATH)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    df["x_average_m"],
    df["y_average_m"],
    df["time_s"],
    c=df["time_rel_s"],
    cmap="viridis",
    s=3,
    alpha=0.75,
)

ax.set_xlabel("X坐标 (m)")
ax.set_ylabel("Y坐标 (m)")
ax.set_zlabel("时间 (s)")
ax.set_title("问题1 10Hz 平均轨迹三维图")

cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
cbar.set_label("相对时间 (s)")

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"已保存：{OUTPUT_PATH}")
