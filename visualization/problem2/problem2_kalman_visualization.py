from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "problem2_trajectory_10hz_kalman.csv"
OUT_3D = ROOT / "outputs" / "problem2_trajectory_10hz_kalman_3d.png"
OUT_PROJECTION = ROOT / "outputs" / "problem2_trajectory_10hz_kalman_projection.png"


df = pd.read_csv(DATA_PATH)

# 三维轨迹图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    df["x_kalman_m"],
    df["y_kalman_m"],
    df["time_s"],
    c=df["time_s"],
    cmap="viridis",
    s=3,
    alpha=0.75,
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("时间 (s)")
ax.set_title("问题2 卡尔曼滤波融合10Hz轨迹三维图")
cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
cbar.set_label("时间 (s)")
plt.tight_layout()
fig.savefig(OUT_3D, dpi=300, bbox_inches="tight")

# 二维投影与分量图
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(df["x_kalman_m"], df["y_kalman_m"], "b-", lw=1.5)
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].set_title("X-Y 投影")
axes[0].axis("equal")
axes[0].grid(alpha=0.3)

axes[1].plot(df["time_s"], df["x_kalman_m"], "r-", lw=1.2)
axes[1].set_xlabel("时间 (s)")
axes[1].set_ylabel("X (m)")
axes[1].set_title("X-t 投影")
axes[1].grid(alpha=0.3)

axes[2].plot(df["time_s"], df["y_kalman_m"], "g-", lw=1.2)
axes[2].set_xlabel("时间 (s)")
axes[2].set_ylabel("Y (m)")
axes[2].set_title("Y-t 投影")
axes[2].grid(alpha=0.3)

plt.tight_layout()
fig2.savefig(OUT_PROJECTION, dpi=300, bbox_inches="tight")
plt.show()

print(f"saved: {OUT_3D}")
print(f"saved: {OUT_PROJECTION}")
