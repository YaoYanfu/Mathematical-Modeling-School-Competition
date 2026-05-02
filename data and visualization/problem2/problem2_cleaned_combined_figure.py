from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "visualization" / "附件2.xlsx"
SOLVER_PATH = ROOT / "main" / "solve_problem2_kalman.py"
OUTPUT_PATH = ROOT / "outputs" / "problem2_cleaned_attachment2_combined.png"


spec = importlib.util.spec_from_file_location("solve_problem2_kalman", SOLVER_PATH)
solver = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = solver
spec.loader.exec_module(solver)

df1_raw = pd.read_excel(DATA_PATH, sheet_name="方式1(4Hz)")
df2_raw = pd.read_excel(DATA_PATH, sheet_name="方式2(5Hz)")
df1 = solver.standardize_frame(df1_raw, "方式1")
df2 = solver.standardize_frame(df2_raw, "方式2")

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
scatter1 = ax1.scatter(
    df1["x_m"],
    df1["y_m"],
    df1["time_s"],
    c=df1["time_s"],
    cmap="viridis",
    s=3,
    alpha=0.75,
)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("时间 (s)")
ax1.set_title("方式1 (4Hz) 清洗后三维轨迹")

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df1["x_m"], df1["y_m"], "b-", lw=1.2)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_title("方式1 (4Hz) 清洗后 X-Y 投影")
ax2.axis("equal")
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
scatter2 = ax3.scatter(
    df2["x_m"],
    df2["y_m"],
    df2["time_s"],
    c=df2["time_s"],
    cmap="plasma",
    s=3,
    alpha=0.75,
)
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_zlabel("时间 (s)")
ax3.set_title("方式2 (5Hz) 清洗后三维轨迹")

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(df2["x_m"], df2["y_m"], "r-", lw=1.2)
ax4.set_xlabel("X (m)")
ax4.set_ylabel("Y (m)")
ax4.set_title("方式2 (5Hz) 清洗后 X-Y 投影")
ax4.axis("equal")
ax4.grid(alpha=0.3)

fig.suptitle("附件2清洗后轨迹与二维投影", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"saved: {OUTPUT_PATH}")
print(solver.CLEANING_REPORT)
