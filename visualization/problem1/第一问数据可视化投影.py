import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

# 读取预处理后、未插值的数据文件
preprocess_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'preprocessing' / 'problem1'
df1 = pd.read_csv(preprocess_dir / 'problem1_method1_standardized.csv')
df2 = pd.read_csv(preprocess_dir / 'problem1_method2_standardized.csv')

# 提取数据
t1, x1, y1 = df1['time_s'], df1['x_m'], df1['y_m']
t2, x2, y2 = df2['time_s'], df2['x_m'], df2['y_m']

# 创建 2 行 3 列子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ---- 第一行：方式1 (4Hz) ----
# X-t 投影
axes[0, 0].scatter(t1, x1, s=2, c='blue', alpha=0.6)
axes[0, 0].set_xlabel('时间 (s)')
axes[0, 0].set_ylabel('X (m)')
axes[0, 0].set_title('方式1 (4Hz) X‑t')
axes[0, 0].grid(alpha=0.3)

# Y-t 投影
axes[0, 1].scatter(t1, y1, s=2, c='blue', alpha=0.6)
axes[0, 1].set_xlabel('时间 (s)')
axes[0, 1].set_ylabel('Y (m)')
axes[0, 1].set_title('方式1 (4Hz) Y‑t')
axes[0, 1].grid(alpha=0.3)

# X-Y 投影 (xOy 面)
axes[0, 2].scatter(x1, y1, s=2, c='blue', alpha=0.6)
axes[0, 2].set_xlabel('X (m)')
axes[0, 2].set_ylabel('Y (m)')
axes[0, 2].set_title('方式1 (4Hz) X‑Y (轨迹)')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].axis('equal')   # 保持纵横比一致

# ---- 第二行：方式2 (5Hz) ----
# X-t 投影
axes[1, 0].scatter(t2, x2, s=2, c='red', alpha=0.6)
axes[1, 0].set_xlabel('时间 (s)')
axes[1, 0].set_ylabel('X (m)')
axes[1, 0].set_title('方式2 (5Hz) X‑t')
axes[1, 0].grid(alpha=0.3)

# Y-t 投影
axes[1, 1].scatter(t2, y2, s=2, c='red', alpha=0.6)
axes[1, 1].set_xlabel('时间 (s)')
axes[1, 1].set_ylabel('Y (m)')
axes[1, 1].set_title('方式2 (5Hz) Y‑t')
axes[1, 1].grid(alpha=0.3)

# X-Y 投影 (xOy 面)
axes[1, 2].scatter(x2, y2, s=2, c='red', alpha=0.6)
axes[1, 2].set_xlabel('X (m)')
axes[1, 2].set_ylabel('Y (m)')
axes[1, 2].set_title('方式2 (5Hz) X‑Y (轨迹)')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axis('equal')

plt.tight_layout()
plt.show()
