import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

# 读取预处理后、未插值的数据文件
preprocess_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'preprocessing' / 'problem1'
df1 = pd.read_csv(preprocess_dir / 'problem1_method1_standardized.csv')
df2 = pd.read_csv(preprocess_dir / 'problem1_method2_standardized.csv')

# 创建三维图
fig = plt.figure(figsize=(12, 5))

# ---- 方式1 （4Hz） ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df1['x_m'], df1['y_m'], df1['time_s'],
            c=df1['time_s'], cmap='viridis', s=2, alpha=0.6)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('时间 (s)')
ax1.set_title('方式1 (4Hz) 三维轨迹')

# ---- 方式2 （5Hz） ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df2['x_m'], df2['y_m'], df2['time_s'],
            c=df2['time_s'], cmap='plasma', s=2, alpha=0.6)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('时间 (s)')
ax2.set_title('方式2 (5Hz) 三维轨迹')

plt.tight_layout()
plt.show()
