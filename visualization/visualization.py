import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

# 读取 Excel 文件，两个 sheet 分别对应两种方式
file_path = '附件2.xlsx'
df1 = pd.read_excel(file_path, sheet_name='方式1(4Hz)')
df2 = pd.read_excel(file_path, sheet_name='方式2(5Hz)')

# 创建三维图
fig = plt.figure(figsize=(12, 5))

# ---- 方式1 （4Hz） ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df1['X坐标(m)'], df1['Y坐标(m)'], df1['时间(s)'],
            c=df1['时间(s)'], cmap='viridis', s=2, alpha=0.6)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('时间 (s)')
ax1.set_title('方式1 (4Hz) 三维轨迹')

# ---- 方式2 （5Hz） ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df2['X坐标(m)'], df2['Y坐标(m)'], df2['时间(s)'],
            c=df2['时间(s)'], cmap='plasma', s=2, alpha=0.6)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('时间 (s)')
ax2.set_title('方式2 (5Hz) 三维轨迹')

plt.tight_layout()
plt.show()