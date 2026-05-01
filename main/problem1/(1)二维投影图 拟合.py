import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import hilbert
from pathlib import Path

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

ALIGN_DT = 0.1
MIN_OVERLAP_RATIO = 0.70

# ------------------------------
# 1. 读取数据
# ------------------------------
preprocess_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'preprocessing' / 'problem1'
df1 = pd.read_csv(preprocess_dir / 'problem1_method1_standardized.csv')
df2 = pd.read_csv(preprocess_dir / 'problem1_method2_standardized.csv')

t1 = df1['time_s'].values
x1 = df1['x_m'].values
y1 = df1['y_m'].values

t2 = df2['time_s'].values
x2 = df2['x_m'].values
y2 = df2['y_m'].values

# ------------------------------
# 2. 主方法：三次样条插值
# ------------------------------
def build_spline(t, signal):
    return CubicSpline(t, signal, bc_type='natural')

def spline_fit_curve(t, spline):
    t_fit = np.linspace(t.min(), t.max(), 2000)
    fit_curve = spline(t_fit)
    return t_fit, fit_curve

# ------------------------------
# 3. 辅助方法：线性调幅余弦拟合
# ------------------------------
def amp_model(t, c, a, b, omega, phi):
    """线性调幅余弦：c + (a + b*t) * cos(omega*t + phi)"""
    return c + (a + b * t) * np.cos(omega * t + phi)

def fit_amp_signal(t, signal):
    c0 = np.median(signal)
    detrended = signal - c0
    envelope = np.abs(hilbert(detrended))
    b0, a0 = np.polyfit(t, envelope, 1)

    sign = detrended[:-1] * detrended[1:] < 0
    zeros = t[:-1][sign]
    if len(zeros) >= 2:
        periods = np.diff(zeros)
        t_mean = 2 * np.mean(periods)
        omega0 = 2 * np.pi / t_mean
    else:
        omega0 = 0.02

    p0 = [c0, a0, b0, omega0, 0.0]
    bounds = (
        [-10, -10, -0.5, 0.001, -2*np.pi],
        [10, 10, 0.5, 0.2, 2*np.pi]
    )
    try:
        popt, _ = curve_fit(
            amp_model, t, signal, p0=p0, bounds=bounds,
            maxfev=10000, method='trf'
        )
    except Exception as e:
        print(f"线性调幅拟合失败: {e}")
        popt = np.array(p0, dtype=float)
    return popt

def amp_fit_curve(t, params):
    t_fit = np.linspace(t.min(), t.max(), 2000)
    fit_curve = amp_model(t_fit, *params)
    return t_fit, fit_curve

# ------------------------------
# 4. 时间配准模型
# ------------------------------
def overlap_bounds(delta_t):
    overlap_start = max(t1.min(), t2.min() - delta_t)
    overlap_end = min(t1.max(), t2.max() - delta_t)
    return overlap_start, overlap_end

def alignment_objective(delta_t, f1x, f1y, f2x, f2y):
    """
    对应公式：
    mean((x1(t_i)-x2(t_i+Δt))^2 + (y1(t_i)-y2(t_i+Δt))^2)
    """
    overlap_start, overlap_end = overlap_bounds(delta_t)
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min(t1.max() - t1.min(), t2.max() - t2.min()))
    if overlap_end - overlap_start < min_overlap:
        return np.inf

    ti = np.arange(overlap_start, overlap_end + 1e-9, ALIGN_DT)
    err = (
        (f1x(ti) - f2x(ti + delta_t)) ** 2
        + (f1y(ti) - f2y(ti + delta_t)) ** 2
    )
    return np.mean(err)

def align_time(f1x, f1y, f2x, f2y):
    min_duration = min(t1.max() - t1.min(), t2.max() - t2.min())
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min_duration)
    delta_min = t2.min() - t1.max() + min_overlap
    delta_max = t2.max() - t1.min() - min_overlap

    search_grid = np.linspace(delta_min, delta_max, 1000)
    scores = np.array([
        alignment_objective(delta, f1x, f1y, f2x, f2y)
        for delta in search_grid
    ])
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("没有满足最小重叠比例的时间偏移搜索区间")

    best_delta = search_grid[np.argmin(scores)]
    grid_step = search_grid[1] - search_grid[0]
    result = minimize_scalar(
        lambda delta: alignment_objective(delta, f1x, f1y, f2x, f2y),
        bounds=(max(delta_min, best_delta - 2 * grid_step), min(delta_max, best_delta + 2 * grid_step)),
        method='bounded',
        options={'xatol': 1e-8},
    )
    delta_t = result.x if result.success else best_delta

    overlap_start, overlap_end = overlap_bounds(delta_t)
    n_points = len(np.arange(overlap_start, overlap_end + 1e-9, ALIGN_DT))
    overlap_ratio = (overlap_end - overlap_start) / min_duration
    mse = alignment_objective(delta_t, f1x, f1y, f2x, f2y)
    return delta_t, mse, overlap_start, overlap_end, overlap_ratio, n_points

def alignment_series(delta_t, f1x, f1y, f2x, f2y):
    overlap_start, overlap_end = overlap_bounds(delta_t)
    ti = np.arange(overlap_start, overlap_end + 1e-9, ALIGN_DT)
    x1_fit = f1x(ti)
    y1_fit = f1y(ti)
    x2_fit = f2x(ti + delta_t)
    y2_fit = f2y(ti + delta_t)
    error = np.sqrt((x1_fit - x2_fit) ** 2 + (y1_fit - y2_fit) ** 2)
    return ti, x1_fit, y1_fit, x2_fit, y2_fit, error

# ------------------------------
# 5. 构造主方法与辅助方法
# ------------------------------
spline1x = build_spline(t1, x1)
spline1y = build_spline(t1, y1)
spline2x = build_spline(t2, x2)
spline2y = build_spline(t2, y2)

tf1x, fx1 = spline_fit_curve(t1, spline1x)
tf1y, fy1 = spline_fit_curve(t1, spline1y)
tf2x, fx2 = spline_fit_curve(t2, spline2x)
tf2y, fy2 = spline_fit_curve(t2, spline2y)

params1x = fit_amp_signal(t1, x1)
params1y = fit_amp_signal(t1, y1)
params2x = fit_amp_signal(t2, x2)
params2y = fit_amp_signal(t2, y2)

tf1x_aux, fx1_aux = amp_fit_curve(t1, params1x)
tf1y_aux, fy1_aux = amp_fit_curve(t1, params1y)
tf2x_aux, fx2_aux = amp_fit_curve(t2, params2x)
tf2y_aux, fy2_aux = amp_fit_curve(t2, params2y)

delta_t, align_mse, align_start, align_end, align_ratio, align_n = align_time(
    spline1x, spline1y, spline2x, spline2y
)

aux_delta_t, aux_mse, aux_start, aux_end, aux_ratio, aux_n = align_time(
    lambda t: amp_model(t, *params1x),
    lambda t: amp_model(t, *params1y),
    lambda t: amp_model(t, *params2x),
    lambda t: amp_model(t, *params2y),
)

# ------------------------------
# 6. 绘图
# ------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 方式1 x-t
axes[0,0].scatter(t1, x1, s=3, alpha=0.45, label='观测数据')
axes[0,0].plot(tf1x, fx1, 'r-', lw=2, label='三次样条主方法')
axes[0,0].plot(tf1x_aux, fx1_aux, 'g--', lw=1.5, label='线性调幅辅助')
axes[0,0].set_xlabel('时间 (s)')
axes[0,0].set_ylabel('X (m)')
axes[0,0].set_title('方式1 (4Hz) X‑t')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 方式1 y-t
axes[0,1].scatter(t1, y1, s=3, alpha=0.45, label='观测数据')
axes[0,1].plot(tf1y, fy1, 'r-', lw=2, label='三次样条主方法')
axes[0,1].plot(tf1y_aux, fy1_aux, 'g--', lw=1.5, label='线性调幅辅助')
axes[0,1].set_xlabel('时间 (s)')
axes[0,1].set_ylabel('Y (m)')
axes[0,1].set_title('方式1 (4Hz) Y‑t')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# 方式2 x-t
axes[1,0].scatter(t2, x2, s=3, alpha=0.45, label='观测数据')
axes[1,0].plot(tf2x, fx2, 'r-', lw=2, label='三次样条主方法')
axes[1,0].plot(tf2x_aux, fx2_aux, 'g--', lw=1.5, label='线性调幅辅助')
axes[1,0].set_xlabel('时间 (s)')
axes[1,0].set_ylabel('X (m)')
axes[1,0].set_title('方式2 (5Hz) X‑t')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# 方式2 y-t
axes[1,1].scatter(t2, y2, s=3, alpha=0.45, label='观测数据')
axes[1,1].plot(tf2y, fy2, 'r-', lw=2, label='三次样条主方法')
axes[1,1].plot(tf2y_aux, fy2_aux, 'g--', lw=1.5, label='线性调幅辅助')
axes[1,1].set_xlabel('时间 (s)')
axes[1,1].set_ylabel('Y (m)')
axes[1,1].set_title('方式2 (5Hz) Y‑t')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

ti_align, x1_align, y1_align, x2_align, y2_align, align_error = alignment_series(
    delta_t, spline1x, spline1y, spline2x, spline2y
)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

axes2[0].plot(x1, y1, 'b-', lw=1, alpha=0.8, label='方式1')
axes2[0].plot(x2, y2, 'r-', lw=1, alpha=0.8, label='方式2')
axes2[0].set_xlabel('X (m)')
axes2[0].set_ylabel('Y (m)')
axes2[0].set_title('对齐前二维轨迹')
axes2[0].axis('equal')
axes2[0].legend()
axes2[0].grid(alpha=0.3)

axes2[1].plot(x1_align, y1_align, 'b-', lw=1.5, label='方式1样条轨迹')
axes2[1].plot(x2_align, y2_align, 'r--', lw=1.5, label='方式2配准后样条轨迹')
axes2[1].set_xlabel('X (m)')
axes2[1].set_ylabel('Y (m)')
axes2[1].set_title(f'样条主方法对齐后 Δt={delta_t:.4f}s')
axes2[1].axis('equal')
axes2[1].legend()
axes2[1].grid(alpha=0.3)

axes2[2].plot(ti_align, align_error, 'k-', lw=1)
axes2[2].set_xlabel('方式1时间 (s)')
axes2[2].set_ylabel('位置误差 (m)')
axes2[2].set_title('样条主方法对齐后位置误差')
axes2[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 输出结果
print("主方法：三次样条插值")
print(f"三次样条时间配准结果 Δt = {delta_t:.6f} s")
print(f"三次样条配准目标函数最小值 = {align_mse:.12g}")
print(f"三次样条参与配准的方式1时间区间 = [{align_start:.3f}, {align_end:.3f}] s, 重叠比例 = {align_ratio:.4f}, N = {align_n}")
print()
print("辅助方法：线性调幅余弦拟合")
print("方式1 X辅助拟合参数 (c, a, b, ω, φ):", params1x)
print("方式1 Y辅助拟合参数 (c, a, b, ω, φ):", params1y)
print("方式2 X辅助拟合参数 (c, a, b, ω, φ):", params2x)
print("方式2 Y辅助拟合参数 (c, a, b, ω, φ):", params2y)
print(f"线性调幅辅助配准结果 Δt = {aux_delta_t:.6f} s")
print(f"线性调幅辅助目标函数最小值 = {aux_mse:.12g}")
print(f"线性调幅辅助参与配准的方式1时间区间 = [{aux_start:.3f}, {aux_end:.3f}] s, 重叠比例 = {aux_ratio:.4f}, N = {aux_n}")
