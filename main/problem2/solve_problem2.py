from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import correlate, correlation_lags

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

# 路径和参数
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "数据以及可视化" / "附件2.xlsx"

DT_OUT = 0.1
COMPARE_DT = 0.25
MIN_OVERLAP_RATIO = 0.70
CORRELATION_DT = 0.05
LM_DELTA_RADIUS_S = 2.0
SMOOTH_WINDOW_POINTS = 3
REFINE_RADIUS_S = 20.0          # 精修搜索半径
DELTA_CANDIDATE_COUNT = 40
DELTA_CANDIDATE_MIN_SEPARATION_S = 0.5

# 保存对齐结果
@dataclass
class AlignmentResult:
    delta_t2_minus_t1: float
    cross_correlation_initial_delta: float
    cross_correlation_score: float
    delta_candidate_count: int
    procrustes_initial_theta_rad: float
    theta_rad: float
    bias_add_to_method2: np.ndarray
    objective_mse: float
    residual_rmse: float
    overlap_start: float
    overlap_end: float
    overlap_ratio: float
    n_compare: int

# 读取附件二
def load_attachment2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df1 = pd.read_excel(DATA_PATH, sheet_name="方式1(4Hz)")
    df2 = pd.read_excel(DATA_PATH, sheet_name="方式2(5Hz)")
    df1 = standardize_frame(df1, "方式1")
    df2 = standardize_frame(df2, "方式2")
    return (
        df1["time_s"].to_numpy(float),
        df1[["x_m", "y_m"]].to_numpy(float),
        df2["time_s"].to_numpy(float),
        df2[["x_m", "y_m"]].to_numpy(float),
    )

def standardize_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.iloc[:, :3].copy()
    out.columns = ["time_s", "x_m", "y_m"]
    for col in ["time_s", "x_m", "y_m"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["time_s", "x_m", "y_m"]].isna().any().any():
        raise ValueError(f"{name} 含有缺失值或非数值。")
    out = out.sort_values("time_s")
    if out["time_s"].duplicated().any():
        out = out.groupby("time_s", as_index=False)[["x_m", "y_m"]].mean()
    if not np.all(np.diff(out["time_s"].to_numpy(float)) > 0):
        raise ValueError(f"{name} 的时间值不是严格递增。")
    return out

# 线性插值
def interp_xy(t: np.ndarray, xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack((
        np.interp(q, t, xy[:, 0]),
        np.interp(q, t, xy[:, 1]),
    ))

# 位置变换
def rotate_xy(xy: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack((
        c * xy[:, 0] - s * xy[:, 1],
        s * xy[:, 0] + c * xy[:, 1],
    ))

def transform_method2_xy(xy: np.ndarray, theta: float, translation: np.ndarray) -> np.ndarray:
    return rotate_xy(xy, theta) + translation

# 时间偏差的可行范围
def overlap_bounds(t1: np.ndarray, t2: np.ndarray, delta: float) -> tuple[float, float, float]:
    lo = max(float(t1[0]), float(t2[0] - delta))
    hi = min(float(t1[-1]), float(t2[-1] - delta))
    return lo, hi, hi - lo

def feasible_delta_bounds(t1: np.ndarray, t2: np.ndarray) -> tuple[float, float, float, float]:
    min_duration = min(float(t1[-1] - t1[0]), float(t2[-1] - t2[0]))
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min_duration)
    delta_min = float(t2[0] - t1[-1] + min_overlap)
    delta_max = float(t2[-1] - t1[0] - min_overlap)
    if delta_min >= delta_max:
        raise ValueError("没有可行的时间偏差搜索区间。")
    return delta_min, delta_max, min_overlap, min_duration

# 三点滑动均值
def smooth_xy(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if len(xy) < SMOOTH_WINDOW_POINTS:
        return xy.copy()
    padded = np.pad(xy, ((1, 1), (0, 0)), mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / SMOOTH_WINDOW_POINTS

# 互相关相关函数
def standardize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    std = float(signal.std(ddof=1))
    if std <= 1e-12:
        return signal - float(signal.mean())
    return (signal - float(signal.mean())) / std

def resample_rotation_invariant_features(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(float(t[0]), float(t[-1]) + 1e-9, CORRELATION_DT)
    x_grid = np.interp(grid, t, xy[:, 0])
    y_grid = np.interp(grid, t, xy[:, 1])
    xy_grid = np.column_stack((x_grid, y_grid))
    centered = xy_grid - xy_grid.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1)
    dt = float(np.median(np.diff(grid)))
    velocity = np.gradient(xy_grid, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    radius_rate = np.gradient(radius, dt)
    return grid, np.column_stack((
        standardize_signal(radius),
        standardize_signal(speed),
        standardize_signal(radius_rate),
    ))

def resample_xy_for_correlation(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(float(t[0]), float(t[-1]) + 1e-9, CORRELATION_DT)
    x_grid = np.interp(grid, t, xy[:, 0])
    y_grid = np.interp(grid, t, xy[:, 1])
    return grid, np.column_stack((standardize_signal(x_grid), standardize_signal(y_grid)))

def estimate_delta_candidates_by_invariant_cross_correlation(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    delta_min: float, delta_max: float, min_overlap: float,
) -> list[tuple[float, float]]:
    grid1, features1 = resample_rotation_invariant_features(t1, xy1)
    grid2, features2 = resample_rotation_invariant_features(t2, xy2)
    corr = np.zeros(len(grid1) + len(grid2) - 1, dtype=float)
    for idx in range(features1.shape[1]):
        corr += correlate(features1[:, idx], features2[:, idx], mode="full", method="fft")
    overlap = correlate(np.ones(len(grid1)), np.ones(len(grid2)), mode="full", method="fft")
    lags = correlation_lags(len(grid1), len(grid2), mode="full")
    delta_values = grid2[0] - grid1[0] - lags * CORRELATION_DT
    min_overlap_samples = min_overlap / CORRELATION_DT
    valid = (delta_values >= delta_min) & (delta_values <= delta_max) & (overlap >= min_overlap_samples)
    if not np.any(valid):
        valid = (delta_values >= delta_min) & (delta_values <= delta_max)
    if not np.any(valid):
        raise ValueError("旋转不变特征互相关法没有得到可行的时间偏差候选值。")
    scores = np.full_like(corr, -np.inf)
    scores[valid] = corr[valid] / np.maximum(overlap[valid], 1.0)
    candidates: list[tuple[float, float]] = []
    for idx in np.argsort(scores)[::-1]:
        score = float(scores[idx])
        if not np.isfinite(score):
            break
        delta = float(delta_values[idx])
        if any(abs(delta - d) < DELTA_CANDIDATE_MIN_SEPARATION_S for d, _ in candidates):
            continue
        candidates.append((delta, score))
        if len(candidates) >= DELTA_CANDIDATE_COUNT:
            break
    if not candidates:
        raise ValueError("没有找到有效的旋转不变互相关候选值。")
    return candidates

def estimate_delta_by_coordinate_cross_correlation(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    delta_min: float, delta_max: float, min_overlap: float,
) -> tuple[float, float]:
    grid1, xy1_grid = resample_xy_for_correlation(t1, xy1)
    grid2, xy2_grid = resample_xy_for_correlation(t2, xy2)
    corr = (correlate(xy1_grid[:, 0], xy2_grid[:, 0], mode="full", method="fft") +
            correlate(xy1_grid[:, 1], xy2_grid[:, 1], mode="full", method="fft"))
    overlap = correlate(np.ones(len(grid1)), np.ones(len(grid2)), mode="full", method="fft")
    lags = correlation_lags(len(grid1), len(grid2), mode="full")
    delta_values = grid2[0] - grid1[0] - lags * CORRELATION_DT
    min_overlap_samples = min_overlap / CORRELATION_DT
    valid = (delta_values >= delta_min) & (delta_values <= delta_max) & (overlap >= min_overlap_samples)
    if not np.any(valid):
        valid = (delta_values >= delta_min) & (delta_values <= delta_max)
    if not np.any(valid):
        raise ValueError("坐标互相关法没有得到可行的时间偏差候选值。")
    scores = np.full_like(corr, -np.inf)
    scores[valid] = corr[valid] / np.maximum(overlap[valid], 1.0)
    best_idx = int(np.argmax(scores))
    return float(delta_values[best_idx]), float(scores[best_idx])

def estimate_rotation_translation(p1: np.ndarray, p2: np.ndarray) -> tuple[float, np.ndarray]:
    center1 = p1.mean(axis=0)
    center2 = p2.mean(axis=0)
    p1_centered = p1 - center1
    p2_centered = p2 - center2
    cos_term = float(np.sum(p1_centered[:, 0] * p2_centered[:, 0] + p1_centered[:, 1] * p2_centered[:, 1]))
    sin_term = float(np.sum(-p1_centered[:, 0] * p2_centered[:, 1] + p1_centered[:, 1] * p2_centered[:, 0]))
    theta = float(np.arctan2(sin_term, cos_term))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = center1 - (rotation_matrix @ center2)
    return theta, translation

def evaluate_initial_alignment_candidate(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    delta: float, min_overlap: float,
) -> tuple[float, float, np.ndarray, int]:
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return np.inf, 0.0, np.zeros(2), 0
    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    if len(q) < 4:
        return np.inf, 0.0, np.zeros(2), len(q)
    p1 = interp_xy(t1, xy1, q)
    p2 = interp_xy(t2, xy2, q + delta)
    theta, translation = estimate_rotation_translation(p1, p2)
    rotated_p2 = p2 @ np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    residual = p1 - (rotated_p2 + translation)
    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    return mse, theta, translation, len(q)

# 精修：只优化时间差，平移用均值差
def alignment_objective(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    delta: float, min_overlap: float,
) -> tuple[float, np.ndarray, float, int, np.ndarray]:
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return np.inf, np.zeros(2), overlap, 0, np.empty((0, 2))
    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    residual = interp_xy(t1, xy1, q) - interp_xy(t2, xy2, q + delta)
    bias = residual.mean(axis=0)
    debiased = residual - bias
    mse = float(np.mean(np.sum(debiased * debiased, axis=1)))
    return mse, bias, overlap, len(q), debiased

# 主对齐流程
def estimate_alignment(t1: np.ndarray, xy1: np.ndarray, t2: np.ndarray, xy2: np.ndarray) -> AlignmentResult:
    delta_min, delta_max, min_overlap, min_duration = feasible_delta_bounds(t1, t2)

    # 先粗略找几个候选时间差。
    delta_candidates = estimate_delta_candidates_by_invariant_cross_correlation(
        t1, xy1, t2, xy2, delta_min, delta_max, min_overlap)
    coord_delta, coord_score = estimate_delta_by_coordinate_cross_correlation(
        t1, xy1, t2, xy2, delta_min, delta_max, min_overlap)
    delta_candidates.append((coord_delta, coord_score))

    candidate_records = []
    for cand_delta, cand_score in delta_candidates:
        mse, theta, trans, n = evaluate_initial_alignment_candidate(
            t1, xy1, t2, xy2, cand_delta, min_overlap)
        if np.isfinite(mse):
            candidate_records.append((mse, cand_delta, cand_score, theta, trans, n))
    if not candidate_records:
        raise ValueError("没有时间偏差候选通过Procrustes评估。")

    _, init_delta, init_score, init_theta, _, _ = min(candidate_records, key=lambda x: x[0])

    # 再精修时间差，平移偏差用均值残差算。
    local_min = max(delta_min, init_delta - REFINE_RADIUS_S)
    local_max = min(delta_max, init_delta + REFINE_RADIUS_S)
    local_grid = np.linspace(local_min, local_max, 500)
    local_scores = np.array([alignment_objective(t1, xy1, t2, xy2, d, min_overlap)[0] for d in local_grid])
    best_delta = float(local_grid[int(np.argmin(local_scores))])
    local_step = float(local_grid[1] - local_grid[0])

    result = minimize_scalar(
        lambda d: alignment_objective(t1, xy1, t2, xy2, d, min_overlap)[0],
        bounds=(max(delta_min, best_delta - 3 * local_step), min(delta_max, best_delta + 3 * local_step)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    delta = float(result.x if result.success else best_delta)
    mse, bias, overlap, n_compare, _ = alignment_objective(t1, xy1, t2, xy2, delta, min_overlap)
    lo, hi, _ = overlap_bounds(t1, t2, delta)

    return AlignmentResult(
        delta_t2_minus_t1=delta,
        cross_correlation_initial_delta=init_delta,
        cross_correlation_score=init_score,
        delta_candidate_count=len(candidate_records),
        procrustes_initial_theta_rad=init_theta,
        theta_rad=0.0,                     # 这里不考虑旋转
        bias_add_to_method2=bias,
        objective_mse=mse,
        residual_rmse=float(np.sqrt(mse)),
        overlap_start=lo,
        overlap_end=hi,
        overlap_ratio=overlap / min_duration,
        n_compare=n_compare,
    )

# 对齐后的RMSE
def compute_aligned_rmse(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    alignment: AlignmentResult,
) -> float:
    delta = alignment.delta_t2_minus_t1
    bias = alignment.bias_add_to_method2
    lo, hi = alignment.overlap_start, alignment.overlap_end
    mask = (t1 >= lo) & (t1 <= hi)
    t1_overlap = t1[mask]
    xy1_overlap = xy1[mask]
    t2_query = t1_overlap + delta
    valid = (t2_query >= t2[0]) & (t2_query <= t2[-1])
    if np.sum(valid) < 2:
        return float("nan")
    t2_query = t2_query[valid]
    xy1_valid = xy1_overlap[valid]
    xy2_interp = interp_xy(t2, xy2, t2_query)
    xy2_corrected = xy2_interp + bias
    residuals = xy1_valid - xy2_corrected
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

# 卡尔曼滤波
def estimate_measurement_covariance(raw_xy: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    noise = raw_xy - smooth
    var = np.var(noise, axis=0, ddof=1)
    var = np.maximum(var, np.array([0.05**2, 0.05**2]))
    return np.diag(var)

def estimate_process_accel_std(t: np.ndarray, xy: np.ndarray) -> float:
    dt = float(np.median(np.diff(t)))
    velocity = np.gradient(xy, dt, axis=0)
    accel = np.gradient(velocity, dt, axis=0)
    accel_norm = np.linalg.norm(accel, axis=1)
    return float(max(0.2, np.nanpercentile(accel_norm, 75) * 0.35))

def predict(state: np.ndarray, cov: np.ndarray, dt: float, accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        return state, cov
    f = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    q = accel_std**2
    process = q * np.array([
        [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
        [0.0, dt**4 / 4, 0.0, dt**3 / 2],
        [dt**3 / 2, 0.0, dt**2, 0.0],
        [0.0, dt**3 / 2, 0.0, dt**2],
    ])
    return f @ state, f @ cov @ f.T + process

def update(state: np.ndarray, cov: np.ndarray, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    innovation = z - h @ state
    s = h @ cov @ h.T + r
    k = cov @ h.T @ np.linalg.inv(s)
    new_state = state + k @ innovation
    new_cov = (np.eye(4) - k @ h) @ cov
    return new_state, new_cov

def initial_state(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2_corrected: np.ndarray,
    delta: float, start: float,
) -> np.ndarray:
    h = 0.5
    p0 = 0.5 * (interp_xy(t1, xy1, np.array([start]))[0] +
                interp_xy(t2, xy2_corrected, np.array([start + delta]))[0])
    p1 = 0.5 * (interp_xy(t1, xy1, np.array([start + h]))[0] +
                interp_xy(t2, xy2_corrected, np.array([start + h + delta]))[0])
    velocity = (p1 - p0) / h
    return np.array([p0[0], p0[1], velocity[0], velocity[1]], dtype=float)

def run_kalman_filter(
    t1: np.ndarray, xy1: np.ndarray,
    t2: np.ndarray, xy2: np.ndarray,
    alignment: AlignmentResult,
    r1: np.ndarray, r2: np.ndarray,
    accel_std: float,
) -> pd.DataFrame:
    # 这里旋转角为0，只做平移修正。
    xy2_corrected = transform_method2_xy(xy2, alignment.theta_rad, alignment.bias_add_to_method2)
    lo, hi = alignment.overlap_start, alignment.overlap_end

    observations: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    mask1 = (t1 >= lo) & (t1 <= hi)
    for t, point in zip(t1[mask1], xy1[mask1]):
        observations.append((float(t), point.astype(float), r1, "method1"))

    t2_aligned = t2 - alignment.delta_t2_minus_t1
    mask2 = (t2_aligned >= lo) & (t2_aligned <= hi)
    for t, point in zip(t2_aligned[mask2], xy2_corrected[mask2]):
        observations.append((float(t), point.astype(float), r2, "method2"))

    observations.sort(key=lambda item: item[0])
    output_times = np.arange(lo, hi + 1e-9, DT_OUT)

    state = initial_state(t1, xy1, t2, xy2_corrected, alignment.delta_t2_minus_t1, lo)
    cov = np.diag([10.0, 10.0, 5.0, 5.0])
    current_time = lo
    obs_index = 0
    rows: list[dict] = []

    for out_time in output_times:
        while obs_index < len(observations) and observations[obs_index][0] <= out_time + 1e-9:
            obs_time, z, r, source = observations[obs_index]
            state, cov = predict(state, cov, obs_time - current_time, accel_std)
            current_time = obs_time
            state, cov = update(state, cov, z, r)
            obs_index += 1

        state, cov = predict(state, cov, float(out_time - current_time), accel_std)
        current_time = float(out_time)

        p1_interp = interp_xy(t1, xy1, np.array([out_time]))[0]
        p2_interp = interp_xy(t2, xy2_corrected, np.array([out_time + alignment.delta_t2_minus_t1]))[0]
        rows.append({
            "time_s": float(out_time - output_times[0]),
            "method1_time_s": float(out_time),
            "method2_time_s": float(out_time + alignment.delta_t2_minus_t1),
            "x_method1_interp_m": float(p1_interp[0]),
            "y_method1_interp_m": float(p1_interp[1]),
            "x_method2_corrected_interp_m": float(p2_interp[0]),
            "y_method2_corrected_interp_m": float(p2_interp[1]),
            "x_kalman_m": float(state[0]),
            "y_kalman_m": float(state[1]),
            "vx_kalman_mps": float(state[2]),
            "vy_kalman_mps": float(state[3]),
            "kalman_position_std_x_m": float(np.sqrt(max(cov[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(cov[1, 1], 0.0))),
        })

    return pd.DataFrame(rows)

# 主程序
def main() -> None:
    t1, xy1_raw, t2, xy2_raw = load_attachment2()

    # 用轻度平滑后的数据做对齐。
    xy1_smooth = smooth_xy(t1, xy1_raw)
    xy2_smooth = smooth_xy(t2, xy2_raw)
    alignment = estimate_alignment(t1, xy1_smooth, t2, xy2_smooth)

    # 输出时间偏差和平移偏差。
    rmse_aligned = compute_aligned_rmse(t1, xy1_raw, t2, xy2_raw, alignment)
    print(f"粗算 Δt: {alignment.cross_correlation_initial_delta:.6f} s")
    print(f"精算 Δt: {alignment.delta_t2_minus_t1:.6f} s")
    print(f"位移偏差（加至方式2）：x方向 = {alignment.bias_add_to_method2[0]:.6f} m，"
          f"y方向 = {alignment.bias_add_to_method2[1]:.6f} m")
    print(f"对齐后两组散点 RMSE: {rmse_aligned:.6f} m")

    # 后面做卡尔曼融合和画图。
    r1 = estimate_measurement_covariance(xy1_raw, xy1_smooth)
    r2 = estimate_measurement_covariance(xy2_raw, xy2_smooth)

    q_time = np.arange(alignment.overlap_start, alignment.overlap_end + 1e-9, DT_OUT)
    method2_corrected = transform_method2_xy(xy2_raw, alignment.theta_rad, alignment.bias_add_to_method2)
    fused_for_accel = 0.5 * (
        interp_xy(t1, xy1_smooth, q_time) +
        interp_xy(t2, method2_corrected, q_time + alignment.delta_t2_minus_t1)
    )
    accel_std = estimate_process_accel_std(q_time, fused_for_accel)

    trajectory = run_kalman_filter(t1, xy1_raw, t2, xy2_raw, alignment, r1, r2, accel_std)

    # 看一下融合轨迹和两种方式的偏差。
    err1 = np.sqrt(np.mean((trajectory["x_kalman_m"] - trajectory["x_method1_interp_m"])**2 +
                           (trajectory["y_kalman_m"] - trajectory["y_method1_interp_m"])**2))
    err2 = np.sqrt(np.mean((trajectory["x_kalman_m"] - trajectory["x_method2_corrected_interp_m"])**2 +
                           (trajectory["y_kalman_m"] - trajectory["y_method2_corrected_interp_m"])**2))
    print(f"融合轨迹相对方式1的平均RMSE：{err1:.4f} m")
    print(f"融合轨迹相对方式2的平均RMSE：{err2:.4f} m")

    # 三维图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("轨迹融合对比（时间‑空间）")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("X坐标 (m)")
    ax.set_zlabel("Y坐标 (m)")

    mask1 = (t1 >= alignment.overlap_start) & (t1 <= alignment.overlap_end)
    ax.scatter(t1[mask1], xy1_raw[mask1, 0], xy1_raw[mask1, 1],
               c="dodgerblue", s=10, alpha=0.7, label="方式1 原始点")

    t2_aligned = t2 - alignment.delta_t2_minus_t1
    mask2 = (t2_aligned >= alignment.overlap_start) & (t2_aligned <= alignment.overlap_end)
    ax.scatter(t2_aligned[mask2], method2_corrected[mask2, 0], method2_corrected[mask2, 1],
               c="orange", s=10, alpha=0.7, label="方式2 校正后点")

    kalman_t = trajectory["time_s"].to_numpy() + alignment.overlap_start
    ax.plot(kalman_t, trajectory["x_kalman_m"], trajectory["y_kalman_m"],
            "r-", linewidth=2, label="卡尔曼融合")

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
