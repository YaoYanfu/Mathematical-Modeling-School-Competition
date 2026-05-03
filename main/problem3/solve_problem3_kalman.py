from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "数据以及可视化" / "附件3.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "问题三"

DT_OUT = 0.1
COMPARE_DT = 0.25
MIN_OVERLAP_RATIO = 0.70
SMOOTH_WINDOW_POINTS = 3
REFINE_RADIUS_S = 20.0
OUTLIER_WINDOW = 9
OUTLIER_MAD_THRESHOLD = 4.0
OUTLIER_MIN_ABS_M = 1.0

# 记录异常点处理情况，方便后面检查。
CLEANING_REPORT: dict[str, dict] = {}


@dataclass
class AlignmentResult:
    delta_t2_minus_t1: float
    cross_correlation_initial_delta: float
    cross_correlation_score: float
    bias_add_to_method2: np.ndarray
    objective_mse: float
    residual_rmse: float
    overlap_start: float
    overlap_end: float
    overlap_ratio: float
    n_compare: int
    bias_norm: float
    system_bias_exists: bool


def load_attachment3() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df1 = pd.read_excel(DATA_PATH, sheet_name=0)
    df2 = pd.read_excel(DATA_PATH, sheet_name=1)
    df1 = standardize_frame(df1, "方式1")
    df2 = standardize_frame(df2, "方式2")
    return (
        df1["time_s"].to_numpy(float),
        df1[["x_m", "y_m"]].to_numpy(float),
        df2["time_s"].to_numpy(float),
        df2[["x_m", "y_m"]].to_numpy(float),
    )


def standardize_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # 附件里只需要时间、x、y三列。
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
    out = replace_local_median_mad_outliers(out, name)
    return out


def replace_local_median_mad_outliers(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # 用局部中位数找明显跳点，找到后用插值补回去。
    t = df["time_s"].to_numpy(float)
    cleaned = df.copy()
    flagged_by_axis: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    robust_scales: dict[str, float] = {}

    for col in ["x_m", "y_m"]:
        values = df[col].to_numpy(float)
        local_median = (
            pd.Series(values)
            .rolling(window=OUTLIER_WINDOW, center=True, min_periods=max(3, OUTLIER_WINDOW // 2))
            .median()
            .bfill()
            .ffill()
            .to_numpy(float)
        )
        residual = values - local_median
        mad = float(np.median(np.abs(residual - np.median(residual))))
        robust_scale = 1.4826 * mad
        threshold = max(OUTLIER_MAD_THRESHOLD * robust_scale, OUTLIER_MIN_ABS_M)
        flagged_by_axis[col] = np.abs(residual) > threshold
        thresholds[col] = threshold
        robust_scales[col] = robust_scale

    flagged = flagged_by_axis["x_m"] | flagged_by_axis["y_m"]
    if flagged.any():
        normal = ~flagged
        if normal.sum() < 2:
            raise ValueError(f"{name} 经过轻度异常点筛查后正常点过少。")
        for col in ["x_m", "y_m"]:
            cleaned.loc[flagged, col] = np.interp(
                t[flagged],
                t[normal],
                cleaned.loc[normal, col].to_numpy(float),
            )

    CLEANING_REPORT[name] = {
        "方法": "局部中位数MAD轻度异常点替换",
        "滚动窗口点数": OUTLIER_WINDOW,
        "MAD倍数": OUTLIER_MAD_THRESHOLD,
        "单坐标最小阈值_m": OUTLIER_MIN_ABS_M,
        "标记异常点数": int(flagged.sum()),
        "x方向标记点数": int(flagged_by_axis["x_m"].sum()),
        "y方向标记点数": int(flagged_by_axis["y_m"].sum()),
        "前30个异常点索引": np.where(flagged)[0][:30].astype(int).tolist(),
        "x方向稳健尺度_m": robust_scales["x_m"],
        "y方向稳健尺度_m": robust_scales["y_m"],
        "x方向阈值_m": thresholds["x_m"],
        "y方向阈值_m": thresholds["y_m"],
        "替换方式": "使用未标记邻近样本进行线性插值",
    }
    return cleaned


def smooth_xy(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    # 三点均值，主要是让配准时别被小抖动带偏。
    if len(xy) < SMOOTH_WINDOW_POINTS:
        return xy.copy()
    padded = np.pad(xy, ((1, 1), (0, 0)), mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / SMOOTH_WINDOW_POINTS


def interp_xy(t: np.ndarray, xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.interp(q, t, xy[:, 0]),
            np.interp(q, t, xy[:, 1]),
        )
    )


def overlap_bounds(t1: np.ndarray, t2: np.ndarray, delta: float) -> tuple[float, float, float]:
    lo = max(float(t1[0]), float(t2[0] - delta))
    hi = min(float(t1[-1]), float(t2[-1] - delta))
    return lo, hi, hi - lo


def feasible_delta_bounds(t1: np.ndarray, t2: np.ndarray) -> tuple[float, float, float]:
    min_duration = min(float(t1[-1] - t1[0]), float(t2[-1] - t2[0]))
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min_duration)
    delta_min = float(t2[0] - t1[-1] + min_overlap)
    delta_max = float(t2[-1] - t1[0] - min_overlap)
    if delta_min >= delta_max:
        raise ValueError("没有满足重叠约束的可行时间偏差搜索区间。")
    return delta_min, delta_max, min_overlap


def cross_correlation_score(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta: float,
    min_overlap: float,
) -> float:
    # 给某一个时间差打分，分数越高说明越像。
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return -np.inf

    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    p1 = interp_xy(t1, xy1, q)
    p2 = interp_xy(t2, xy2, q + delta)
    p1 = p1 - p1.mean(axis=0)
    p2 = p2 - p2.mean(axis=0)
    numerator = float(np.sum(p1 * p2))
    denominator = float(np.sqrt(np.sum(p1 * p1) * np.sum(p2 * p2)))
    if denominator <= 1e-12:
        return -np.inf
    return numerator / denominator


def estimate_delta_by_cross_correlation(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
) -> tuple[float, float]:
    # 先粗略扫一遍，得到时间偏差初值。
    delta_min, delta_max, min_overlap = feasible_delta_bounds(t1, t2)
    grid = np.linspace(delta_min, delta_max, 1600)
    scores = []
    for d in grid:
        scores.append(cross_correlation_score(t1, xy1, t2, xy2, d, min_overlap))
    scores = np.array(scores)
    best = float(grid[int(np.nanargmax(scores))])
    best_score = float(np.nanmax(scores))
    return best, best_score


def alignment_objective(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta: float,
    min_overlap: float,
) -> tuple[float, np.ndarray, float, int, np.ndarray]:
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return np.inf, np.zeros(2), overlap, 0, np.empty((0, 2))

    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    residual = interp_xy(t1, xy1, q) - interp_xy(t2, xy2, q + delta)
    bias = residual.mean(axis=0)
    debiased = residual - bias
    mse = float(np.mean(np.sum(debiased * debiased, axis=1)))
    return mse, bias, overlap, len(q), residual


def estimate_alignment(t1: np.ndarray, xy1: np.ndarray, t2: np.ndarray, xy2: np.ndarray) -> AlignmentResult:
    delta_min, delta_max, min_overlap = feasible_delta_bounds(t1, t2)
    init_delta, init_score = estimate_delta_by_cross_correlation(t1, xy1, t2, xy2)

    # 再在初值附近精修，同时估计平移偏差。
    local_min = max(delta_min, init_delta - REFINE_RADIUS_S)
    local_max = min(delta_max, init_delta + REFINE_RADIUS_S)
    local_grid = np.linspace(local_min, local_max, 500)
    local_scores = []
    for d in local_grid:
        local_scores.append(alignment_objective(t1, xy1, t2, xy2, d, min_overlap)[0])
    local_scores = np.array(local_scores)
    local_best = float(local_grid[int(np.nanargmin(local_scores))])
    local_step = float(local_grid[1] - local_grid[0])

    result = minimize_scalar(
        lambda delta: alignment_objective(t1, xy1, t2, xy2, delta, min_overlap)[0],
        bounds=(max(delta_min, local_best - 3 * local_step), min(delta_max, local_best + 3 * local_step)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    delta = float(result.x if result.success else local_best)
    mse, bias, overlap, n_compare, residual = alignment_objective(t1, xy1, t2, xy2, delta, min_overlap)
    lo, hi, _ = overlap_bounds(t1, t2, delta)
    bias_norm = float(np.linalg.norm(bias))

    min_duration = min(float(t1[-1] - t1[0]), float(t2[-1] - t2[0]))
    return AlignmentResult(
        delta_t2_minus_t1=delta,
        cross_correlation_initial_delta=init_delta,
        cross_correlation_score=init_score,
        bias_add_to_method2=bias,
        objective_mse=mse,
        residual_rmse=float(np.sqrt(mse)),
        overlap_start=lo,
        overlap_end=hi,
        overlap_ratio=overlap / min_duration,
        n_compare=n_compare,
        bias_norm=bias_norm,
        system_bias_exists=True,
    )


def estimate_measurement_covariance(raw_xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
    noise = raw_xy - reference_xy
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
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q = accel_std**2
    process = q * np.array(
        [
            [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
            [0.0, dt**4 / 4, 0.0, dt**3 / 2],
            [dt**3 / 2, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2, 0.0, dt**2],
        ]
    )
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
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2_for_filter: np.ndarray,
    delta: float,
    start: float,
) -> np.ndarray:
    h = 0.5
    p0 = 0.5 * (interp_xy(t1, xy1, np.array([start]))[0] + interp_xy(t2, xy2_for_filter, np.array([start + delta]))[0])
    p1 = 0.5 * (
        interp_xy(t1, xy1, np.array([start + h]))[0]
        + interp_xy(t2, xy2_for_filter, np.array([start + h + delta]))[0]
    )
    velocity = (p1 - p0) / h
    return np.array([p0[0], p0[1], velocity[0], velocity[1]], dtype=float)


def run_kalman_filter(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    alignment: AlignmentResult,
    r1: np.ndarray,
    r2: np.ndarray,
    accel_std: float,
) -> pd.DataFrame:
    # 方式2先去偏，然后两路观测一起进入卡尔曼滤波。
    bias_used = alignment.bias_add_to_method2
    xy2_for_filter = xy2 + bias_used
    lo, hi = alignment.overlap_start, alignment.overlap_end

    observations: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    mask1 = (t1 >= lo) & (t1 <= hi)
    for t, point in zip(t1[mask1], xy1[mask1]):
        observations.append((float(t), point.astype(float), r1, "method1"))

    t2_aligned = t2 - alignment.delta_t2_minus_t1
    mask2 = (t2_aligned >= lo) & (t2_aligned <= hi)
    for t, point in zip(t2_aligned[mask2], xy2_for_filter[mask2]):
        observations.append((float(t), point.astype(float), r2, "method2"))

    observations.sort(key=lambda item: item[0])
    output_times = np.arange(lo, hi + 1e-9, DT_OUT)

    state = initial_state(t1, xy1, t2, xy2_for_filter, alignment.delta_t2_minus_t1, lo)
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
        p2_interp = interp_xy(t2, xy2_for_filter, np.array([out_time + alignment.delta_t2_minus_t1]))[0]
        rows.append(
            {
                "time_s": float(out_time - output_times[0]),
                "method1_time_s": float(out_time),
                "method2_time_s": float(out_time + alignment.delta_t2_minus_t1),
                "x_method1_interp_m": float(p1_interp[0]),
                "y_method1_interp_m": float(p1_interp[1]),
                "x_method2_filter_input_interp_m": float(p2_interp[0]),
                "y_method2_filter_input_interp_m": float(p2_interp[1]),
                "x_kalman_m": float(state[0]),
                "y_kalman_m": float(state[1]),
                "vx_kalman_mps": float(state[2]),
                "vy_kalman_mps": float(state[3]),
                "kalman_position_std_x_m": float(np.sqrt(max(cov[0, 0], 0.0))),
                "kalman_position_std_y_m": float(np.sqrt(max(cov[1, 1], 0.0))),
            }
        )

    return pd.DataFrame(rows)


def calculate_aligned_methods_rmse(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    alignment: AlignmentResult,
) -> dict[str, float | int]:
    method2_query_time = t1 + alignment.delta_t2_minus_t1
    valid = (method2_query_time >= t2[0]) & (method2_query_time <= t2[-1])
    if int(valid.sum()) < 2:
        return {
            "对齐后两方式RMSE_m": float("nan"),
            "对齐后两方式RMSE使用点数": int(valid.sum()),
        }

    xy2_aligned = interp_xy(
        t2,
        xy2 + alignment.bias_add_to_method2,
        method2_query_time[valid],
    )
    residual = xy1[valid] - xy2_aligned
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return {
        "对齐后两方式RMSE_m": rmse,
        "对齐后两方式RMSE使用点数": int(valid.sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    t1, xy1_raw, t2, xy2_raw = load_attachment3()

    # 对齐时用轻度平滑后的数据，最终滤波仍保留原始观测进入。
    xy1_smooth = smooth_xy(t1, xy1_raw)
    xy2_smooth = smooth_xy(t2, xy2_raw)
    alignment = estimate_alignment(t1, xy1_smooth, t2, xy2_smooth)

    r1 = estimate_measurement_covariance(xy1_raw, xy1_smooth)
    r2 = estimate_measurement_covariance(xy2_raw, xy2_smooth)

    bias_used = alignment.bias_add_to_method2
    q_time = np.arange(alignment.overlap_start, alignment.overlap_end + 1e-9, DT_OUT)
    fused_for_accel = 0.5 * (
        interp_xy(t1, xy1_smooth, q_time)
        + interp_xy(t2, xy2_smooth + bias_used, q_time + alignment.delta_t2_minus_t1)
    )
    accel_std = estimate_process_accel_std(q_time, fused_for_accel)

    trajectory = run_kalman_filter(t1, xy1_raw, t2, xy2_raw, alignment, r1, r2, accel_std)
    aligned_methods_rmse = calculate_aligned_methods_rmse(t1, xy1_raw, t2, xy2_raw, alignment)
    trajectory_path = OUT_DIR / "problem3_trajectory_10hz_kalman.csv"
    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    summary = {
        "源工作簿": str(DATA_PATH),
        "流程": "标准化 -> 局部中位数MAD轻度异常点替换 -> 使用三点滑动均值平滑辅助对齐和噪声估计 -> 互相关法估计时间偏差初值 -> 残差最小化精修 -> 估计空间偏差b -> 方式2去偏 -> 卡尔曼融合",
        "是否使用去趋势残差3σ": False,
        "是否使用平滑": True,
        "平滑方法": "三点滑动均值",
        "平滑窗口点数": SMOOTH_WINDOW_POINTS,
        "是否使用轻度异常点处理": True,
        "方式2相对方式1时间偏差_s": alignment.delta_t2_minus_t1,
        "互相关初始时间偏差_s": alignment.cross_correlation_initial_delta,
        "互相关得分": alignment.cross_correlation_score,
        "方式1时间偏差_s": 0.0,
        "方式2时间偏差_s": alignment.delta_t2_minus_t1,
        "偏差定义": "需加到方式2的偏差 = mean(方式1位置 - 时间对齐后的方式2位置)",
        "估计偏差_x_m": float(alignment.bias_add_to_method2[0]),
        "估计偏差_y_m": float(alignment.bias_add_to_method2[1]),
        "滤波使用偏差_x_m": float(bias_used[0]),
        "滤波使用偏差_y_m": float(bias_used[1]),
        "偏差模长_m": alignment.bias_norm,
        "系统偏差判定依据": "直接依据估计出的空间偏差b，不进行统计假设检验",
        "是否存在系统偏差": alignment.system_bias_exists,
        "对齐目标函数MSE": alignment.objective_mse,
        "对齐残差RMSE_m": alignment.residual_rmse,
        "重叠区间起点_方式1时间_s": alignment.overlap_start,
        "重叠区间终点_方式1时间_s": alignment.overlap_end,
        "重叠比例": alignment.overlap_ratio,
        "对齐比较点数": alignment.n_compare,
        "卡尔曼状态量": "[x, y, vx, vy]",
        "卡尔曼输出步长_s": DT_OUT,
        "过程加速度标准差_mps2": accel_std,
        "方式1测量协方差": r1.tolist(),
        "方式2测量协方差": r2.tolist(),
        "RMSE定义": "时间对齐并进行空间偏差校正后，方式1与方式2之间的二维位置RMSE",
        **aligned_methods_rmse,
        "数据清洗": CLEANING_REPORT,
        "10Hz轨迹行数": int(len(trajectory)),
        "轨迹CSV": str(trajectory_path),
    }
    summary_path = OUT_DIR / "problem3_kalman_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
