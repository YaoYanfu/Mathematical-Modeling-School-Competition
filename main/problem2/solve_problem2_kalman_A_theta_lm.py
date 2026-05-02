from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import correlate, correlation_lags


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "数据以及可视化" / "附件2.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "问题二"

DT_OUT = 0.1
COMPARE_DT = 0.25
MIN_OVERLAP_RATIO = 0.70
CORRELATION_DT = 0.05
LM_DELTA_RADIUS_S = 2.0
DELTA_CANDIDATE_COUNT = 40
DELTA_CANDIDATE_MIN_SEPARATION_S = 0.5

CLEANING_REPORT: dict[str, dict] = {}


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
        raise ValueError(f"{name} contains missing or non-numeric values.")
    out = out.sort_values("time_s")
    if out["time_s"].duplicated().any():
        out = out.groupby("time_s", as_index=False)[["x_m", "y_m"]].mean()
    if not np.all(np.diff(out["time_s"].to_numpy(float)) > 0):
        raise ValueError(f"{name} time values are not strictly increasing.")
    return out


def interp_xy(t: np.ndarray, xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.interp(q, t, xy[:, 0]),
            np.interp(q, t, xy[:, 1]),
        )
    )


def rotate_xy(xy: np.ndarray, theta: float) -> np.ndarray:
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.column_stack(
        (
            cos_theta * xy[:, 0] - sin_theta * xy[:, 1],
            sin_theta * xy[:, 0] + cos_theta * xy[:, 1],
        )
    )


def transform_method2_xy(xy: np.ndarray, theta: float, translation: np.ndarray) -> np.ndarray:
    return rotate_xy(xy, theta) + translation


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
        raise ValueError("No feasible time-offset search interval satisfies the overlap constraint.")
    return delta_min, delta_max, min_overlap, min_duration


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
    return grid, np.column_stack(
        (
            standardize_signal(radius),
            standardize_signal(speed),
            standardize_signal(radius_rate),
        )
    )


def resample_xy_for_correlation(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(float(t[0]), float(t[-1]) + 1e-9, CORRELATION_DT)
    x_grid = np.interp(grid, t, xy[:, 0])
    y_grid = np.interp(grid, t, xy[:, 1])
    return grid, np.column_stack((standardize_signal(x_grid), standardize_signal(y_grid)))


def estimate_delta_by_coordinate_cross_correlation(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta_min: float,
    delta_max: float,
    min_overlap: float,
) -> tuple[float, float]:
    grid1, xy1_grid = resample_xy_for_correlation(t1, xy1)
    grid2, xy2_grid = resample_xy_for_correlation(t2, xy2)
    corr = (
        correlate(xy1_grid[:, 0], xy2_grid[:, 0], mode="full", method="fft")
        + correlate(xy1_grid[:, 1], xy2_grid[:, 1], mode="full", method="fft")
    )
    overlap = correlate(
        np.ones(len(grid1), dtype=float),
        np.ones(len(grid2), dtype=float),
        mode="full",
        method="fft",
    )
    lags = correlation_lags(len(grid1), len(grid2), mode="full")
    delta_values = grid2[0] - grid1[0] - lags * CORRELATION_DT
    min_overlap_samples = min_overlap / CORRELATION_DT
    valid = (
        (delta_values >= delta_min)
        & (delta_values <= delta_max)
        & (overlap >= min_overlap_samples)
    )
    if not np.any(valid):
        valid = (delta_values >= delta_min) & (delta_values <= delta_max)
    if not np.any(valid):
        raise ValueError("No feasible time-offset candidate is available for coordinate cross-correlation.")

    scores = np.full_like(corr, -np.inf, dtype=float)
    scores[valid] = corr[valid] / np.maximum(overlap[valid], 1.0)
    best_index = int(np.argmax(scores))
    return float(delta_values[best_index]), float(scores[best_index])


def estimate_delta_candidates_by_invariant_cross_correlation(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta_min: float,
    delta_max: float,
    min_overlap: float,
) -> list[tuple[float, float]]:
    grid1, features1 = resample_rotation_invariant_features(t1, xy1)
    grid2, features2 = resample_rotation_invariant_features(t2, xy2)
    corr = np.zeros(len(grid1) + len(grid2) - 1, dtype=float)
    for feature_index in range(features1.shape[1]):
        corr += correlate(
            features1[:, feature_index],
            features2[:, feature_index],
            mode="full",
            method="fft",
        )
    overlap = correlate(
        np.ones(len(grid1), dtype=float),
        np.ones(len(grid2), dtype=float),
        mode="full",
        method="fft",
    )
    lags = correlation_lags(len(grid1), len(grid2), mode="full")
    delta_values = grid2[0] - grid1[0] - lags * CORRELATION_DT
    min_overlap_samples = min_overlap / CORRELATION_DT

    valid = (
        (delta_values >= delta_min)
        & (delta_values <= delta_max)
        & (overlap >= min_overlap_samples)
    )
    if not np.any(valid):
        valid = (delta_values >= delta_min) & (delta_values <= delta_max)
    if not np.any(valid):
        raise ValueError("No feasible time-offset candidate is available for cross-correlation.")

    scores = np.full_like(corr, -np.inf, dtype=float)
    scores[valid] = corr[valid] / np.maximum(overlap[valid], 1.0)
    candidates: list[tuple[float, float]] = []
    for index in np.argsort(scores)[::-1]:
        score = float(scores[index])
        if not np.isfinite(score):
            break
        delta = float(delta_values[index])
        if any(abs(delta - old_delta) < DELTA_CANDIDATE_MIN_SEPARATION_S for old_delta, _ in candidates):
            continue
        candidates.append((delta, score))
        if len(candidates) >= DELTA_CANDIDATE_COUNT:
            break
    if not candidates:
        raise ValueError("No feasible time-offset candidate is available for cross-correlation.")
    return candidates


def estimate_rotation_translation(p1: np.ndarray, p2: np.ndarray) -> tuple[float, np.ndarray]:
    center1 = p1.mean(axis=0)
    center2 = p2.mean(axis=0)
    p1_centered = p1 - center1
    p2_centered = p2 - center2
    cos_term = float(np.sum(p1_centered[:, 0] * p2_centered[:, 0] + p1_centered[:, 1] * p2_centered[:, 1]))
    sin_term = float(np.sum(-p1_centered[:, 0] * p2_centered[:, 1] + p1_centered[:, 1] * p2_centered[:, 0]))
    theta = float(np.arctan2(sin_term, cos_term))
    translation = center1 - rotate_xy(center2.reshape(1, 2), theta)[0]
    return theta, translation


def bounded_delta(raw_delta: float, center: float, radius: float) -> float:
    return float(center + radius * np.tanh(raw_delta))


def raw_delta_from_value(delta: float, center: float, radius: float) -> float:
    scaled = np.clip((delta - center) / radius, -0.95, 0.95)
    return float(np.arctanh(scaled))


def rigid_residual_vector(
    params: np.ndarray,
    delta_center: float,
    delta_radius: float,
    q: np.ndarray,
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
) -> np.ndarray:
    raw_delta, theta, tx, ty = params
    delta = bounded_delta(float(raw_delta), delta_center, delta_radius)
    p1 = interp_xy(t1, xy1, q)
    p2 = interp_xy(t2, xy2, q + delta)
    p2_corrected = transform_method2_xy(p2, theta, np.array([tx, ty]))
    residual = p1 - p2_corrected
    return np.concatenate((residual[:, 0], residual[:, 1]))


def alignment_metrics(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta: float,
    theta: float,
    translation: np.ndarray,
    min_overlap: float,
) -> tuple[float, np.ndarray, float, int, np.ndarray]:
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return np.inf, np.zeros(2), overlap, 0, np.empty((0, 2))

    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    p1 = interp_xy(t1, xy1, q)
    p2 = interp_xy(t2, xy2, q + delta)
    p2_corrected = transform_method2_xy(p2, theta, translation)
    residual = p1 - p2_corrected
    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    return mse, translation, overlap, len(q), residual


def evaluate_initial_alignment_candidate(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta: float,
    min_overlap: float,
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
    mse, _, _, n_compare, _ = alignment_metrics(t1, xy1, t2, xy2, delta, theta, translation, min_overlap)
    return mse, theta, translation, n_compare


def estimate_alignment(t1: np.ndarray, xy1: np.ndarray, t2: np.ndarray, xy2: np.ndarray) -> AlignmentResult:
    delta_min, delta_max, min_overlap, min_duration = feasible_delta_bounds(t1, t2)
    delta_candidates = estimate_delta_candidates_by_invariant_cross_correlation(
        t1, xy1, t2, xy2, delta_min, delta_max, min_overlap
    )
    coordinate_delta, coordinate_score = estimate_delta_by_coordinate_cross_correlation(
        t1, xy1, t2, xy2, delta_min, delta_max, min_overlap
    )
    delta_candidates.append((coordinate_delta, coordinate_score))

    candidate_records = []
    for candidate_delta, candidate_score in delta_candidates:
        candidate_mse, candidate_theta, candidate_translation, candidate_n = evaluate_initial_alignment_candidate(
            t1, xy1, t2, xy2, candidate_delta, min_overlap
        )
        if np.isfinite(candidate_mse):
            candidate_records.append(
                (
                    candidate_mse,
                    candidate_delta,
                    candidate_score,
                    candidate_theta,
                    candidate_translation,
                    candidate_n,
                )
            )
    if not candidate_records:
        raise ValueError("No feasible time-offset candidate survived the Procrustes residual check.")

    _, init_delta, init_score, _, _, _ = min(candidate_records, key=lambda item: item[0])
    local_min = max(delta_min, init_delta - LM_DELTA_RADIUS_S)
    local_max = min(delta_max, init_delta + LM_DELTA_RADIUS_S)

    q_start = max(float(t1[0]), float(t2[0] - local_min))
    q_end = min(float(t1[-1]), float(t2[-1] - local_max))
    if q_end - q_start < min_overlap:
        q_start, q_end, _ = overlap_bounds(t1, t2, init_delta)
    q = np.arange(q_start, q_end + 1e-9, COMPARE_DT)
    if len(q) < 4:
        raise ValueError("Too few overlapping samples for least-squares alignment refinement.")

    p1_initial = interp_xy(t1, xy1, q)
    p2_initial = interp_xy(t2, xy2, q + init_delta)
    init_theta, initial_translation = estimate_rotation_translation(p1_initial, p2_initial)
    delta_center = 0.5 * (local_min + local_max)
    delta_radius = max(1e-6, 0.5 * (local_max - local_min))
    init_raw_delta = raw_delta_from_value(init_delta, delta_center, delta_radius)
    result = least_squares(
        rigid_residual_vector,
        x0=np.array([init_raw_delta, init_theta, initial_translation[0], initial_translation[1]], dtype=float),
        args=(delta_center, delta_radius, q, t1, xy1, t2, xy2),
        method="lm",
        max_nfev=800,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    raw_delta, theta, tx, ty = [float(v) for v in result.x]
    delta = bounded_delta(raw_delta, delta_center, delta_radius)
    theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
    bias = np.array([tx, ty], dtype=float)
    mse, bias, overlap, n_compare, residual = alignment_metrics(t1, xy1, t2, xy2, delta, theta, bias, min_overlap)
    lo, hi, _ = overlap_bounds(t1, t2, delta)
    return AlignmentResult(
        delta_t2_minus_t1=delta,
        cross_correlation_initial_delta=init_delta,
        cross_correlation_score=init_score,
        delta_candidate_count=len(candidate_records),
        procrustes_initial_theta_rad=init_theta,
        theta_rad=theta,
        bias_add_to_method2=bias,
        objective_mse=mse,
        residual_rmse=float(np.sqrt(mse)),
        overlap_start=lo,
        overlap_end=hi,
        overlap_ratio=overlap / min_duration,
        n_compare=n_compare,
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
    xy2_corrected: np.ndarray,
    delta: float,
    start: float,
) -> np.ndarray:
    h = 0.5
    p0 = 0.5 * (interp_xy(t1, xy1, np.array([start]))[0] + interp_xy(t2, xy2_corrected, np.array([start + delta]))[0])
    p1 = 0.5 * (
        interp_xy(t1, xy1, np.array([start + h]))[0]
        + interp_xy(t2, xy2_corrected, np.array([start + h + delta]))[0]
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
        rows.append(
            {
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
            }
        )

    return pd.DataFrame(rows)


def calculate_post_alignment_rmse(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    alignment: AlignmentResult,
) -> dict[str, float | int | str]:
    t_min = max(float(t1.min()), float(t2.min()))
    t_max = min(float(t1.max()), float(t2.max()))
    mask = (t1 >= t_min) & (t1 <= t_max)
    t_common = t1[mask]
    xy1_common = xy1[mask]

    method2_query_time = t_common + alignment.delta_t2_minus_t1
    valid = (method2_query_time >= t2[0]) & (method2_query_time <= t2[-1])
    if int(valid.sum()) < 2:
        return {
            "post_alignment_rmse_m": float("nan"),
            "post_alignment_rmse_points": int(valid.sum()),
            "post_alignment_rmse_definition": "RMSE between method1 raw positions and time-aligned, rotated-translated method2 raw positions",
        }

    xy2_interp = interp_xy(t2, xy2, method2_query_time[valid])
    xy2_corrected = transform_method2_xy(xy2_interp, alignment.theta_rad, alignment.bias_add_to_method2)
    residual = xy1_common[valid] - xy2_corrected
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return {
        "post_alignment_rmse_m": rmse,
        "post_alignment_rmse_points": int(valid.sum()),
        "post_alignment_rmse_definition": "RMSE between method1 raw positions and time-aligned, rotated-translated method2 raw positions",
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    t1, xy1_raw, t2, xy2_raw = load_attachment2()

    alignment = estimate_alignment(t1, xy1_raw, t2, xy2_raw)

    r1 = estimate_measurement_covariance(xy1_raw, xy1_raw)
    r2 = estimate_measurement_covariance(xy2_raw, xy2_raw)

    q_time = np.arange(alignment.overlap_start, alignment.overlap_end + 1e-9, DT_OUT)
    xy2_corrected_for_accel = transform_method2_xy(xy2_raw, alignment.theta_rad, alignment.bias_add_to_method2)
    fused_for_accel = 0.5 * (
        interp_xy(t1, xy1_raw, q_time)
        + interp_xy(t2, xy2_corrected_for_accel, q_time + alignment.delta_t2_minus_t1)
    )
    accel_std = estimate_process_accel_std(q_time, fused_for_accel)

    trajectory = run_kalman_filter(t1, xy1_raw, t2, xy2_raw, alignment, r1, r2, accel_std)
    post_alignment_rmse = calculate_post_alignment_rmse(t1, xy1_raw, t2, xy2_raw, alignment)
    trajectory_path = OUT_DIR / "problem2_trajectory_10hz_kalman_A_theta_lm.csv"
    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    summary = {
        "source_workbook": str(DATA_PATH),
        "workflow": "standardization -> rotation-invariant cross-correlation delta candidates plus coordinate cross-correlation anchor -> Procrustes initial theta and translation bias -> bounded LM joint refinement of delta_t, theta, and translation bias without Savitzky-Golay smoothing -> Kalman fusion",
        "smoothing_used": False,
        "rotation_theta_estimated": True,
        "delta_refine_radius_s": LM_DELTA_RADIUS_S,
        "delta_t2_minus_t1_s": alignment.delta_t2_minus_t1,
        "cross_correlation_initial_delta_s": alignment.cross_correlation_initial_delta,
        "delta_candidate_count": alignment.delta_candidate_count,
        "procrustes_initial_theta_rad": alignment.procrustes_initial_theta_rad,
        "procrustes_initial_theta_deg": float(np.rad2deg(alignment.procrustes_initial_theta_rad)),
        "cross_correlation_score": alignment.cross_correlation_score,
        "theta_rad": alignment.theta_rad,
        "theta_deg": float(np.rad2deg(alignment.theta_rad)),
        "method1_time_bias_s": 0.0,
        "method2_time_bias_s": alignment.delta_t2_minus_t1,
        "bias_definition": "method2_corrected = R(theta) @ method2 + bias_add_to_method2 after time alignment",
        "bias_add_to_method2_x_m": float(alignment.bias_add_to_method2[0]),
        "bias_add_to_method2_y_m": float(alignment.bias_add_to_method2[1]),
        "alignment_objective_mse": alignment.objective_mse,
        "alignment_residual_rmse_m": alignment.residual_rmse,
        **post_alignment_rmse,
        "overlap_start_method1_time_s": alignment.overlap_start,
        "overlap_end_method1_time_s": alignment.overlap_end,
        "overlap_ratio": alignment.overlap_ratio,
        "alignment_compare_points": alignment.n_compare,
        "kalman_state": "[x, y, vx, vy]",
        "kalman_dt_output_s": DT_OUT,
        "process_accel_std_mps2": accel_std,
        "measurement_cov_method1": r1.tolist(),
        "measurement_cov_method2": r2.tolist(),
        "data_cleaning": CLEANING_REPORT,
        "trajectory_rows_10hz": int(len(trajectory)),
        "trajectory_csv": str(trajectory_path),
    }
    summary_path = OUT_DIR / "problem2_kalman_summary_A_theta_lm.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
