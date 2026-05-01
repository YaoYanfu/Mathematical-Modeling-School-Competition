from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessing" / "problem1"
OUT_DIR = PROJECT_ROOT / "outputs"

ALIGN_DT = 0.1
MIN_OVERLAP_RATIO = 0.70


def build_spline(t: np.ndarray, signal: np.ndarray) -> CubicSpline:
    return CubicSpline(t, signal, bc_type="natural")


def overlap_bounds(t1: np.ndarray, t2: np.ndarray, delta_t: float) -> tuple[float, float]:
    overlap_start = max(float(t1.min()), float(t2.min() - delta_t))
    overlap_end = min(float(t1.max()), float(t2.max() - delta_t))
    return overlap_start, overlap_end


def alignment_objective(
    delta_t: float,
    t1: np.ndarray,
    t2: np.ndarray,
    spline1x: CubicSpline,
    spline1y: CubicSpline,
    spline2x: CubicSpline,
    spline2y: CubicSpline,
) -> float:
    overlap_start, overlap_end = overlap_bounds(t1, t2, delta_t)
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min(float(np.ptp(t1)), float(np.ptp(t2))))
    if overlap_end - overlap_start < min_overlap:
        return np.inf

    ti = np.arange(overlap_start, overlap_end + 1e-9, ALIGN_DT)
    error = (
        (spline1x(ti) - spline2x(ti + delta_t)) ** 2
        + (spline1y(ti) - spline2y(ti + delta_t)) ** 2
    )
    return float(np.mean(error))


def align_time_by_spline(
    t1: np.ndarray,
    t2: np.ndarray,
    spline1x: CubicSpline,
    spline1y: CubicSpline,
    spline2x: CubicSpline,
    spline2y: CubicSpline,
) -> tuple[float, float, float, float, float]:
    min_duration = min(float(np.ptp(t1)), float(np.ptp(t2)))
    min_overlap = max(30.0, MIN_OVERLAP_RATIO * min_duration)
    delta_min = float(t2.min() - t1.max() + min_overlap)
    delta_max = float(t2.max() - t1.min() - min_overlap)

    search_grid = np.linspace(delta_min, delta_max, 1000)
    scores = np.array(
        [
            alignment_objective(delta, t1, t2, spline1x, spline1y, spline2x, spline2y)
            for delta in search_grid
        ]
    )
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("No feasible time offset satisfies the minimum overlap ratio.")

    best_delta = float(search_grid[int(np.argmin(scores))])
    grid_step = float(search_grid[1] - search_grid[0])

    result = minimize_scalar(
        lambda delta: alignment_objective(delta, t1, t2, spline1x, spline1y, spline2x, spline2y),
        bounds=(max(delta_min, best_delta - 2 * grid_step), min(delta_max, best_delta + 2 * grid_step)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    delta_t = float(result.x if result.success else best_delta)
    overlap_start, overlap_end = overlap_bounds(t1, t2, delta_t)
    overlap_ratio = (overlap_end - overlap_start) / min_duration
    mse = alignment_objective(delta_t, t1, t2, spline1x, spline1y, spline2x, spline2y)
    return delta_t, mse, overlap_start, overlap_end, overlap_ratio


def main() -> None:
    df1 = pd.read_csv(PREPROCESS_DIR / "problem1_method1_standardized.csv")
    df2 = pd.read_csv(PREPROCESS_DIR / "problem1_method2_standardized.csv")

    t1 = df1["time_s"].to_numpy(float)
    x1 = df1["x_m"].to_numpy(float)
    y1 = df1["y_m"].to_numpy(float)
    t2 = df2["time_s"].to_numpy(float)
    x2 = df2["x_m"].to_numpy(float)
    y2 = df2["y_m"].to_numpy(float)

    spline1x = build_spline(t1, x1)
    spline1y = build_spline(t1, y1)
    spline2x = build_spline(t2, x2)
    spline2y = build_spline(t2, y2)

    delta_t, align_mse, overlap_start, overlap_end, overlap_ratio = align_time_by_spline(
        t1, t2, spline1x, spline1y, spline2x, spline2y
    )

    time_s = np.arange(overlap_start, overlap_end + 1e-9, ALIGN_DT)
    method2_time_s = time_s + delta_t

    x1_spline = spline1x(time_s)
    y1_spline = spline1y(time_s)
    x2_spline = spline2x(method2_time_s)
    y2_spline = spline2y(method2_time_s)

    trajectory = pd.DataFrame(
        {
            "time_s": time_s,
            "time_rel_s": time_s - time_s[0],
            "method1_time_s": time_s,
            "method2_time_s": method2_time_s,
            "x_method1_spline_m": x1_spline,
            "y_method1_spline_m": y1_spline,
            "x_method2_aligned_spline_m": x2_spline,
            "y_method2_aligned_spline_m": y2_spline,
            "x_average_m": 0.5 * (x1_spline + x2_spline),
            "y_average_m": 0.5 * (y1_spline + y2_spline),
            "position_difference_m": np.sqrt((x1_spline - x2_spline) ** 2 + (y1_spline - y2_spline) ** 2),
        }
    )

    OUT_DIR.mkdir(exist_ok=True)
    output_csv = OUT_DIR / "problem1_trajectory_10hz_fitted_average.csv"
    trajectory.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "method": "cubic_spline_interpolation",
        "delta_t2_minus_t1_s": delta_t,
        "objective_mse": align_mse,
        "objective_rmse_m": float(np.sqrt(align_mse)),
        "overlap_start_method1_time_s": overlap_start,
        "overlap_end_method1_time_s": overlap_end,
        "overlap_ratio": overlap_ratio,
        "dt_s": ALIGN_DT,
        "trajectory_rows": int(len(trajectory)),
        "trajectory_csv": str(output_csv),
        "source_files": {
            "method1": str(PREPROCESS_DIR / "problem1_method1_standardized.csv"),
            "method2": str(PREPROCESS_DIR / "problem1_method2_standardized.csv"),
        },
    }
    output_summary = OUT_DIR / "problem1_trajectory_10hz_fitted_average_summary.json"
    with output_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
