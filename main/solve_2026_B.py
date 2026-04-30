from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

DT_OUT = 0.1
COMPARE_DT = 0.25

SHOOT_DISTANCE = (5.0, 30.0)
SHOOT_MAX_SPEED = 2.0
SHOOT_MAX_ACCEL = 1.5
SHOOT_PREP = 1.5

PHOTO_DISTANCE = (10.0, 40.0)
PHOTO_MAX_SPEED = 1.5
PHOTO_MAX_ACCEL = 1.5
PHOTO_PREP = 0.5
PHOTO_MIN_ANGLE_DEG = 60.0

# The raw data in attachment 3 has visible measurement noise.  This window is
# used only for task-feasibility derivatives, not for estimating sensor offsets.
TASK_SMOOTH_WINDOW = 251


@dataclass
class AlignResult:
    delta_t2_minus_t1: float
    bias_add_to_method2: np.ndarray
    residual_rmse: float
    overlap_seconds: float
    n_compare: int
    system_bias_exists: bool
    bias_threshold_95: float


@dataclass
class Candidate:
    target_id: str
    task: str
    prep_start: float
    exec_time: float
    angle_deg: float | None = None


def load_position_book(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m1 = pd.read_excel(path, sheet_name="方式1(4Hz)").to_numpy(float)
    m2 = pd.read_excel(path, sheet_name="方式2(5Hz)").to_numpy(float)
    return m1[:, 0], m1[:, 1:3], m2[:, 0], m2[:, 1:3]


def interp_xy(t: np.ndarray, xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack((np.interp(q, t, xy[:, 0]), np.interp(q, t, xy[:, 1])))


def moving_average_xy(xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return xy.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(xy.shape[1])]
    )


def overlap_bounds(
    t1: np.ndarray, t2: np.ndarray, delta: float
) -> tuple[float, float, float]:
    lo = max(float(t1[0]), float(t2[0] - delta))
    hi = min(float(t1[-1]), float(t2[-1] - delta))
    return lo, hi, hi - lo


def alignment_objective(
    t1: np.ndarray,
    xy1: np.ndarray,
    t2: np.ndarray,
    xy2: np.ndarray,
    delta: float,
    debias: bool,
    min_overlap: float,
) -> tuple[float, np.ndarray, float, int, np.ndarray]:
    lo, hi, overlap = overlap_bounds(t1, t2, delta)
    if overlap < min_overlap:
        return float("inf"), np.zeros(2), overlap, 0, np.zeros(2)

    q = np.arange(lo, hi + 1e-9, COMPARE_DT)
    diff = interp_xy(t1, xy1, q) - interp_xy(t2, xy2, q + delta)
    bias = diff.mean(axis=0) if debias else np.zeros(2)
    residual = diff - bias
    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    return mse, bias, overlap, len(q), residual


def estimate_alignment(path: Path, debias: bool) -> AlignResult:
    t1, xy1, t2, xy2 = load_position_book(path)
    min_overlap = max(30.0, 0.55 * min(t1[-1] - t1[0], t2[-1] - t2[0]))

    d_lo = float(t2[0] - t1[-1] + min_overlap)
    d_hi = float(t2[-1] - t1[0] - min_overlap)
    if d_lo >= d_hi:
        raise ValueError(f"No feasible overlap interval for {path.name}")

    coarse = np.arange(d_lo, d_hi + 1e-9, 1.0)
    scores = np.array(
        [
            alignment_objective(t1, xy1, t2, xy2, d, debias, min_overlap)[0]
            for d in coarse
        ]
    )
    seeds = coarse[np.argsort(scores)[: min(5, len(coarse))]]

    best_delta = float(seeds[0])
    best_score = float("inf")
    for seed in seeds:
        delta = float(seed)
        for step in (0.2, 0.05, 0.01, 0.002):
            grid = np.arange(delta - 2.0, delta + 2.0 + step / 2, step)
            local_scores = np.array(
                [
                    alignment_objective(t1, xy1, t2, xy2, d, debias, min_overlap)[0]
                    for d in grid
                ]
            )
            delta = float(grid[int(np.argmin(local_scores))])
        score = alignment_objective(t1, xy1, t2, xy2, delta, debias, min_overlap)[0]
        if score < best_score:
            best_score = score
            best_delta = delta

    mse, bias, overlap, n, residual = alignment_objective(
        t1, xy1, t2, xy2, best_delta, debias, min_overlap
    )
    rmse = math.sqrt(mse)

    if debias and n > 1:
        std = residual.std(axis=0, ddof=1)
        threshold = 2.0 * float(np.linalg.norm(std / math.sqrt(n)))
        exists = float(np.linalg.norm(bias)) > threshold
    else:
        threshold = 0.0
        exists = False

    return AlignResult(
        delta_t2_minus_t1=best_delta,
        bias_add_to_method2=bias,
        residual_rmse=rmse,
        overlap_seconds=overlap,
        n_compare=n,
        system_bias_exists=exists,
        bias_threshold_95=threshold,
    )


def build_trajectory(
    path: Path, align: AlignResult, use_bias: bool, smooth_window: int = 1
) -> pd.DataFrame:
    t1, xy1, t2, xy2 = load_position_book(path)
    lo, hi, _ = overlap_bounds(t1, t2, align.delta_t2_minus_t1)
    q_abs = np.arange(lo, hi + 1e-9, DT_OUT)

    p1 = interp_xy(t1, xy1, q_abs)
    p2 = interp_xy(t2, xy2, q_abs + align.delta_t2_minus_t1)
    if use_bias:
        p2 = p2 + align.bias_add_to_method2

    fused = 0.5 * (p1 + p2)
    smooth = moving_average_xy(fused, smooth_window)
    rel_time = q_abs - q_abs[0]
    return pd.DataFrame(
        {
            "time_s": rel_time,
            "method1_time_s": q_abs,
            "method2_time_s": q_abs + align.delta_t2_minus_t1,
            "x_method1_m": p1[:, 0],
            "y_method1_m": p1[:, 1],
            "x_method2_corrected_m": p2[:, 0],
            "y_method2_corrected_m": p2[:, 1],
            "x_fused_m": fused[:, 0],
            "y_fused_m": fused[:, 1],
            "x_smooth_m": smooth[:, 0],
            "y_smooth_m": smooth[:, 1],
        }
    )


def valid_window(mask: np.ndarray, idx: int, prep_points: int) -> bool:
    start = idx - prep_points
    return start >= 0 and bool(mask[start : idx + 1].all())


def circular_angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def generate_task_candidates(traj: pd.DataFrame) -> tuple[list[Candidate], list[Candidate]]:
    xy = traj[["x_smooth_m", "y_smooth_m"]].to_numpy(float)
    t = traj["time_s"].to_numpy(float)
    dt = float(np.median(np.diff(t)))

    vel = np.gradient(xy, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(np.gradient(vel, dt, axis=0), axis=1)

    shoot_targets = pd.read_excel(ROOT / "附件4.xlsx", sheet_name="射击目标")
    photo_targets = pd.read_excel(ROOT / "附件4.xlsx", sheet_name="拍照目标")

    shoot_base = (speed <= SHOOT_MAX_SPEED) & (accel <= SHOOT_MAX_ACCEL)
    photo_base = (speed <= PHOTO_MAX_SPEED) & (accel <= PHOTO_MAX_ACCEL)
    shoot_prep_points = int(round(SHOOT_PREP / dt))
    photo_prep_points = int(round(PHOTO_PREP / dt))
    sample_step = max(1, int(round(0.5 / dt)))

    shoot_candidates: list[Candidate] = []
    for _, row in shoot_targets.iterrows():
        target = np.array([row["X坐标(m)"], row["Y坐标(m)"]], dtype=float)
        dist = np.linalg.norm(xy - target, axis=1)
        mask = shoot_base & (dist >= SHOOT_DISTANCE[0]) & (dist <= SHOOT_DISTANCE[1])
        for idx in range(0, len(t), sample_step):
            if valid_window(mask, idx, shoot_prep_points):
                shoot_candidates.append(
                    Candidate(
                        target_id=str(row["编号"]),
                        task="射击",
                        prep_start=float(t[idx] - SHOOT_PREP),
                        exec_time=float(t[idx]),
                    )
                )

    photo_candidates: list[Candidate] = []
    for _, row in photo_targets.iterrows():
        target = np.array([row["X坐标(m)"], row["Y坐标(m)"]], dtype=float)
        vec = xy - target
        dist = np.linalg.norm(vec, axis=1)
        angle = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
        mask = photo_base & (dist >= PHOTO_DISTANCE[0]) & (dist <= PHOTO_DISTANCE[1])
        for idx in range(0, len(t), sample_step):
            if valid_window(mask, idx, photo_prep_points):
                photo_candidates.append(
                    Candidate(
                        target_id=str(row["编号"]),
                        task="拍照",
                        prep_start=float(t[idx] - PHOTO_PREP),
                        exec_time=float(t[idx]),
                        angle_deg=float(angle[idx]),
                    )
                )

    return shoot_candidates, photo_candidates


def interval_is_free(selected: list[Candidate], cand: Candidate) -> bool:
    for item in selected:
        if cand.prep_start < item.exec_time and item.prep_start < cand.exec_time:
            return False
    return True


def schedule_tasks(
    shoot_candidates: list[Candidate], photo_candidates: list[Candidate]
) -> list[Candidate]:
    selected: list[Candidate] = []
    shot_targets: set[str] = set()
    photo_angles: dict[str, list[float]] = {}

    candidates = shoot_candidates + photo_candidates
    for cand in sorted(candidates, key=lambda c: (c.exec_time, 0 if c.task == "拍照" else 1, c.target_id)):
        if not interval_is_free(selected, cand):
            continue
        if cand.task == "射击":
            if cand.target_id in shot_targets:
                continue
            selected.append(cand)
            shot_targets.add(cand.target_id)
        else:
            used = photo_angles.setdefault(cand.target_id, [])
            assert cand.angle_deg is not None
            if any(circular_angle_diff(cand.angle_deg, a) < PHOTO_MIN_ANGLE_DEG for a in used):
                continue
            if len(used) >= int(360 // PHOTO_MIN_ANGLE_DEG):
                continue
            selected.append(cand)
            used.append(cand.angle_deg)

    return sorted(selected, key=lambda c: c.exec_time)


def write_result_workbook(tasks: list[Candidate]) -> Path:
    src = ROOT / "result.xlsx"
    dst = OUT_DIR / "result_filled.xlsx"
    wb = load_workbook(src)
    ws = wb.active

    max_row = max(ws.max_row, len(tasks) + 2)
    for row in range(2, max_row + 1):
        for col in range(1, 6):
            ws.cell(row=row, column=col).value = None

    for i, task in enumerate(tasks, start=1):
        row = i + 1
        ws.cell(row=row, column=1, value=i)
        ws.cell(row=row, column=2, value=task.target_id)
        ws.cell(row=row, column=3, value=task.task)
        ws.cell(row=row, column=4, value=round(task.prep_start, 2))
        ws.cell(row=row, column=5, value=round(task.exec_time, 2))

    wb.save(dst)
    return dst


def save_summary(summary: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    jobs = [
        ("problem1", ROOT / "附件1.xlsx", False, 1),
        ("problem2", ROOT / "附件2.xlsx", True, 11),
        ("problem3", ROOT / "附件3.xlsx", True, 11),
    ]

    summary: dict[str, dict] = {}
    trajectories: dict[str, pd.DataFrame] = {}

    for name, path, debias, smooth_window in jobs:
        align = estimate_alignment(path, debias=debias)
        use_bias = bool(debias and align.system_bias_exists)
        traj = build_trajectory(path, align, use_bias=use_bias, smooth_window=smooth_window)
        trajectories[name] = traj
        traj.to_csv(OUT_DIR / f"{name}_trajectory_10hz.csv", index=False, encoding="utf-8-sig")

        summary[name] = {
            "delta_t2_minus_t1_s": round(align.delta_t2_minus_t1, 6),
            "method1_time_bias_s": 0.0,
            "method2_time_bias_s": round(align.delta_t2_minus_t1, 6),
            "bias_add_to_method2_x_m": round(float(align.bias_add_to_method2[0]), 6),
            "bias_add_to_method2_y_m": round(float(align.bias_add_to_method2[1]), 6),
            "system_bias_exists": bool(align.system_bias_exists),
            "bias_threshold_95_m": round(align.bias_threshold_95, 6),
            "residual_rmse_m": round(align.residual_rmse, 6),
            "overlap_seconds": round(align.overlap_seconds, 3),
            "trajectory_rows_10hz": int(len(traj)),
        }

    task_traj = trajectories["problem3"].copy()
    task_traj[["x_smooth_m", "y_smooth_m"]] = moving_average_xy(
        task_traj[["x_fused_m", "y_fused_m"]].to_numpy(float), TASK_SMOOTH_WINDOW
    )
    task_traj.to_csv(OUT_DIR / "problem3_task_trajectory_10hz.csv", index=False, encoding="utf-8-sig")

    shoot_candidates, photo_candidates = generate_task_candidates(task_traj)
    tasks = schedule_tasks(shoot_candidates, photo_candidates)
    result_path = write_result_workbook(tasks)

    summary["problem4"] = {
        "shoot_candidate_count": len(shoot_candidates),
        "photo_candidate_count": len(photo_candidates),
        "scheduled_task_count": len(tasks),
        "scheduled_shoot_count": sum(1 for t in tasks if t.task == "射击"),
        "scheduled_photo_count": sum(1 for t in tasks if t.task == "拍照"),
        "result_workbook": str(result_path),
    }
    save_summary(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
