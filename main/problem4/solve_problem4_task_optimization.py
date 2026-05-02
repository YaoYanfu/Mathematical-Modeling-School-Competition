from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.signal import savgol_filter
from scipy.sparse import lil_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_PATH = PROJECT_ROOT / "outputs" / "问题三" / "problem3_trajectory_10hz_kalman.csv"
TARGET_PATH = PROJECT_ROOT / "数据以及可视化" / "附件4.xlsx"
RESULT_TEMPLATE_PATH = PROJECT_ROOT / "result.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "问题四"

SHOOT_DISTANCE_RANGE_M = (5.0, 30.0)
PHOTO_DISTANCE_RANGE_M = (10.0, 40.0)
SHOOT_SPEED_MAX_MPS = 2.0
PHOTO_SPEED_MAX_MPS = 1.5
ACCEL_MAX_MPS2 = 1.5
SHOOT_PREP_S = 1.5
PHOTO_PREP_S = 0.5
PHOTO_MIN_ANGLE_DIFF_DEG = 60.0
SHOOT_EXPECTED_SCORE = 0.85
PHOTO_SCORE = 1.0

# Position is the submitted 10Hz fused trajectory.  A light smoothing is used
# only for kinematic derivatives, so velocity/acceleration constraints are not
# dominated by numerical differentiation noise.
KINEMATIC_SAVGOL_WINDOW = 101
KINEMATIC_SAVGOL_POLYORDER = 3


@dataclass(frozen=True)
class Candidate:
    task_type: str
    target_id: str
    target_index: int
    start_index: int
    exec_index: int
    start_time: float
    exec_time: float
    angle_deg: float | None
    weight: float


def read_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    shoot = pd.read_excel(TARGET_PATH, sheet_name="射击目标")
    photo = pd.read_excel(TARGET_PATH, sheet_name="拍照目标")
    return shoot, photo


def load_trajectory() -> pd.DataFrame:
    df = pd.read_csv(TRAJECTORY_PATH)
    required = {"time_s", "x_kalman_m", "y_kalman_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory file misses columns: {sorted(missing)}")
    return df


def kinematics(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df["time_s"].to_numpy(float)
    x = df["x_kalman_m"].to_numpy(float)
    y = df["y_kalman_m"].to_numpy(float)
    window = min(KINEMATIC_SAVGOL_WINDOW, len(df) - 1 if (len(df) - 1) % 2 == 1 else len(df) - 2)
    window = max(window, 5)
    x_smooth = savgol_filter(x, window_length=window, polyorder=KINEMATIC_SAVGOL_POLYORDER, mode="interp")
    y_smooth = savgol_filter(y, window_length=window, polyorder=KINEMATIC_SAVGOL_POLYORDER, mode="interp")
    vx = np.gradient(x_smooth, t)
    vy = np.gradient(y_smooth, t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    return np.hypot(vx, vy), np.hypot(ax, ay)


def consecutive_window_ok(mask: np.ndarray, prep_steps: int) -> np.ndarray:
    ok = np.zeros_like(mask, dtype=bool)
    if prep_steps < 0:
        raise ValueError("prep_steps must be non-negative")
    for idx in range(prep_steps, len(mask)):
        ok[idx] = bool(mask[idx - prep_steps : idx + 1].all())
    return ok


def angle_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_candidates(
    trajectory: pd.DataFrame,
    shoot_targets: pd.DataFrame,
    photo_targets: pd.DataFrame,
) -> list[Candidate]:
    t = trajectory["time_s"].to_numpy(float)
    x = trajectory["x_kalman_m"].to_numpy(float)
    y = trajectory["y_kalman_m"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    speed, accel = kinematics(trajectory)

    candidates: list[Candidate] = []
    shoot_prep_steps = int(round(SHOOT_PREP_S / dt))
    photo_prep_steps = int(round(PHOTO_PREP_S / dt))

    shoot_motion_ok = (speed <= SHOOT_SPEED_MAX_MPS) & (accel <= ACCEL_MAX_MPS2)
    photo_motion_ok = (speed <= PHOTO_SPEED_MAX_MPS) & (accel <= ACCEL_MAX_MPS2)

    for target_index, row in shoot_targets.iterrows():
        target_id = str(row.iloc[0])
        tx = float(row.iloc[1])
        ty = float(row.iloc[2])
        distance = np.hypot(x - tx, y - ty)
        feasible = (
            shoot_motion_ok
            & (distance >= SHOOT_DISTANCE_RANGE_M[0])
            & (distance <= SHOOT_DISTANCE_RANGE_M[1])
        )
        window_ok = consecutive_window_ok(feasible, shoot_prep_steps)
        for exec_index in np.where(window_ok)[0]:
            candidates.append(
                Candidate(
                    task_type="射击",
                    target_id=target_id,
                    target_index=int(target_index),
                    start_index=int(exec_index - shoot_prep_steps),
                    exec_index=int(exec_index),
                    start_time=float(t[exec_index] - SHOOT_PREP_S),
                    exec_time=float(t[exec_index]),
                    angle_deg=None,
                    weight=SHOOT_EXPECTED_SCORE,
                )
            )

    for target_index, row in photo_targets.iterrows():
        target_id = str(row.iloc[0])
        tx = float(row.iloc[1])
        ty = float(row.iloc[2])
        distance = np.hypot(x - tx, y - ty)
        feasible = (
            photo_motion_ok
            & (distance >= PHOTO_DISTANCE_RANGE_M[0])
            & (distance <= PHOTO_DISTANCE_RANGE_M[1])
        )
        window_ok = consecutive_window_ok(feasible, photo_prep_steps)
        angles = (np.degrees(np.arctan2(y - ty, x - tx)) + 360.0) % 360.0
        for exec_index in np.where(window_ok)[0]:
            candidates.append(
                Candidate(
                    task_type="拍照",
                    target_id=target_id,
                    target_index=int(target_index),
                    start_index=int(exec_index - photo_prep_steps),
                    exec_index=int(exec_index),
                    start_time=float(t[exec_index] - PHOTO_PREP_S),
                    exec_time=float(t[exec_index]),
                    angle_deg=float(angles[exec_index]),
                    weight=PHOTO_SCORE,
                )
            )

    return candidates


def solve_candidate_selection(candidates: list[Candidate], n_time: int, n_shoot_targets: int) -> list[int]:
    if not candidates:
        return []

    n = len(candidates)
    rows: list[tuple[list[int], list[float], float, float]] = []

    # At any 10Hz instant, the robot can be in at most one task preparation/execution interval.
    active_by_time: list[list[int]] = [[] for _ in range(n_time)]
    for col, cand in enumerate(candidates):
        for idx in range(cand.start_index, cand.exec_index + 1):
            active_by_time[idx].append(col)
    for cols in active_by_time:
        if len(cols) > 1:
            rows.append((cols, [1.0] * len(cols), -np.inf, 1.0))

    # Each shooting target is shot at most once.
    for target_idx in range(n_shoot_targets):
        cols = [
            col
            for col, cand in enumerate(candidates)
            if cand.task_type == "射击" and cand.target_index == target_idx
        ]
        if len(cols) > 1:
            rows.append((cols, [1.0] * len(cols), -np.inf, 1.0))

    # For the same photo target, chosen photo angles must differ by at least 60 degrees.
    by_photo_target: dict[int, list[int]] = {}
    for col, cand in enumerate(candidates):
        if cand.task_type == "拍照":
            by_photo_target.setdefault(cand.target_index, []).append(col)
    for cols in by_photo_target.values():
        for i, left in enumerate(cols):
            angle_left = candidates[left].angle_deg
            assert angle_left is not None
            for right in cols[i + 1 :]:
                angle_right = candidates[right].angle_deg
                assert angle_right is not None
                if angle_difference_deg(angle_left, angle_right) < PHOTO_MIN_ANGLE_DIFF_DEG - 1e-9:
                    rows.append(([left, right], [1.0, 1.0], -np.inf, 1.0))

    matrix = lil_matrix((len(rows), n), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_idx, (cols, vals, lo, hi) in enumerate(rows):
        matrix[row_idx, cols] = vals
        lower[row_idx] = lo
        upper[row_idx] = hi

    result = milp(
        c=-np.array([cand.weight for cand in candidates], dtype=float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 180.0, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    return [idx for idx, value in enumerate(result.x) if value > 0.5]


def selected_table(candidates: list[Candidate], selected_indices: list[int]) -> pd.DataFrame:
    selected = sorted((candidates[idx] for idx in selected_indices), key=lambda item: item.exec_time)
    rows = []
    for order, cand in enumerate(selected, start=1):
        rows.append(
            {
                "序号": order,
                "目标编号": cand.target_id,
                "任务": cand.task_type,
                "开始准备时刻(s)": round(cand.start_time, 2),
                "任务执行时刻(s)": round(cand.exec_time, 2),
                "拍照方向角(deg)": None if cand.angle_deg is None else round(cand.angle_deg, 2),
                "目标函数权重": cand.weight,
            }
        )
    return pd.DataFrame(rows)


def write_result_workbook(table: pd.DataFrame) -> Path:
    wb = load_workbook(RESULT_TEMPLATE_PATH)
    ws = wb.active

    # Preserve the right-side red-text instructions; only clear and fill A:E.
    for row in range(2, max(ws.max_row, len(table) + 1) + 1):
        for col in range(1, 6):
            ws.cell(row=row, column=col).value = None

    for row_offset, (_, row) in enumerate(table.iterrows(), start=2):
        ws.cell(row=row_offset, column=1).value = int(row["序号"])
        ws.cell(row=row_offset, column=2).value = row["目标编号"]
        ws.cell(row=row_offset, column=3).value = row["任务"]
        ws.cell(row=row_offset, column=4).value = float(row["开始准备时刻(s)"])
        ws.cell(row=row_offset, column=5).value = float(row["任务执行时刻(s)"])

    output_copy = OUT_DIR / "result_problem4.xlsx"
    wb.save(output_copy)
    try:
        wb.save(RESULT_TEMPLATE_PATH)
        filled_result = RESULT_TEMPLATE_PATH
    except PermissionError:
        filled_result = output_copy
    return filled_result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory()
    shoot_targets, photo_targets = read_targets()
    candidates = build_candidates(trajectory, shoot_targets, photo_targets)
    selected_indices = solve_candidate_selection(candidates, len(trajectory), len(shoot_targets))
    table = selected_table(candidates, selected_indices)

    output_csv = OUT_DIR / "problem4_selected_tasks.csv"
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_workbook = write_result_workbook(table)

    summary = {
        "trajectory_source": str(TRAJECTORY_PATH),
        "target_source": str(TARGET_PATH),
        "constraints": {
            "shoot_distance_range_m": SHOOT_DISTANCE_RANGE_M,
            "photo_distance_range_m": PHOTO_DISTANCE_RANGE_M,
            "shoot_speed_max_mps": SHOOT_SPEED_MAX_MPS,
            "photo_speed_max_mps": PHOTO_SPEED_MAX_MPS,
            "accel_max_mps2": ACCEL_MAX_MPS2,
            "shoot_prep_s": SHOOT_PREP_S,
            "photo_prep_s": PHOTO_PREP_S,
            "photo_min_angle_diff_deg": PHOTO_MIN_ANGLE_DIFF_DEG,
            "shoot_expected_score": SHOOT_EXPECTED_SCORE,
            "photo_score": PHOTO_SCORE,
        },
        "candidate_count": len(candidates),
        "selected_count": int(len(table)),
        "selected_shoot_count": int((table["任务"] == "射击").sum()) if not table.empty else 0,
        "selected_photo_count": int((table["任务"] == "拍照").sum()) if not table.empty else 0,
        "objective_value": float(table["目标函数权重"].sum()) if not table.empty else 0.0,
        "selected_tasks_csv": str(output_csv),
        "filled_result_workbook": str(output_workbook),
        "result_template_path": str(RESULT_TEMPLATE_PATH),
    }
    summary_path = OUT_DIR / "problem4_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(table[["序号", "目标编号", "任务", "开始准备时刻(s)", "任务执行时刻(s)"]].to_string(index=False))


if __name__ == "__main__":
    main()
