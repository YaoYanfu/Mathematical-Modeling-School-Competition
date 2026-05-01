from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "visualization"
OUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing" / "problem1"

OUTPUT_HZ = 10.0
OUTPUT_DT = 1.0 / OUTPUT_HZ


def find_attachment(number: int) -> Path:
    matches = sorted(DATA_DIR.glob(f"*{number}.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Could not find attachment {number} in {DATA_DIR}")
    if len(matches) > 1:
        raise FileExistsError(f"Attachment {number} is ambiguous: {matches}")
    return matches[0]


def as_numeric_frame(raw: pd.DataFrame, method: str, nominal_hz: float) -> pd.DataFrame:
    if raw.shape[1] < 3:
        raise ValueError(f"{method} must contain at least three columns: time, x, y")

    df = raw.iloc[:, :3].copy()
    df.columns = ["time_s", "x_m", "y_m"]

    for column in ["time_s", "x_m", "y_m"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    bad_rows = df[df[["time_s", "x_m", "y_m"]].isna().any(axis=1)]
    if not bad_rows.empty:
        raise ValueError(
            f"{method} contains non-numeric or blank values; preprocessing does not drop or fill rows."
        )

    df.insert(0, "method", method)
    df.insert(1, "sample_index", np.arange(len(df), dtype=int))
    df.insert(2, "nominal_hz", nominal_hz)
    df["time_rel_s"] = df["time_s"] - df["time_s"].iloc[0]
    df["dt_s"] = df["time_s"].diff()
    df["dx_m"] = df["x_m"].diff()
    df["dy_m"] = df["y_m"].diff()
    df["segment_distance_m"] = np.hypot(df["dx_m"], df["dy_m"])
    df["speed_fd_mps"] = df["segment_distance_m"] / df["dt_s"]
    return df


def resample_to_10hz(df: pd.DataFrame, method: str, nominal_hz: float) -> pd.DataFrame:
    duration = float(df["time_rel_s"].iloc[-1])
    # Keep the grid inside the observed interval; no extrapolation is introduced.
    n_steps = int(np.floor(duration / OUTPUT_DT + 1e-9))
    grid = np.round(np.arange(n_steps + 1, dtype=float) * OUTPUT_DT, 10)

    out = pd.DataFrame(
        {
            "method": method,
            "nominal_hz": nominal_hz,
            "time_rel_s": grid,
            "time_s": grid + float(df["time_s"].iloc[0]),
            "x_m": np.interp(grid, df["time_rel_s"], df["x_m"]),
            "y_m": np.interp(grid, df["time_rel_s"], df["y_m"]),
        }
    )
    return out


def sampling_summary(df: pd.DataFrame, method: str, nominal_hz: float, resampled_rows: int) -> dict:
    dt = df["dt_s"].dropna().to_numpy(float)
    expected_dt = 1.0 / nominal_hz
    return {
        "method": method,
        "source_rows": int(len(df)),
        "resampled_10hz_rows": int(resampled_rows),
        "raw_time_start_s": round(float(df["time_s"].iloc[0]), 10),
        "raw_time_end_s": round(float(df["time_s"].iloc[-1]), 10),
        "duration_s": round(float(df["time_rel_s"].iloc[-1]), 10),
        "nominal_hz": nominal_hz,
        "expected_dt_s": expected_dt,
        "median_dt_s": round(float(np.median(dt)), 10),
        "min_dt_s": round(float(np.min(dt)), 10),
        "max_dt_s": round(float(np.max(dt)), 10),
        "max_abs_dt_error_s": round(float(np.max(np.abs(dt - expected_dt))), 12),
        "time_is_strictly_increasing": bool(np.all(dt > 0)),
        "has_missing_values": bool(df[["time_s", "x_m", "y_m"]].isna().any().any()),
    }


def write_outputs(frames: dict[str, pd.DataFrame], resampled: dict[str, pd.DataFrame], source_path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames["method1"].to_csv(OUT_DIR / "problem1_method1_standardized.csv", index=False, encoding="utf-8-sig")
    frames["method2"].to_csv(OUT_DIR / "problem1_method2_standardized.csv", index=False, encoding="utf-8-sig")
    pd.concat(frames.values(), ignore_index=True).to_csv(
        OUT_DIR / "problem1_long_standardized.csv", index=False, encoding="utf-8-sig"
    )

    resampled["method1"].to_csv(OUT_DIR / "problem1_method1_10hz_device_time.csv", index=False, encoding="utf-8-sig")
    resampled["method2"].to_csv(OUT_DIR / "problem1_method2_10hz_device_time.csv", index=False, encoding="utf-8-sig")

    common_end = min(float(x["time_rel_s"].iloc[-1]) for x in resampled.values())
    common_grid = np.round(np.arange(int(np.floor(common_end / OUTPUT_DT + 1e-9)) + 1) * OUTPUT_DT, 10)
    common = pd.DataFrame({"time_rel_s": common_grid})
    for method, df in resampled.items():
        prefix = method
        common[f"{prefix}_time_s"] = np.interp(common_grid, df["time_rel_s"], df["time_s"])
        common[f"{prefix}_x_m"] = np.interp(common_grid, df["time_rel_s"], df["x_m"])
        common[f"{prefix}_y_m"] = np.interp(common_grid, df["time_rel_s"], df["y_m"])
    common.to_csv(OUT_DIR / "problem1_alignment_input_10hz_by_device_time.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_workbook": str(source_path),
        "output_hz": OUTPUT_HZ,
        "preprocessing_boundary": {
            "done": [
                "standardized the two source sheets into method/sample/time/x/y columns",
                "converted each device clock to a relative time axis starting at zero",
                "computed finite-difference interval, displacement, and speed features for diagnostics",
                "interpolated each method onto a 10Hz grid within its own observed time span",
                "built a 10Hz alignment input table on device-relative time",
            ],
            "not_done": [
                "no row deletion, filling, outlier filtering, or denoising",
                "no time-offset estimation",
                "no trajectory fusion",
                "no extrapolation outside observed time ranges",
            ],
        },
        "method_summaries": [
            sampling_summary(frames["method1"], "method1", 4.0, len(resampled["method1"])),
            sampling_summary(frames["method2"], "method2", 5.0, len(resampled["method2"])),
        ],
        "outputs": {
            "method1_standardized": str(OUT_DIR / "problem1_method1_standardized.csv"),
            "method2_standardized": str(OUT_DIR / "problem1_method2_standardized.csv"),
            "long_standardized": str(OUT_DIR / "problem1_long_standardized.csv"),
            "method1_10hz": str(OUT_DIR / "problem1_method1_10hz_device_time.csv"),
            "method2_10hz": str(OUT_DIR / "problem1_method2_10hz_device_time.csv"),
            "alignment_input_10hz": str(OUT_DIR / "problem1_alignment_input_10hz_by_device_time.csv"),
        },
    }

    with (OUT_DIR / "problem1_preprocess_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    source_path = find_attachment(1)
    workbook = pd.ExcelFile(source_path)
    if len(workbook.sheet_names) < 2:
        raise ValueError(f"{source_path} must contain two method sheets")

    raw_method1 = pd.read_excel(workbook, sheet_name=workbook.sheet_names[0])
    raw_method2 = pd.read_excel(workbook, sheet_name=workbook.sheet_names[1])

    frames = {
        "method1": as_numeric_frame(raw_method1, method="method1", nominal_hz=4.0),
        "method2": as_numeric_frame(raw_method2, method="method2", nominal_hz=5.0),
    }
    resampled = {
        "method1": resample_to_10hz(frames["method1"], method="method1", nominal_hz=4.0),
        "method2": resample_to_10hz(frames["method2"], method="method2", nominal_hz=5.0),
    }

    write_outputs(frames, resampled, source_path)


if __name__ == "__main__":
    main()
