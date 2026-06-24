import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from controller.fiber_coupling import FiberCoupling

from configuration import PICOSCOPE_RANGE

# --------------------- Settings ---------------------

# "No information" center. This is the middle of the 0-4095 servo range.
CENTER_POS = np.asarray([2048, 2048, 2048, 2048, 2048], dtype=float)
LOAD_EXISTING_BROAD_SCAN = False

SERVO_MIN = 0
SERVO_MAX = 4095

# Optional manual reference for reporting only.
# Set to None if you do not want percentages relative to manual best.
MANUAL_REFERENCE_VOLTAGE = None

# Output folder
EXPERIMENT_NAME = "fc_large_misalignment"

# Broad range: CENTER_POS +- RANGE_BROAD.
# With CENTER_POS=2048 and RANGE_BROAD=2000, this searches approx 48..4048.
RANGE_BROAD = np.asarray([2000, 2000, 2000, 2000, 2000], dtype=float)

BROAD_GLOBAL_SAMPLES = 3000

# How many measured points from broad scan to consider for clustering.
BROAD_TOP_N_FOR_CLUSTERING = 30

# How many distinct high-voltage clusters to keep.
N_CLUSTERS = 5

# Minimum distance between cluster representatives in full 5D servo space.
# Increase this if top clusters are still essentially the same point.
CLUSTER_DISTANCE_STEPS = 1000


# Medium search box around each cluster center.
MEDIUM_RANGE = np.asarray([500, 500, 500, 500, 500], dtype=float)

MEDIUM_OPT_CONFIG = {
    "global_samples": 250,
    "bo_iterations": 30,
    "local_step": 40,
    "local_z_step": 30,
    "local_rounds": 5,
    "validation_measurements": 5,
}


FINE_RANGE = np.asarray([200, 200, 200, 200, 200], dtype=float)

FINE_OPT_CONFIG = {
    "global_samples": 100,
    "bo_iterations": 30,
    "local_step": 20,
    "local_z_step": 10,
    "local_rounds": 6,
    "validation_measurements": 10,
}

DATASETS_FOLDER = "datasets"



# Helpers

def now_output_dir():
    date_folder = datetime.now().strftime("%Y-%m-%d")
    time_folder = datetime.now().strftime("%H-%M-%S")
    out = Path("Data") / date_folder / f"{EXPERIMENT_NAME}_{time_folder}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clip_position(x: np.ndarray):
    return np.clip(np.asarray(x, dtype=float), SERVO_MIN, SERVO_MAX)


def make_bounds(center: np.ndarray, radius: np.ndarray):
    center = np.asarray(center, dtype=float).reshape(-1)
    radius = np.asarray(radius, dtype=float).reshape(-1)
    if len(center) != 5 or len(radius) != 5:
        raise ValueError("center and radius must both be 5D arrays.")
    return clip_position(center - radius), clip_position(center + radius)


def history_to_dataframe(history: List[Dict]):
    rows = []
    for item in history:
        row = dict(item)
        for key in ["x_real", "x_full", "best_x"]:
            if key in row and row[key] is not None:
                arr = np.asarray(row.pop(key), dtype=float).reshape(-1)
                for i, value in enumerate(arr):
                    suffix = "z" if i == 4 else str(i)
                    row[f"{key}_{suffix}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)




def load_stage_dataset(csv_path: Path):
    return  pd.read_csv(csv_path)


def get_position_columns(df: pd.DataFrame):
    if "z" in df.columns:
        return ["m0", "m1", "m2", "m3", "z"]
    return ["m0", "m1", "m2", "m3"]


def greedy_top_clusters(df: pd.DataFrame, top_n: int, n_clusters: int, min_distance: float):
    """
    Greedy clustering by distance: Sort points by measured voltage descending and keep a point if it is at least min_distance away from already selected centers.
    """
    if "voltage_mV" not in df.columns:
        raise ValueError("Dataset must contain a voltage.")

    pos_cols = get_position_columns(df)
    candidates = df.sort_values("voltage_mV", ascending=False).head(top_n).copy()

    selected_rows = []
    selected_positions = []

    for _, row in candidates.iterrows():
        pos = row[pos_cols].values.astype(float)
        if not selected_positions:
            selected_rows.append(row)
            selected_positions.append(pos)
        else:
            distances = [np.linalg.norm(pos - p) for p in selected_positions]
            if min(distances) >= min_distance:
                selected_rows.append(row)
                selected_positions.append(pos)

        if len(selected_rows) >= n_clusters:
            break

    # if clustering distance was too strict, put in  top unused points.
    if len(selected_rows) < n_clusters:
        selected_indices = {int(r.name) for r in selected_rows}
        for idx, row in candidates.iterrows():
            if int(idx) not in selected_indices:
                selected_rows.append(row)
                if len(selected_rows) >= n_clusters:
                    break

    clusters = pd.DataFrame(selected_rows).reset_index(drop=True)
    clusters.insert(0, "cluster_id", np.arange(1, len(clusters) + 1))

    ordered = ["cluster_id"] + pos_cols + ["voltage_mV"]
    remaining = [c for c in clusters.columns if c not in ordered]
    clusters = clusters[ordered + remaining]

    return clusters


def ensure_subdirs(root: Path) -> Dict[str, Path]:
    folders = {
        "datasets": root / DATASETS_FOLDER,
            }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders



def run_optimization_stage(stage: str, region_id: int, center: np.ndarray, radius: np.ndarray, config: Dict, output_root: Path):
    stage_dir = output_root / stage / f"region_{region_id:02d}"
    subdirs = ensure_subdirs(stage_dir)

    min_b, max_b = make_bounds(center, radius)

    save_json(stage_dir / "stage_config.json", {
        "stage": stage,
        "region_id": region_id,
        "center": np.asarray(center, dtype=float).tolist(),
        "radius": np.asarray(radius, dtype=float).tolist(),
        "min_boundary": min_b.tolist(),
        "max_boundary": max_b.tolist(),
        "config": config,
        "dataset_csv": str(subdirs["datasets"] / "global_scan_dataset.csv"),
        "standardized_dataset_csv": str(subdirs["datasets"] / "global_scan_dataset_standardized.csv"),
        "plots_dir": str(subdirs["plots"]),
        "gp_dir": str(subdirs["gp"]),
    })

    pd.DataFrame([{
        "stage": stage,
        "region_id": int(region_id),
        "center_m0": float(center[0]),
        "center_m1": float(center[1]),
        "center_m2": float(center[2]),
        "center_m3": float(center[3]),
        "center_z": float(center[4]),
        "radius_m0": float(radius[0]),
        "radius_m1": float(radius[1]),
        "radius_m2": float(radius[2]),
        "radius_m3": float(radius[3]),
        "radius_z": float(radius[4]),
    }]).to_csv(subdirs["datasets"] / "region_center_and_radius.csv", index=False)

    print("\n" + "=" * 80)
    print(f"{stage.upper()} REGION {region_id}")
    print("=" * 80)
    print("Center:", np.round(center).astype(int))
    print("Min boundary:", np.round(min_b).astype(int))
    print("Max boundary:", np.round(max_b).astype(int))

    dataset_csv = subdirs["datasets"] / "global_scan_dataset.csv"

    fc = FiberCoupling(
        csv_path=str(dataset_csv),
        settle_time=1,
        oversampling=10,
        min_boundary=min_b,
        max_boundary=max_b,
        center=np.round(center).astype(int),
    )

    t0 = time.time()
    final_x, final_voltage, final_std = fc.run_full_optimization(
        global_samples=int(config["global_samples"]),
        bo_iterations=int(config["bo_iterations"]),
        local_step=float(config["local_step"]),
        local_z_step=float(config["local_z_step"]),
        local_rounds=int(config["local_rounds"]),
        validation_measurements=int(config["validation_measurements"]),
        load_global_scan=False,
    )
    duration = time.time() - t0

    final_x = np.asarray(final_x, dtype=float).reshape(-1)
    if len(final_x) == 4:
        final_x = np.append(final_x, center[4])
    final_x = clip_position(final_x)

    history_df = history_to_dataframe(fc.history)
    history_csv = subdirs["datasets"] / "optimization_history.csv"
    history_df.to_csv(history_csv, index=False)

    ref_percent = float("nan")
    if MANUAL_REFERENCE_VOLTAGE is not None and MANUAL_REFERENCE_VOLTAGE > 0:
        ref_percent = 100.0 * float(final_voltage) / float(MANUAL_REFERENCE_VOLTAGE)

    result = {
        "stage": stage,
        "region_id": int(region_id),
        "center_x0": float(center[0]),
        "center_x1": float(center[1]),
        "center_x2": float(center[2]),
        "center_x3": float(center[3]),
        "center_x4_z": float(center[4]),
        "final_x0": float(final_x[0]),
        "final_x1": float(final_x[1]),
        "final_x2": float(final_x[2]),
        "final_x3": float(final_x[3]),
        "final_x4_z": float(final_x[4]),
        "final_voltage": float(final_voltage),
        "final_std": float(final_std),
        "reference_percent": float(ref_percent),
        "duration_s": float(duration),
        "run_dir": str(stage_dir),
        "history_csv": str(history_csv),
        "dataset_csv": str(dataset_csv),
    }
    pd.DataFrame([result]).to_csv(stage_dir / "stage_result.csv", index=False)

    print(f"{stage} region {region_id} final voltage: {final_voltage:.3f} ± {final_std:.3f} mV")
    if not np.isnan(ref_percent):
        print(f"Reference percent: {ref_percent:.2f}%")
    print(f"Duration: {duration/60:.2f} min")
    print("Dataset:", dataset_csv)
    print("GP diagnostics:", subdirs["gp"])

    return result, history_df


def main():
    output_root = now_output_dir()
    print("Output folder:", output_root)

    broad_min, broad_max = make_bounds(CENTER_POS, RANGE_BROAD)

    save_json(output_root / "experiment_config.json", {
        "center_pos": CENTER_POS.tolist(),
        "range_broad": RANGE_BROAD.tolist(),
        "broad_min": broad_min.tolist(),
        "broad_max": broad_max.tolist(),
        "broad_global_samples": BROAD_GLOBAL_SAMPLES,
        "broad_top_n_for_clustering": BROAD_TOP_N_FOR_CLUSTERING,
        "n_clusters": N_CLUSTERS,
        "cluster_distance_steps": CLUSTER_DISTANCE_STEPS,
        "medium_range": MEDIUM_RANGE.tolist(),
        "medium_opt_config": MEDIUM_OPT_CONFIG,
        "fine_range": FINE_RANGE.tolist(),
        "fine_opt_config": FINE_OPT_CONFIG,
        "manual_reference_voltage": MANUAL_REFERENCE_VOLTAGE,
        "picoscope_range": PICOSCOPE_RANGE,
    })

    # LHS scan only
    print("\n" + "=" * 80)
    print("STAGE 1: MASSIVE BROAD LHS SCAN")
    print("=" * 80)
    print("Broad center:", CENTER_POS)
    print("Broad min:", broad_min)
    print("Broad max:", broad_max)

    broad_dir = output_root / "broad_scan"
    broad_subdirs = ensure_subdirs(broad_dir)

    fc_broad = FiberCoupling(
        csv_path=str(broad_subdirs["datasets"] / "broad_global_scan_dataset.csv"),
        settle_time=1,
        oversampling=10,
        min_boundary=broad_min,
        max_boundary=broad_max,
    )

    if LOAD_EXISTING_BROAD_SCAN:

        broad_csv = r"C:\Users\eqela\Desktop\fiber_coupling\Data\2026-06-09\broad_global_scan_dataset.csv"

        broad_df = pd.read_csv(broad_csv)
        pos_cols = get_position_columns(broad_df)
        X_broad = broad_df[pos_cols].values
        y_broad = broad_df["voltage_mV"].values
        broad_df.to_csv(broad_subdirs["datasets"] / "broad_global_scan_loaded_standardized.csv", index=False)

        broad_duration = 0

        print(f"Loaded existing broad scan: {broad_csv}")
        print("Dataset shape:", X_broad.shape)

    else:

        fc_broad.initialize_hardware()

        t0 = time.time()

        X_broad, y_broad = fc_broad.run_global_scan(
            n_samples=BROAD_GLOBAL_SAMPLES,
            load_only=False,
            include_z=True
        )

        fc_broad.close_hardware()

        broad_duration = time.time() - t0
        broad_df = pd.read_csv(broad_subdirs["datasets"] / "broad_global_scan_dataset.csv")

    broad_best_idx = int(np.argmax(y_broad))
    broad_best_x = np.asarray(X_broad[broad_best_idx], dtype=float).reshape(-1)
    if len(broad_best_x) == 4:
        broad_best_x = np.append(broad_best_x, CENTER_POS[4])
    broad_best_y = float(y_broad[broad_best_idx])

    print("Broad scan best position:", broad_best_x)
    print(f"Broad scan best voltage: {broad_best_y:.3f} mV")
    print(f"Broad scan duration: {broad_duration/60:.2f} min")

    # Cluster best measured points
    clusters_df = greedy_top_clusters(
        broad_df,
        top_n=BROAD_TOP_N_FOR_CLUSTERING,
        n_clusters=N_CLUSTERS,
        min_distance=CLUSTER_DISTANCE_STEPS,
    )
    if "z" not in clusters_df.columns:
        clusters_df["z"] = CENTER_POS[4]

    clusters_csv = broad_subdirs["datasets"] / "broad_top_cluster_centers.csv"
    clusters_df.to_csv(clusters_csv, index=False)
    print("\nSelected cluster centers:")
    print(clusters_df[["cluster_id", "m0", "m1", "m2", "m3", "z", "voltage_mV"]])

    # Medium optimization around each cluster

    medium_results = []
    for itera, row in clusters_df.iterrows():
        region_id = int(row["cluster_id"])
        center = row[["m0", "m1", "m2", "m3", "z"]].values.astype(float)
        result, _ = run_optimization_stage(
                stage="medium",
                region_id=region_id,
                center=center,
                radius=MEDIUM_RANGE,
                config=MEDIUM_OPT_CONFIG,
                output_root=output_root,
        )
        medium_results.append(result)

    medium_df = pd.DataFrame(medium_results)

    medium_df.to_csv(output_root / "medium_results.csv", index=False)

    if medium_df.empty:
        raise RuntimeError("No medium results were completed.")

    best_medium = medium_df.loc[medium_df["final_voltage"].idxmax()]
    best_medium_center = np.asarray([
        best_medium["final_x0"],
        best_medium["final_x1"],
        best_medium["final_x2"],
        best_medium["final_x3"],
        best_medium["final_x4_z"],
    ], dtype=float)

    print("\nBest medium result:")
    print(best_medium[["region_id", "final_voltage", "final_std", "reference_percent"]])
    print("Best medium position:", best_medium_center)

    # Fine optimization around best medium result
    fine_result, _ = run_optimization_stage(
        stage="fine",
        region_id=int(best_medium["region_id"]),
        center=best_medium_center,
        radius=FINE_RANGE,
        config=FINE_OPT_CONFIG,
        output_root=output_root,
    )

    print("\n" + "=" * 80)
    print("LARGE-MISALIGNMENT EXPERIMENT COMPLETE")
    print("=" * 80)
    print("Output:", output_root)
    print("Cluster centers:", clusters_csv)
    print("All results:", output_root / "all_stage_results.csv")


if __name__ == "__main__":
    main()
