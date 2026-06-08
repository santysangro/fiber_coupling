"""
Hierarchical fiber-coupling optimizer.

1. Broad search:
   Large actuator range around SERVOS_TEST_POS.
   Runs FiberCoupling optimization and extracts the top 5 candidate regions.

2. Medium search:
   Runs a smaller optimization around each of the top 5 broad candidates.
   Selects the best validated result.

3. Fine search:
   Runs a small-range refinement around the best medium result.
   Intended for drift correction / final repeatable coupling.

Important assumptions
---------------------
- Your FiberCoupling class accepts min_boundary and max_boundary.
- FiberCoupling.run_full_optimization(...) returns:
      best_x, validated_voltage, validated_std
- FiberCoupling.history stores dictionaries containing stage, voltage, etc.
- Your system uses 5D actuator vectors: [m0, m1, m2, m3, z]
- Broad/global scan and BO may internally use only angular 4D, but final returned
  best_x should be 5D once z refinement is active.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configuration import SERVOS_TEST_POS
from controller.fiber_coupling import FiberCoupling


# =============================================================================
# User-editable configuration
# =============================================================================

MANUAL_BEST_VOLTAGE = 570.0  # mV, update if your manual best changes
TOP_K_BROAD = 3

# Keep these conservative first. Increase only after confirming safe motion.
BROAD_CONFIG = {
    "angular_range": 2000,
    "z_range": 2000,
    "global_samples": 2000,
    "local_z_step": 30,
    "bo_iterations": 30,
    "local_step": 60,
    "local_rounds": 5,
    "validation_measurements": 10,
}

MEDIUM_CONFIG = {
    "angular_range": 500,
    "z_range": 2000,
    "global_samples": 500,
    "local_z_step": 25,
    "bo_iterations": 50,
    "local_step": 30,
    "local_rounds": 5,
    "validation_measurements": 10,
}

FINE_CONFIG = {
    "angular_range": 200,
    "z_range": 2000,
    "local_z_step": 20,
    "global_samples": 200,
    "bo_iterations": 100,
    "local_step": 15,
    "local_rounds": 6,
    "validation_measurements": 20,
}

VERY_FINE_CONFIG = {
    "angular_range": 100,
    "z_range": 2000,
    "local_z_step": 2,
    "global_samples": 100,
    "bo_iterations": 100,
    "local_step": 15,
    "local_rounds": 6,
    "validation_measurements": 20,
}
# If you want to skip stages during testing, set these.
RUN_BROAD = True
RUN_MEDIUM = True
RUN_FINE = True
RUN_VERY_FINE = True

# If RUN_BROAD=False, load broad candidates from a previous run here.
# Example:
# PREVIOUS_TOP_K_FILE = r"C:\Users\eqela\Desktop\fiber_coupling\Data\2026-06-06\hierarchical_fc_18-29-54\broad_top5_positions.csv"
PREVIOUS_TOP_K_FILE: Optional[str] = None

# GP diagnostic plots saved after every stage.
# These are slices through the fitted GP at the best position found in that stage.
# If the GP was trained in 5D, z is included as dimension 4. If the GP was trained
# in 4D, the plotting code automatically falls back to motors 0--3 only.
SAVE_GP_DIAGNOSTIC_PLOTS = True
GP_GRID_POINTS_1D = 250
GP_GRID_POINTS_2D = 80
GP_KAPPA_FOR_ACQUISITION = 1.0
GP_DIMS_TO_PLOT = [0, 1, 2, 3, 4]
GP_PAIRS_TO_PLOT = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (0, 4),
    (2, 4),
]

# =============================================================================
# Helpers
# =============================================================================

@dataclass
class StageResult:
    stage: str
    region_id: int
    center_x0: float
    center_x1: float
    center_x2: float
    center_x3: float
    center_x4_z: float
    final_x0: float
    final_x1: float
    final_x2: float
    final_x3: float
    final_x4_z: float
    final_voltage: float
    final_std: float
    percent_manual_best: float
    success_80pct_manual: bool
    duration_s: float
    run_dir: str
    history_csv: str


def now_run_dir() -> Path:
    date_folder = datetime.now().strftime("%Y-%m-%d")
    time_folder = datetime.now().strftime("%H-%M-%S")
    out = Path("Data") / date_folder / f"hierarchical_fc_{time_folder}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def bounds_from_center(center: np.ndarray, angular_range: float, z_range: float) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float).reshape(-1)
    if len(center) != 5:
        raise ValueError(f"Expected 5D center [m0,m1,m2,m3,z], got shape {center.shape}: {center}")

    delta = np.array([angular_range, angular_range, angular_range, angular_range, z_range], dtype=float)
    min_b = center - delta
    max_b = center + delta

    # Servo safety clipping to nominal actuator range. Adjust if your hardware differs.
    min_b = np.clip(min_b, 0, 4095)
    max_b = np.clip(max_b, 0, 4095)
    return min_b, max_b


def history_to_dataframe(history: List[Dict]) -> pd.DataFrame:
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


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def extract_top_k_candidates(
    history_df: pd.DataFrame,
    dataset_csv: Optional[Path] = None,
    k: int = 5,
    min_distance: float = 300,
    noise_threshold: float = 12,
) -> pd.DataFrame:
    """
    Extract top-k broad-stage candidate regions from BOTH:
      1. raw Latin Hypercube / global sampling CSV
      2. BO measured points
      3. z-scan measured points
      4. global_scan best summary row, if present

    This is used to pass broad candidates into the medium stage.
    It intentionally does NOT use medium results for fine-stage top-k selection.
    Fine/very-fine should be centered on the best validated medium result.

    Returns 5D centers [x0,x1,x2,x3,x4_z].
    """
    candidates = []

    # ------------------------------------------------------------
    # 1. Raw global scan / Latin Hypercube dataset
    # ------------------------------------------------------------
    if dataset_csv is not None and Path(dataset_csv).exists():
        ds = pd.read_csv(dataset_csv)

        # Expected columns: m0,m1,m2,m3,voltage_mV,std_mV
        if all(c in ds.columns for c in ["m0", "m1", "m2", "m3", "voltage_mV"]):
            for _, row in ds.iterrows():
                if pd.isna(row["voltage_mV"]):
                    continue

                candidates.append({
                    "source_stage": "global_sampling",
                    "voltage": float(row["voltage_mV"]),
                    "x0": float(row["m0"]),
                    "x1": float(row["m1"]),
                    "x2": float(row["m2"]),
                    "x3": float(row["m3"]),
                    # Broad/global sampling keeps z fixed, so use current config z.
                    "x4_z": float(SERVOS_TEST_POS[4]),
                })
        else:
            print(
                f"Warning: dataset CSV {dataset_csv} does not have expected "
                "m0,m1,m2,m3,voltage_mV columns. Skipping raw sampling candidates."
            )

    # ------------------------------------------------------------
    # 2. History-derived candidates: BO, z scan, global_scan best
    # ------------------------------------------------------------
    if "stage" in history_df.columns:
        # BO measured points: usually x_full_* contains the 5D hardware position.
        bo = history_df[history_df["stage"] == "bayesian_optimization"].copy()
        for _, row in bo.iterrows():
            if pd.isna(row.get("voltage", np.nan)):
                continue

            x = []
            for i in range(4):
                val = row.get(f"x_real_{i}", np.nan)
                if pd.isna(val):
                    val = row.get(f"x_full_{i}", np.nan)
                x.append(val)

            z = row.get("x_full_z", np.nan)
            if pd.isna(z):
                z = row.get("x_real_z", np.nan)
            if pd.isna(z):
                z = SERVOS_TEST_POS[4]

            if any(pd.isna(v) for v in x):
                continue

            candidates.append({
                "source_stage": "bayesian_optimization",
                "voltage": float(row["voltage"]),
                "x0": float(x[0]),
                "x1": float(x[1]),
                "x2": float(x[2]),
                "x3": float(x[3]),
                "x4_z": float(z),
            })

        # z scans are real measured 5D candidates.
        zscan = history_df[history_df["stage"] == "z_scan"].copy()
        for _, row in zscan.iterrows():
            if pd.isna(row.get("voltage", np.nan)):
                continue

            x = []
            for i in range(4):
                x.append(row.get(f"x_real_{i}", np.nan))

            z = row.get("x_real_z", np.nan)
            if pd.isna(z):
                z = row.get("z", np.nan)

            if any(pd.isna(v) for v in x) or pd.isna(z):
                continue

            candidates.append({
                "source_stage": "z_scan",
                "voltage": float(row["voltage"]),
                "x0": float(x[0]),
                "x1": float(x[1]),
                "x2": float(x[2]),
                "x3": float(x[3]),
                "x4_z": float(z),
            })

        # Global scan summary row only stores the best x, usually 4D.
        gscan = history_df[history_df["stage"] == "global_scan"].copy()
        for _, row in gscan.iterrows():
            voltage = row.get("best_voltage", np.nan)
            if pd.isna(voltage):
                continue

            x = []
            for i in range(4):
                x.append(row.get(f"best_x_{i}", np.nan))

            if any(pd.isna(v) for v in x):
                continue

            candidates.append({
                "source_stage": "global_scan_best",
                "voltage": float(voltage),
                "x0": float(x[0]),
                "x1": float(x[1]),
                "x2": float(x[2]),
                "x3": float(x[3]),
                "x4_z": float(SERVOS_TEST_POS[4]),
            })

    if not candidates:
        raise RuntimeError("No candidates found from dataset or history.")

    df = pd.DataFrame(candidates)
    df = df.dropna(subset=["voltage", "x0", "x1", "x2", "x3", "x4_z"])
    df = df[df["voltage"] >= noise_threshold]
    df = df.sort_values("voltage", ascending=False).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No candidates survived noise_threshold. "
            f"Try lowering noise_threshold={noise_threshold}."
        )

    kept = []
    motor_cols = ["x0", "x1", "x2", "x3", "x4_z"]

    for _, row in df.iterrows():
        x = row[motor_cols].values.astype(float)

        too_close = False
        for prev_row in kept:
            prev = prev_row[motor_cols].values.astype(float)
            dist = np.linalg.norm(x - prev)

            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            kept.append(row)

        if len(kept) >= k:
            break

    top = pd.DataFrame(kept).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top


# Backwards-compatible alias, in case other code imports the old name.
def extract_top_k_from_history(history_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return extract_top_k_candidates(history_df=history_df, dataset_csv=None, k=k)

def run_stage(
    stage: str,
    region_id: int,
    center: np.ndarray,
    config: Dict,
    output_root: Path,
    load_global_scan: bool = False,
) -> Tuple[StageResult, pd.DataFrame, Optional[pd.DataFrame]]:
    """Run one FiberCoupling optimization stage and save history/summary."""
    stage_dir = output_root / stage / f"region_{region_id:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    min_b, max_b = bounds_from_center(
        center=center,
        angular_range=config["angular_range"],
        z_range=config["z_range"],
    )

    save_json(stage_dir / "config.json", {
        "stage": stage,
        "region_id": region_id,
        "center": np.asarray(center, dtype=float).tolist(),
        "min_boundary": min_b.tolist(),
        "max_boundary": max_b.tolist(),
        "config": config,
        "manual_best_voltage": MANUAL_BEST_VOLTAGE,
    })

    fc = FiberCoupling(
        csv_path=str(stage_dir / f"{stage}_region_{region_id:02d}_dataset.csv"),
        settle_time=1,
        oversampling=10,
        min_boundary=min_b,
        max_boundary=max_b,
    )

    t0 = time.time()
    final_x, final_voltage, final_std = fc.run_full_optimization(
        global_samples=config["global_samples"],
        bo_iterations=config["bo_iterations"],
        local_step=config["local_step"],
        local_z_step=config["local_z_step"],
        local_rounds=config["local_rounds"],
        validation_measurements=config["validation_measurements"],
        load_global_scan=load_global_scan,
    )
    duration_s = time.time() - t0

    final_x = np.asarray(final_x, dtype=float).reshape(-1)
    if len(final_x) == 4:
        final_x = np.append(final_x, SERVOS_TEST_POS[4])

    history_df = history_to_dataframe(fc.history)
    history_csv = stage_dir / "optimization_history.csv"
    history_df.to_csv(history_csv, index=False)

    # Try to copy dataset from fc.csv_path to stage folder if it exists elsewhere.
    try:
        src = Path(fc.csv_path)
        if src.exists():
            dst = stage_dir / "global_scan_dataset.csv"
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)
    except Exception as exc:
        print(f"Warning: could not copy dataset CSV: {exc}")

    result = StageResult(
        stage=stage,
        region_id=region_id,
        center_x0=float(center[0]),
        center_x1=float(center[1]),
        center_x2=float(center[2]),
        center_x3=float(center[3]),
        center_x4_z=float(center[4]),
        final_x0=float(final_x[0]),
        final_x1=float(final_x[1]),
        final_x2=float(final_x[2]),
        final_x3=float(final_x[3]),
        final_x4_z=float(final_x[4]),
        final_voltage=float(final_voltage),
        final_std=float(final_std),
        percent_manual_best=float(100.0 * final_voltage / MANUAL_BEST_VOLTAGE),
        success_80pct_manual=bool(final_voltage >= 0.8 * MANUAL_BEST_VOLTAGE),
        duration_s=float(duration_s),
        run_dir=str(stage_dir),
        history_csv=str(history_csv),
    )

    pd.DataFrame([asdict(result)]).to_csv(stage_dir / "stage_summary.csv", index=False)
    make_basic_plots(history_df, stage_dir, title_prefix=f"{stage} region {region_id}")
    save_gp_diagnostic_plots(fc, stage_dir, title_prefix=f"{stage} region {region_id}")

    top_df = None
    if stage == "broad":
        # IMPORTANT: medium candidates should come from the broad stage only,
        # and from BOTH the raw sampling CSV and the optimizer history.
        dataset_csv = stage_dir / "global_scan_dataset.csv"

        top_df = extract_top_k_candidates(
            history_df=history_df,
            dataset_csv=dataset_csv,
            k=TOP_K_BROAD,
            min_distance=150,
            noise_threshold=12,
        )

        # Save both names for compatibility with previous runs/scripts.
        top_df.to_csv(stage_dir / "top_k_positions.csv", index=False)
        top_df.to_csv(stage_dir / "top5_positions.csv", index=False)
        make_top_positions_plot(top_df, stage_dir)

    return result, history_df, top_df


# =============================================================================
# Plotting
# =============================================================================


# =============================================================================
# GP diagnostic plotting
# =============================================================================


def _gp_dimension_name(dim: int) -> str:
    return "z" if dim == 4 else f"m{dim}"


def _get_gp_plot_context(fc: FiberCoupling):
    """
    Return fitted GP context for diagnostic plotting.

    This is intentionally defensive because some versions of FiberCoupling train
    the GP only on the four angular motors, while others train on the full 5D
    vector [m0, m1, m2, m3, z].
    """
    gp_model = getattr(fc, "gp_model", None)
    if gp_model is None or getattr(gp_model, "gp", None) is None:
        print("Skipping GP diagnostic plots: no gp_model.gp found.")
        return None

    gp = gp_model.gp
    n_features = getattr(gp, "n_features_in_", None)
    if n_features is None:
        print("Skipping GP diagnostic plots: GP does not appear to be fitted yet.")
        return None

    center = getattr(fc, "best_x_real", None)
    if center is None:
        center = SERVOS_TEST_POS
    center = np.asarray(center, dtype=float).reshape(-1)

    if len(center) < n_features:
        # Fall back to SERVOS_TEST_POS if best_x_real is only 4D but GP is 5D.
        fallback = np.asarray(SERVOS_TEST_POS, dtype=float).reshape(-1)
        if len(fallback) >= n_features:
            center = fallback
        else:
            print(
                "Skipping GP diagnostic plots: center has fewer dimensions than GP "
                f"({len(center)} vs {n_features})."
            )
            return None

    center = center[:n_features]

    min_b = getattr(fc, "min_boundary", None)
    max_b = getattr(fc, "max_boundary", None)

    if min_b is None or max_b is None:
        min_b = np.zeros(n_features, dtype=float)
        max_b = np.ones(n_features, dtype=float) * 4095.0
    else:
        min_b = np.asarray(min_b, dtype=float).reshape(-1)[:n_features]
        max_b = np.asarray(max_b, dtype=float).reshape(-1)[:n_features]

    if len(min_b) < n_features or len(max_b) < n_features:
        print("Skipping GP diagnostic plots: boundary dimensions do not match GP dimensions.")
        return None

    return gp_model, gp, int(n_features), center, min_b, max_b


def _predict_gp_real_units(gp_model, X_real: np.ndarray):
    """
    Predict GP mean/std from real motor units.

    The GP is expected to be trained on normalized coordinates, so normalize
    before prediction.
    """
    X_real = np.asarray(X_real, dtype=float)
    X_norm = gp_model.normalize_X(X_real)
    return gp_model.gp.predict(X_norm, return_std=True)


def save_gp_1d_slices(fc: FiberCoupling, out_dir: Path, title_prefix: str = "") -> None:
    """
    Save 1D GP slices for each available dimension.

    Observations are deliberately not shown because projected observations can be
    misleading in a high-dimensional GP slice. The plotted curve is the GP
    prediction while all other dimensions are fixed at the best point found in
    the stage.
    """
    ctx = _get_gp_plot_context(fc)
    if ctx is None:
        return

    gp_model, gp, n_features, center, min_b, max_b = ctx
    plot_dir = out_dir / "gp_diagnostics" / "1d_slices"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dims = [d for d in GP_DIMS_TO_PLOT if d < n_features]

    for dim in dims:
        x_values = np.linspace(min_b[dim], max_b[dim], GP_GRID_POINTS_1D)
        X_real = np.tile(center, (len(x_values), 1))
        X_real[:, dim] = x_values

        mean, std = _predict_gp_real_units(gp_model, X_real)
        acq = mean + GP_KAPPA_FOR_ACQUISITION * std

        dim_name = _gp_dimension_name(dim)

        pd.DataFrame({
            dim_name: x_values,
            "gp_mean": mean,
            "gp_std": std,
            "gp_acquisition": acq,
        }).to_csv(plot_dir / f"gp_1d_{dim_name}.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, mean, linewidth=2, label="GP mean")
        plt.fill_between(x_values, mean - std, mean + std, alpha=0.25, label="± std")
        plt.plot(
            x_values,
            acq,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std",
        )
        plt.axvline(center[dim], linestyle=":", linewidth=1.5, label="Slice center")
        plt.xlabel(f"{dim_name} position [servo steps]")
        plt.ylabel("Predicted voltage [mV]")
        plt.title(f"{title_prefix}: 1D GP slice along {dim_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"gp_1d_{dim_name}.png", dpi=250)
        plt.close()


def _save_gp_2d_heatmap(x_values, y_values, Z, title, xlabel, ylabel, cbar_label, path: Path) -> None:
    plt.figure(figsize=(7, 5.5))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_gp_2d_slices(fc: FiberCoupling, out_dir: Path, title_prefix: str = "") -> None:
    """
    Save 2D GP slices through the fitted GP.

    Two dimensions vary and all remaining dimensions are fixed at the best
    position. For a 5D GP this can include z; for a 4D GP, z pairs are skipped.
    """
    ctx = _get_gp_plot_context(fc)
    if ctx is None:
        return

    gp_model, gp, n_features, center, min_b, max_b = ctx
    plot_dir = out_dir / "gp_diagnostics" / "2d_slices"
    plot_dir.mkdir(parents=True, exist_ok=True)

    valid_pairs = [
        (a, b)
        for a, b in GP_PAIRS_TO_PLOT
        if a < n_features and b < n_features and a != b
    ]

    for dim_x, dim_y in valid_pairs:
        x_values = np.linspace(min_b[dim_x], max_b[dim_x], GP_GRID_POINTS_2D)
        y_values = np.linspace(min_b[dim_y], max_b[dim_y], GP_GRID_POINTS_2D)
        XX, YY = np.meshgrid(x_values, y_values)

        X_real = np.tile(center, (GP_GRID_POINTS_2D * GP_GRID_POINTS_2D, 1))
        X_real[:, dim_x] = XX.ravel()
        X_real[:, dim_y] = YY.ravel()

        mean, std = _predict_gp_real_units(gp_model, X_real)
        mean_grid = mean.reshape(GP_GRID_POINTS_2D, GP_GRID_POINTS_2D)
        std_grid = std.reshape(GP_GRID_POINTS_2D, GP_GRID_POINTS_2D)
        acq_grid = mean_grid + GP_KAPPA_FOR_ACQUISITION * std_grid

        name_x = _gp_dimension_name(dim_x)
        name_y = _gp_dimension_name(dim_y)
        pair_name = f"{name_x}_vs_{name_y}"

        np.save(plot_dir / f"gp_mean_{pair_name}.npy", mean_grid)
        np.save(plot_dir / f"gp_std_{pair_name}.npy", std_grid)
        np.save(plot_dir / f"gp_acquisition_{pair_name}.npy", acq_grid)

        _save_gp_2d_heatmap(
            x_values,
            y_values,
            mean_grid,
            title=f"{title_prefix}: GP mean {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label="Predicted voltage [mV]",
            path=plot_dir / f"gp_mean_{pair_name}.png",
        )

        _save_gp_2d_heatmap(
            x_values,
            y_values,
            std_grid,
            title=f"{title_prefix}: GP uncertainty {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label="Predicted std [mV]",
            path=plot_dir / f"gp_std_{pair_name}.png",
        )

        _save_gp_2d_heatmap(
            x_values,
            y_values,
            acq_grid,
            title=f"{title_prefix}: GP acquisition {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std [mV]",
            path=plot_dir / f"gp_acquisition_{pair_name}.png",
        )


def save_gp_diagnostic_plots(fc: FiberCoupling, out_dir: Path, title_prefix: str = "") -> None:
    """Save GP diagnostic plots and kernel text if a fitted GP is available."""
    if not SAVE_GP_DIAGNOSTIC_PLOTS:
        return

    ctx = _get_gp_plot_context(fc)
    if ctx is None:
        return

    _, gp, n_features, center, min_b, max_b = ctx
    plot_dir = out_dir / "gp_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    names = [_gp_dimension_name(i) for i in range(n_features)]
    pd.DataFrame([center], columns=names).to_csv(plot_dir / "gp_slice_center.csv", index=False)
    pd.DataFrame([min_b], columns=names).to_csv(plot_dir / "gp_min_boundary.csv", index=False)
    pd.DataFrame([max_b], columns=names).to_csv(plot_dir / "gp_max_boundary.csv", index=False)

    with open(plot_dir / "gp_kernel.txt", "w", encoding="utf-8") as f:
        f.write(str(gp.kernel_))
        try:
            f.write(f"\nLog marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.6f}\n")
        except Exception:
            pass

    try:
        save_gp_1d_slices(fc, out_dir, title_prefix=title_prefix)
        save_gp_2d_slices(fc, out_dir, title_prefix=title_prefix)
        print(f"Saved GP diagnostic plots to: {plot_dir}")
    except Exception as exc:
        print(f"Warning: could not save GP diagnostic plots: {exc}")


def make_basic_plots(history_df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # BO trace
    if "stage" in history_df.columns:
        bo = history_df[history_df["stage"] == "bayesian_optimization"].copy()
        if not bo.empty and "iteration" in bo.columns and "voltage" in bo.columns:
            bo = bo.dropna(subset=["iteration", "voltage"])
            if not bo.empty:
                bo = bo.sort_values("iteration")
                bo["running_best"] = bo["voltage"].cummax()
                plt.figure(figsize=(8, 4.8))
                plt.plot(bo["iteration"], bo["voltage"], marker="o", linewidth=1, label="Measured voltage")
                plt.plot(bo["iteration"], bo["running_best"], linewidth=2, label="Running best")
                plt.xlabel("BO iteration")
                plt.ylabel("Voltage [mV]")
                plt.title(f"{title_prefix}: BO trace")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "bo_trace.png", dpi=200)
                plt.close()

        # z scan plot, separated by scan number so lines don't connect wrong scans
        zscan = history_df[history_df["stage"] == "z_scan"].copy()
        if not zscan.empty and "z" in zscan.columns and "voltage" in zscan.columns:
            zscan = zscan.dropna(subset=["z", "voltage"]).copy()
            if not zscan.empty:
                # Create scan_id based on z decreasing or restarting.
                z_values = zscan["z"].values
                scan_ids = []
                scan_id = 1
                prev_z = None
                for z in z_values:
                    if prev_z is not None and z < prev_z:
                        scan_id += 1
                    scan_ids.append(scan_id)
                    prev_z = z
                zscan["scan_id"] = scan_ids

                plt.figure(figsize=(8, 4.8))
                for sid, group in zscan.groupby("scan_id"):
                    group = group.sort_values("z")
                    if "voltage_std" in group.columns:
                        yerr = group["voltage_std"].values
                    else:
                        yerr = None
                    plt.errorbar(
                        group["z"],
                        group["voltage"],
                        yerr=yerr,
                        marker="o",
                        linewidth=1.5,
                        capsize=2,
                        label=f"z scan {sid}",
                    )
                plt.xlabel("z position [servo steps]")
                plt.ylabel("Voltage [mV]")
                plt.title(f"{title_prefix}: z refinement scan")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "z_scan_voltage.png", dpi=200)
                plt.close()

        # Final validation scatter
        val = history_df[history_df["stage"] == "validation"].copy()
        if not val.empty and "mean_voltage" in val.columns:
            val = val.dropna(subset=["mean_voltage"])
            if not val.empty:
                plt.figure(figsize=(6, 4))
                x = np.arange(1, len(val) + 1)
                plt.errorbar(
                    x,
                    val["mean_voltage"],
                    yerr=val.get("std_voltage", pd.Series(np.zeros(len(val)))).values,
                    marker="o",
                    capsize=3,
                )
                plt.xlabel("Validation block")
                plt.ylabel("Voltage [mV]")
                plt.title(f"{title_prefix}: validation")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "validation.png", dpi=200)
                plt.close()


def make_top_positions_plot(top_df: pd.DataFrame, out_dir: Path) -> None:
    if top_df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(top_df["rank"].astype(str), top_df["voltage"])
    plt.xlabel("Candidate rank")
    plt.ylabel("Voltage [mV]")
    plt.title("Top broad-search candidate regions")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "top5_positions.png", dpi=200)
    plt.close()


def make_summary_plot(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    labels = summary_df["stage"] + " r" + summary_df["region_id"].astype(str)
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, summary_df["final_voltage"])
    plt.axhline(MANUAL_BEST_VOLTAGE, linestyle="--", linewidth=1.5, label="Manual best")
    plt.axhline(0.8 * MANUAL_BEST_VOLTAGE, linestyle=":", linewidth=1.5, label="80% manual best")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final validated voltage [mV]")
    plt.title("Hierarchical fiber-coupling optimization summary")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hierarchical_summary.png", dpi=200)
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================


def main():
    output_root = now_run_dir()
    print("Output folder:", output_root)

    start_center = np.asarray(SERVOS_TEST_POS, dtype=float).reshape(-1)
    if len(start_center) != 5:
        raise ValueError(
            "SERVOS_TEST_POS must be 5D: [m0, m1, m2, m3, z]. "
            f"Got {start_center}"
        )

    save_json(output_root / "master_config.json", {
        "SERVOS_TEST_POS": start_center.tolist(),
        "MANUAL_BEST_VOLTAGE": MANUAL_BEST_VOLTAGE,
        "TOP_K_BROAD": TOP_K_BROAD,
        "BROAD_CONFIG": BROAD_CONFIG,
        "MEDIUM_CONFIG": MEDIUM_CONFIG,
        "FINE_CONFIG": FINE_CONFIG,
        "RUN_BROAD": RUN_BROAD,
        "RUN_MEDIUM": RUN_MEDIUM,
        "PREVIOUS_TOP_K_FILE": PREVIOUS_TOP_K_FILE,
        "RUN_FINE": RUN_FINE,
        "RUN_VERY_FINE": RUN_VERY_FINE,
        "PREVIOUS_TOP_K_FILE": PREVIOUS_TOP_K_FILE,
        "SAVE_GP_DIAGNOSTIC_PLOTS": SAVE_GP_DIAGNOSTIC_PLOTS,
        "GP_DIMS_TO_PLOT": GP_DIMS_TO_PLOT,
        "GP_PAIRS_TO_PLOT": GP_PAIRS_TO_PLOT,
        "GP_GRID_POINTS_1D": GP_GRID_POINTS_1D,
        "GP_GRID_POINTS_2D": GP_GRID_POINTS_2D,
        "GP_KAPPA_FOR_ACQUISITION": GP_KAPPA_FOR_ACQUISITION,
    })

    all_results: List[StageResult] = []

    # -------------------------------------------------------------------------
    # Stage 1: Broad search
    # -------------------------------------------------------------------------
    if RUN_BROAD:
        print("\n" + "=" * 80)
        print("STAGE 1: BROAD SEARCH")
        print("=" * 80)
        broad_result, broad_history, top5_df = run_stage(
            stage="broad",
            region_id=1,
            center=start_center,
            config=BROAD_CONFIG,
            output_root=output_root,
            load_global_scan=False,
        )
        all_results.append(broad_result)
    else:
        if PREVIOUS_TOP_K_FILE is None:
            raise RuntimeError(
                "RUN_BROAD=False, but PREVIOUS_TOP_K_FILE is not set. "
                "Set it to a previous broad_top5_positions.csv or top_k_positions.csv file."
            )
        top5_df = pd.read_csv(PREVIOUS_TOP_K_FILE)
        print(f"Loaded previous broad top-k candidates from: {PREVIOUS_TOP_K_FILE}")

    if top5_df is None or top5_df.empty:
        raise RuntimeError("No top-5 positions were generated from broad search.")

    # Also save copies at root for easy access.
    top5_df.to_csv(output_root / "broad_top_k_positions.csv", index=False)
    top5_df.to_csv(output_root / "broad_top5_positions.csv", index=False)
    print(f"\nTop {len(top5_df)} broad-search candidate positions:")
    print(top5_df)

    # -------------------------------------------------------------------------
    # Stage 2: Medium search around each top-5 candidate
    # -------------------------------------------------------------------------
    medium_results: List[StageResult] = []
    if RUN_MEDIUM:
        print("\n" + "=" * 80)
        print("STAGE 2: MEDIUM SEARCH AROUND BROAD TOP-K REGIONS")
        print("=" * 80)

        for _, row in top5_df.iterrows():
            region_id = int(row["rank"])
            center = row[["x0", "x1", "x2", "x3", "x4_z"]].values.astype(float)
            print("\n" + "-" * 80)
            print(f"Medium search around broad candidate {region_id}: {center}")
            print("-" * 80)

            result, _, _ = run_stage(
                stage="medium",
                region_id=region_id,
                center=center,
                config=MEDIUM_CONFIG,
                output_root=output_root,
                load_global_scan=False,
            )
            medium_results.append(result)
            all_results.append(result)

        medium_df = pd.DataFrame([asdict(r) for r in medium_results])
        medium_df = medium_df.sort_values("final_voltage", ascending=False).reset_index(drop=True)
        medium_df.to_csv(output_root / "medium_results_ranked.csv", index=False)
        print("\nMedium results ranked:")
        print(medium_df[["region_id", "final_voltage", "final_std", "percent_manual_best", "success_80pct_manual"]])
    else:
        medium_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Stage 3: Fine search around best medium, or broad if medium skipped
    # -------------------------------------------------------------------------
    fine_result = None
    fine_df = pd.DataFrame()
    if RUN_FINE:
        print("\n" + "=" * 80)
        print("STAGE 3: FINE SEARCH")
        print("=" * 80)

        if not medium_df.empty:
            best_row = medium_df.iloc[0]
            fine_center = best_row[["final_x0", "final_x1", "final_x2", "final_x3", "final_x4_z"]].values.astype(float)
            source_region = int(best_row["region_id"])
        else:
            fine_center = np.array([
                broad_result.final_x0,
                broad_result.final_x1,
                broad_result.final_x2,
                broad_result.final_x3,
                broad_result.final_x4_z,
            ], dtype=float)
            source_region = 1

        print(f"Fine search centered on best previous result from region {source_region}: {fine_center}")

        fine_result, _, _ = run_stage(
            stage="fine",
            region_id=source_region,
            center=fine_center,
            config=FINE_CONFIG,
            output_root=output_root,
            load_global_scan=False,
        )
        all_results.append(fine_result)
        fine_df = pd.DataFrame([asdict(fine_result)])
        fine_df.to_csv(output_root / "fine_result.csv", index=False)
        
    if RUN_VERY_FINE:
        print("\n" + "=" * 80)
        print("STAGE 4: VERY FINE SEARCH")
        print("=" * 80)

        if fine_result is not None:
            very_fine_center = np.array([
                fine_result.final_x0,
                fine_result.final_x1,
                fine_result.final_x2,
                fine_result.final_x3,
                fine_result.final_x4_z,
            ], dtype=float)
            source_region = int(fine_result.region_id)

        elif not medium_df.empty:
            best_row = medium_df.iloc[0]
            very_fine_center = best_row[
                ["final_x0", "final_x1", "final_x2", "final_x3", "final_x4_z"]
            ].values.astype(float)
            source_region = int(best_row["region_id"])

        else:
            very_fine_center = np.array([
                broad_result.final_x0,
                broad_result.final_x1,
                broad_result.final_x2,
                broad_result.final_x3,
                broad_result.final_x4_z,
            ], dtype=float)
            source_region = 1

        print(
            f"Very fine search centered on best previous result "
            f"from region {source_region}: {very_fine_center}"
        )

        very_fine_result, _, _ = run_stage(
            stage="very_fine",
            region_id=source_region,
            center=very_fine_center,
            config=VERY_FINE_CONFIG,
            output_root=output_root,
            load_global_scan=False,
        )

        all_results.append(very_fine_result)

        very_fine_df = pd.DataFrame([asdict(very_fine_result)])
        very_fine_df.to_csv(output_root / "very_fine_result.csv", index=False)
        


if __name__ == "__main__":
    main()