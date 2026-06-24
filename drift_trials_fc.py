

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from configuration import SERVOS_INTIAL_POS
from controller.fiber_coupling import FiberCoupling
from controller.servos import Servos
from controller.picoscope import Picoscope


# If SERVOS_TEST_POS already stores your manual best, leave this as SERVOS_TEST_POS.
MANUAL_BEST_POS = SERVOS_INTIAL_POS

# IMPORTANT: Update this to the manually observed best voltage in the same PD gain /
# Picoscope range settings used during the experiment.
MANUAL_BEST_VOLTAGE = 1555  # mV

N_TRIALS = 1
RANDOM_SEED = 10

# Random initial perturbation applied before each trial.
# This tests recovery from nearby drift. 
PERTURB_ANGULAR = 100  #steps for m0-m3
PERTURB_Z = 100        #steps for z

# Search box around MANUAL_BEST_POS used by the optimizer.
# Should be smlall if the experiment is specifically local recovery.
SEARCH_ANGULAR_RANGE = 150
SEARCH_Z_RANGE = 200

# Optimization budget per trial.
# For local recovery, BO can often be modest -> local/z refinement usually matters most.
OPT_CONFIG = {
    "global_samples": 50,
    "bo_iterations": 30,
    "local_step": 15,
    "local_z_step": 10,
    "local_rounds": 5,
    "validation_measurements": 10,
}

SETTLE_AFTER_INITIAL_MOVE_S = 1.0
INITIAL_MEASUREMENTS = 10
PICOSCOPE_RANGE = "PS2000_5V" 

# Success thresholds RELATIVE to manual best
SUCCESS_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

# GP diagnostic plots from the optimizer after every trial.
SAVE_GP_DIAGNOSTIC_PLOTS = True
GP_GRID_POINTS_1D = 250
GP_GRID_POINTS_2D = 80
GP_KAPPA_FOR_ACQUISITION = 1.0
GP_DIMS_TO_PLOT = [0, 1, 2, 3, 4]
GP_PAIRS_TO_PLOT = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 4), (2, 4)]


RUN_GP_SAMPLE_SWEEP = True
GP_SAMPLE_SWEEP_TRIALS = [1]        
GP_SAMPLE_COUNTS = [20, 40, 80, 100, 200]
GP_SAMPLE_SWEEP_RANDOM_SEED = 123

# Safety bounds for servos
SERVO_MIN = 0
SERVO_MAX = 4095

@dataclass
class TrialResult:
    trial: int
    initial_x0: float
    initial_x1: float
    initial_x2: float
    initial_x3: float
    initial_x4_z: float
    final_x0: float
    final_x1: float
    final_x2: float
    final_x3: float
    final_x4_z: float
    initial_voltage: float
    initial_std: float
    final_voltage: float
    final_std: float
    manual_best_voltage: float
    percent_recovered: float
    improvement_factor: float
    percent_improvement: float
    success_80pct_manual: bool
    success_90pct_manual: bool
    success_95pct_manual: bool
    initial_distance_l2: float
    initial_distance_angular_l2: float
    initial_distance_z: float
    duration_s: float
    estimated_voltage_measurements: int
    run_dir: str
    history_csv: str


# Helpers & such (not super important)

def now_output_dir() -> Path:
    date_folder = datetime.now().strftime("%Y-%m-%d")
    time_folder = datetime.now().strftime("%H-%M-%S")
    out = Path("Data") / date_folder / f"fc_recovery_benchmark_{time_folder}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clip_position(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), SERVO_MIN, SERVO_MAX)


def make_bounds(center: np.ndarray, angular_range: float, z_range: float) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float).reshape(-1)
    if len(center) != 5:
        raise ValueError(f"Expected 5D center [m0,m1,m2,m3,z], got {center}")
    delta = np.array([angular_range, angular_range, angular_range, angular_range, z_range], dtype=float)
    return clip_position(center - delta), clip_position(center + delta)


def random_perturbation(rng: np.random.Generator) -> np.ndarray:
    delta = np.array([
        rng.uniform(-PERTURB_ANGULAR, PERTURB_ANGULAR),
        rng.uniform(-PERTURB_ANGULAR, PERTURB_ANGULAR),
        rng.uniform(-PERTURB_ANGULAR, PERTURB_ANGULAR),
        rng.uniform(-PERTURB_ANGULAR, PERTURB_ANGULAR),
        rng.uniform(-PERTURB_Z, PERTURB_Z),
    ])
    return clip_position(MANUAL_BEST_POS + delta)


def move_servos(position: np.ndarray) -> None:
    position = np.round(clip_position(position)).astype(int).tolist()
    with Servos() as servos:
        servos.write(position)


def measure_voltage(n: int = 10) -> Tuple[float, float]:
    pico = Picoscope(voltage_range=PICOSCOPE_RANGE)
    values = []
    try:
        for _ in range(n):
            voltage, std = pico.get_voltage()
            values.append(float(voltage))
    finally:
        pico.close_device()

    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


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


def count_voltage_measurements(history_df: pd.DataFrame) -> int:
    if history_df.empty:
        return 0
    count = 0
    for col in ["voltage", "mean_voltage", "best_voltage"]:
        if col in history_df.columns:
            count += int(history_df[col].notna().sum())
    return count


def safe_improvement_factor(final_v: float, initial_v: float) -> float:
    if initial_v <= 1e-9:
        return float("nan")
    return float(final_v / initial_v)


def safe_percent_improvement(final_v: float, initial_v: float) -> float:
    if abs(initial_v) <= 1e-9:
        return float("nan")
    return float(100.0 * (final_v - initial_v) / initial_v)



# -------------- GP plotting --------------

def _gp_dimension_name(dim: int) -> str:
    return "z" if dim == 4 else f"m{dim}"


def _get_gp_plot_context(fc: FiberCoupling):
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
        center = MANUAL_BEST_POS
    center = np.asarray(center, dtype=float).reshape(-1)

    if len(center) < n_features:
        fallback = np.asarray(MANUAL_BEST_POS, dtype=float).reshape(-1)
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
    X_real = np.asarray(X_real, dtype=float)
    X_norm = gp_model.normalize_X(X_real)
    return gp_model.gp.predict(X_norm, return_std=True)


def save_gp_1d_slices(fc: FiberCoupling, out_dir: Path, title_prefix: str = "") -> None:
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
        plt.plot(x_values, acq, linestyle="--", linewidth=1.5, label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std")
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
    ctx = _get_gp_plot_context(fc)
    if ctx is None:
        return

    gp_model, gp, n_features, center, min_b, max_b = ctx
    plot_dir = out_dir / "gp_diagnostics" / "2d_slices"
    plot_dir.mkdir(parents=True, exist_ok=True)

    valid_pairs = [(a, b) for a, b in GP_PAIRS_TO_PLOT if a < n_features and b < n_features and a != b]
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
            x_values, y_values, mean_grid,
            title=f"{title_prefix}: GP mean {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label="Predicted voltage [mV]",
            path=plot_dir / f"gp_mean_{pair_name}.png",
        )
        _save_gp_2d_heatmap(
            x_values, y_values, std_grid,
            title=f"{title_prefix}: GP uncertainty {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label="Predicted std [mV]",
            path=plot_dir / f"gp_std_{pair_name}.png",
        )
        _save_gp_2d_heatmap(
            x_values, y_values, acq_grid,
            title=f"{title_prefix}: GP acquisition {name_x} vs {name_y}",
            xlabel=f"{name_x} position [servo steps]",
            ylabel=f"{name_y} position [servo steps]",
            cbar_label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std [mV]",
            path=plot_dir / f"gp_acquisition_{pair_name}.png",
        )


def save_gp_diagnostic_plots(fc: FiberCoupling, out_dir: Path, title_prefix: str = "") -> None:
    """Save GP diagnostic plots and kernel text if the optimizer has a fitted GP."""
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


# GP sample-count sweep plotting

def _normalize_with_bounds(X: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> np.ndarray:
    denom = np.maximum(max_b - min_b, 1e-9)
    return (np.asarray(X, dtype=float) - min_b) / denom


def _fit_standalone_gp(X_real: np.ndarray, y: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> GaussianProcessRegressor:
    X_norm = _normalize_with_bounds(X_real, min_b, max_b)
    kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) *
            #RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e3)) +
            Matern(length_scale=1.0, nu=1.5) +
            WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(X_norm, np.asarray(y, dtype=float).reshape(-1))
    return gp


def _predict_standalone_gp(gp: GaussianProcessRegressor, X_real: np.ndarray, min_b: np.ndarray, max_b: np.ndarray):
    X_norm = _normalize_with_bounds(X_real, min_b, max_b)
    return gp.predict(X_norm, return_std=True)


def _save_measured_projection(dataset: pd.DataFrame, out_dir: Path, dims: Tuple[int, int] = (0, 1)) -> None:
    xcol, ycol = f"m{dims[0]}", f"m{dims[1]}"
    if not all(c in dataset.columns for c in [xcol, ycol, "voltage_mV"]):
        return
    plt.figure(figsize=(7, 5.5))
    sc = plt.scatter(dataset[xcol], dataset[ycol], c=dataset["voltage_mV"], s=28, alpha=0.85)
    plt.xlabel(f"{xcol} position [servo steps]")
    plt.ylabel(f"{ycol} position [servo steps]")
    plt.title(f"Measured samples: {xcol} vs {ycol}")
    plt.colorbar(sc, label="Measured voltage [mV]")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"measured_projection_{xcol}_vs_{ycol}.png", dpi=250)
    plt.close()


def save_gp_sample_count_sweep(
    dataset_csv: Path,
    out_dir: Path,
    min_b: np.ndarray,
    max_b: np.ndarray,
    center: np.ndarray,
) -> None:
    """Compare GP maps trained with different numbers of global-scan samples."""
    if not RUN_GP_SAMPLE_SWEEP or not Path(dataset_csv).exists():
        return

    ds = pd.read_csv(dataset_csv)
    required = ["m0", "m1", "m2", "m3", "voltage_mV"]
    if not all(c in ds.columns for c in required):
        print(f"Skipping GP sample sweep: {dataset_csv} lacks {required}")
        return

    ds = ds.dropna(subset=required).reset_index(drop=True)
    if ds.empty:
        return

    sweep_dir = out_dir / "gp_sample_count_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    _save_measured_projection(ds, sweep_dir, dims=(0, 1))
    _save_measured_projection(ds, sweep_dir, dims=(0, 2))

    X_all = ds[["m0", "m1", "m2", "m3"]].values.astype(float)
    y_all = ds["voltage_mV"].values.astype(float)
    min4 = np.asarray(min_b, dtype=float).reshape(-1)[:4]
    max4 = np.asarray(max_b, dtype=float).reshape(-1)[:4]
    center4 = np.asarray(center, dtype=float).reshape(-1)[:4]

    rng = np.random.default_rng(GP_SAMPLE_SWEEP_RANDOM_SEED)
    summary_rows = []
    available_counts = [n for n in GP_SAMPLE_COUNTS if n <= len(ds)]
    if len(ds) not in available_counts:
        available_counts.append(len(ds))

    for n in sorted(set(available_counts)):
        n_dir = sweep_dir / f"n_{n:04d}"
        n_dir.mkdir(parents=True, exist_ok=True)

        if n == len(ds):
            idx = np.arange(len(ds))
        else:
            # Include the best measured point and sample the rest randomly.
            best_idx = int(np.argmax(y_all))
            pool = np.array([i for i in range(len(ds)) if i != best_idx])
            extra = rng.choice(pool, size=n - 1, replace=False) if n > 1 else np.array([], dtype=int)
            idx = np.concatenate([[best_idx], extra])

        X = X_all[idx]
        y = y_all[idx]
        gp = _fit_standalone_gp(X, y, min4, max4)

        with open(n_dir / "gp_kernel.txt", "w", encoding="utf-8") as f:
            f.write(str(gp.kernel_))
            try:
                f.write(f"\nLog marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.6f}\n")
            except Exception:
                pass

        summary_rows.append({
            "n_samples": int(n),
            "best_measured_voltage": float(np.max(y)),
            "mean_measured_voltage": float(np.mean(y)),
            "kernel": str(gp.kernel_),
        })

        # 1D slices
        one_dir = n_dir / "1d_slices"
        one_dir.mkdir(exist_ok=True)
        for dim in [0, 1, 2, 3]:
            x_values = np.linspace(min4[dim], max4[dim], GP_GRID_POINTS_1D)
            X_grid = np.tile(center4, (len(x_values), 1))
            X_grid[:, dim] = x_values
            mean, std = _predict_standalone_gp(gp, X_grid, min4, max4)
            acq = mean + GP_KAPPA_FOR_ACQUISITION * std
            dim_name = f"m{dim}"
            pd.DataFrame({dim_name: x_values, "gp_mean": mean, "gp_std": std, "gp_acquisition": acq}).to_csv(
                one_dir / f"gp_1d_{dim_name}.csv", index=False
            )
            plt.figure(figsize=(8, 5))
            plt.plot(x_values, mean, linewidth=2, label="GP mean")
            plt.fill_between(x_values, mean - std, mean + std, alpha=0.25, label="± std")
            plt.plot(x_values, acq, linestyle="--", linewidth=1.5, label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std")
            plt.axvline(center4[dim], linestyle=":", linewidth=1.5, label="Slice center")
            plt.xlabel(f"{dim_name} position [servo steps]")
            plt.ylabel("Predicted voltage [mV]")
            plt.title(f"GP 1D slice with {n} samples along {dim_name}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(one_dir / f"gp_1d_{dim_name}.png", dpi=250)
            plt.close()

        # 2D heatmaps for a few pairs
        two_dir = n_dir / "2d_slices"
        two_dir.mkdir(exist_ok=True)
        for dim_x, dim_y in [(0, 1), (0, 2), (1, 2), (2, 3)]:
            x_values = np.linspace(min4[dim_x], max4[dim_x], GP_GRID_POINTS_2D)
            y_values = np.linspace(min4[dim_y], max4[dim_y], GP_GRID_POINTS_2D)
            XX, YY = np.meshgrid(x_values, y_values)
            X_grid = np.tile(center4, (GP_GRID_POINTS_2D * GP_GRID_POINTS_2D, 1))
            X_grid[:, dim_x] = XX.ravel()
            X_grid[:, dim_y] = YY.ravel()
            mean, std = _predict_standalone_gp(gp, X_grid, min4, max4)
            mean_grid = mean.reshape(GP_GRID_POINTS_2D, GP_GRID_POINTS_2D)
            std_grid = std.reshape(GP_GRID_POINTS_2D, GP_GRID_POINTS_2D)
            acq_grid = mean_grid + GP_KAPPA_FOR_ACQUISITION * std_grid
            name_x, name_y = f"m{dim_x}", f"m{dim_y}"
            pair_name = f"{name_x}_vs_{name_y}"
            _save_gp_2d_heatmap(
                x_values, y_values, mean_grid,
                title=f"GP mean {pair_name} with {n} samples",
                xlabel=f"{name_x} position [servo steps]",
                ylabel=f"{name_y} position [servo steps]",
                cbar_label="Predicted voltage [mV]",
                path=two_dir / f"gp_mean_{pair_name}.png",
            )
            _save_gp_2d_heatmap(
                x_values, y_values, std_grid,
                title=f"GP std {pair_name} with {n} samples",
                xlabel=f"{name_x} position [servo steps]",
                ylabel=f"{name_y} position [servo steps]",
                cbar_label="Predicted std [mV]",
                path=two_dir / f"gp_std_{pair_name}.png",
            )
            _save_gp_2d_heatmap(
                x_values, y_values, acq_grid,
                title=f"GP acquisition {pair_name} with {n} samples",
                xlabel=f"{name_x} position [servo steps]",
                ylabel=f"{name_y} position [servo steps]",
                cbar_label=f"Mean + {GP_KAPPA_FOR_ACQUISITION:g} std [mV]",
                path=two_dir / f"gp_acquisition_{pair_name}.png",
            )

    pd.DataFrame(summary_rows).to_csv(sweep_dir / "sample_sweep_summary.csv", index=False)


# More Plotting

def plot_trial_summary(results_df: pd.DataFrame, out_dir: Path) -> None:
    if results_df.empty:
        return

    # 1. Initial vs final voltage, y=x line
    plt.figure(figsize=(6, 5))
    plt.scatter(results_df["initial_voltage"], results_df["final_voltage"], alpha=0.8)
    min_v = min(results_df["initial_voltage"].min(), results_df["final_voltage"].min())
    max_v = max(results_df["initial_voltage"].max(), results_df["final_voltage"].max(), MANUAL_BEST_VOLTAGE)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5, label="y = x")
    plt.axhline(MANUAL_BEST_VOLTAGE, linestyle=":", linewidth=1.5, label="Manual best")
    plt.axhline(0.8 * MANUAL_BEST_VOLTAGE, linestyle="-.", linewidth=1.5, label="80% manual best")
    plt.xlabel("Initial voltage [mV]")
    plt.ylabel("Final validated voltage [mV]")
    plt.title("Initial vs final coupling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "initial_vs_final_voltage.png", dpi=250)
    plt.close()

    # 2. Recovery histogram
    plt.figure(figsize=(7, 4.5))
    plt.hist(results_df["percent_recovered"], bins=15, edgecolor="black", alpha=0.8)
    for thr in SUCCESS_THRESHOLDS:
        plt.axvline(100 * thr, linestyle="--", linewidth=1.5, label=f"{int(100*thr)}% threshold")
    plt.xlabel("Recovery [% of manual best]")
    plt.ylabel("Trial count")
    plt.title("Recovery distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "recovery_histogram.png", dpi=250)
    plt.close()

    # 3. Final voltage by trial
    plt.figure(figsize=(9, 4.5))
    plt.errorbar(
        results_df["trial"],
        results_df["final_voltage"],
        yerr=results_df["final_std"],
        marker="o",
        linestyle="-",
        capsize=3,
    )
    plt.axhline(MANUAL_BEST_VOLTAGE, linestyle=":", linewidth=1.5, label="Manual best")
    plt.axhline(0.8 * MANUAL_BEST_VOLTAGE, linestyle="--", linewidth=1.5, label="80% manual best")
    plt.xlabel("Trial")
    plt.ylabel("Final validated voltage [mV]")
    plt.title("Final coupling per recovery trial")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "final_voltage_by_trial.png", dpi=250)
    plt.close()

    # 4. Duration histogram
    plt.figure(figsize=(7, 4.5))
    plt.hist(results_df["duration_s"], bins=15, edgecolor="black", alpha=0.8)
    plt.xlabel("Duration [s]")
    plt.ylabel("Trial count")
    plt.title("Optimization duration distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "duration_histogram.png", dpi=250)
    plt.close()

    # 5. Initial perturbation distance vs recovery
    plt.figure(figsize=(7, 4.8))
    plt.scatter(results_df["initial_distance_angular_l2"], results_df["percent_recovered"], alpha=0.85)
    plt.axhline(80, linestyle="--", linewidth=1.5, label="80% threshold")
    plt.axhline(90, linestyle=":", linewidth=1.5, label="90% threshold")
    plt.xlabel("Initial angular displacement L2 [servo steps]")
    plt.ylabel("Recovery [% of manual best]")
    plt.title("Capture range: displacement vs recovery")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "displacement_vs_recovery.png", dpi=250)
    plt.close()

    # 6. Success rate by displacement bin
    df = results_df.copy()
    bins = [0, 50, 100, 150, 200, 300, 500]
    df["distance_bin"] = pd.cut(df["initial_distance_angular_l2"], bins=bins, include_lowest=True)
    grouped = df.groupby("distance_bin", observed=False).agg(
        trials=("trial", "count"),
        success80=("success_80pct_manual", "mean"),
        mean_recovery=("percent_recovered", "mean"),
    ).reset_index()
    grouped.to_csv(out_dir / "success_by_displacement_bin.csv", index=False)

    if not grouped.empty:
        plt.figure(figsize=(8, 4.5))
        labels = grouped["distance_bin"].astype(str)
        plt.bar(labels, 100 * grouped["success80"].fillna(0))
        plt.xlabel("Initial angular displacement bin [servo steps]")
        plt.ylabel("Success rate at 80% manual best [%]")
        plt.title("Success rate vs perturbation magnitude")
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 105)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "success_rate_by_displacement.png", dpi=250)
        plt.close()


def save_text_summary(results_df: pd.DataFrame, out_dir: Path):
    if results_df.empty:
        return

    summary = {
        "n_trials": int(len(results_df)),
        "manual_best_voltage_mV": float(MANUAL_BEST_VOLTAGE),
        "mean_initial_voltage_mV": float(results_df["initial_voltage"].mean()),
        "median_initial_voltage_mV": float(results_df["initial_voltage"].median()),
        "mean_final_voltage_mV": float(results_df["final_voltage"].mean()),
        "median_final_voltage_mV": float(results_df["final_voltage"].median()),
        "best_final_voltage_mV": float(results_df["final_voltage"].max()),
        "std_between_trials_mV": float(results_df["final_voltage"].std(ddof=1)),
        "mean_percent_recovered": float(results_df["percent_recovered"].mean()),
        "median_percent_recovered": float(results_df["percent_recovered"].median()),
        "success_rate_80pct_manual": float(results_df["success_80pct_manual"].mean()),
        "success_rate_90pct_manual": float(results_df["success_90pct_manual"].mean()),
        "success_rate_95pct_manual": float(results_df["success_95pct_manual"].mean()),
        "mean_duration_s": float(results_df["duration_s"].mean()),
        "median_duration_s": float(results_df["duration_s"].median()),
        "total_duration_s": float(results_df["duration_s"].sum()),
        "mean_estimated_voltage_measurements": float(results_df["estimated_voltage_measurements"].mean()),
    }

    save_json(out_dir / "summary_metrics.json", summary)

    with open(out_dir / "summary_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Fiber-coupling recovery benchmark summary\n")
        f.write("=" * 48 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


# Main experiment

def run_one_trial(trial: int, initial_pos: np.ndarray, output_root: Path):
    trial_dir = output_root / f"trial_{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    min_b, max_b = make_bounds(MANUAL_BEST_POS, SEARCH_ANGULAR_RANGE, SEARCH_Z_RANGE)

    save_json(trial_dir / "trial_config.json", {
        "trial": trial,
        "manual_best_pos": MANUAL_BEST_POS.tolist(),
        "manual_best_voltage": MANUAL_BEST_VOLTAGE,
        "initial_pos": initial_pos.tolist(),
        "min_boundary": min_b.tolist(),
        "max_boundary": max_b.tolist(),
        "opt_config": OPT_CONFIG,
        "perturb_angular": PERTURB_ANGULAR,
        "perturb_z": PERTURB_Z,
    })

    print("\n" + "=" * 80)
    print(f"TRIAL {trial}/{N_TRIALS}")
    print("=" * 80)
    print("Initial perturbed position:", np.round(initial_pos).astype(int))

    # Move to the perturbed state and measure initial coupling.
    move_servos(initial_pos)
    time.sleep(SETTLE_AFTER_INITIAL_MOVE_S)
    initial_voltage, initial_std = measure_voltage(n=INITIAL_MEASUREMENTS)
    print(f"Initial voltage: {initial_voltage:.6f} ± {initial_std:.6f} mV")

    fc = FiberCoupling(
        csv_path=str(trial_dir / "global_scan_dataset.csv"),
        settle_time=1,
        oversampling=10,
        min_boundary=min_b,
        max_boundary=max_b,
    )

    t0 = time.time()
    final_x, final_voltage, final_std = fc.run_full_optimization(
        global_samples=OPT_CONFIG["global_samples"],
        bo_iterations=OPT_CONFIG["bo_iterations"],
        local_step=OPT_CONFIG["local_step"],
        local_z_step=OPT_CONFIG["local_z_step"],
        local_rounds=OPT_CONFIG["local_rounds"],
        validation_measurements=OPT_CONFIG["validation_measurements"],
        load_global_scan=False,
    )
    duration_s = time.time() - t0

    final_x = np.asarray(final_x, dtype=float).reshape(-1)
    if len(final_x) == 4:
        final_x = np.append(final_x, MANUAL_BEST_POS[4])

    history_df = history_to_dataframe(fc.history)
    history_csv = trial_dir / "optimization_history.csv"
    history_df.to_csv(history_csv, index=False)

    # Save GP diagnostics from the actual optimizer used in this trial.
    save_gp_diagnostic_plots(fc, trial_dir, title_prefix=f"trial {trial}")

    # Optional analysis: compare GP maps trained with different numbers of raw
    # global-scan samples from this trial. This is usually expensive, so use it
    # only for selected trials via GP_SAMPLE_SWEEP_TRIALS.
    if RUN_GP_SAMPLE_SWEEP and (not GP_SAMPLE_SWEEP_TRIALS or trial in GP_SAMPLE_SWEEP_TRIALS):
        save_gp_sample_count_sweep(
            dataset_csv=trial_dir / "global_scan_dataset.csv",
            out_dir=trial_dir,
            min_b=min_b,
            max_b=max_b,
            center=MANUAL_BEST_POS,
        )

    estimated_measurements = count_voltage_measurements(history_df) + INITIAL_MEASUREMENTS

    delta = initial_pos - MANUAL_BEST_POS
    initial_distance_l2 = float(np.linalg.norm(delta))
    initial_distance_angular_l2 = float(np.linalg.norm(delta[:4]))
    initial_distance_z = float(abs(delta[4]))

    result = TrialResult(
        trial=trial,
        initial_x0=float(initial_pos[0]),
        initial_x1=float(initial_pos[1]),
        initial_x2=float(initial_pos[2]),
        initial_x3=float(initial_pos[3]),
        initial_x4_z=float(initial_pos[4]),
        final_x0=float(final_x[0]),
        final_x1=float(final_x[1]),
        final_x2=float(final_x[2]),
        final_x3=float(final_x[3]),
        final_x4_z=float(final_x[4]),
        initial_voltage=float(initial_voltage),
        initial_std=float(initial_std),
        final_voltage=float(final_voltage),
        final_std=float(final_std),
        manual_best_voltage=float(MANUAL_BEST_VOLTAGE),
        percent_recovered=float(100.0 * final_voltage / MANUAL_BEST_VOLTAGE),
        improvement_factor=safe_improvement_factor(final_voltage, initial_voltage),
        percent_improvement=safe_percent_improvement(final_voltage, initial_voltage),
        success_80pct_manual=bool(final_voltage >= 0.80 * MANUAL_BEST_VOLTAGE),
        success_90pct_manual=bool(final_voltage >= 0.90 * MANUAL_BEST_VOLTAGE),
        success_95pct_manual=bool(final_voltage >= 0.95 * MANUAL_BEST_VOLTAGE),
        initial_distance_l2=initial_distance_l2,
        initial_distance_angular_l2=initial_distance_angular_l2,
        initial_distance_z=initial_distance_z,
        duration_s=float(duration_s),
        estimated_voltage_measurements=int(estimated_measurements),
        run_dir=str(trial_dir),
        history_csv=str(history_csv),
    )

    pd.DataFrame([asdict(result)]).to_csv(trial_dir / "trial_result.csv", index=False)

    print(f"Final voltage: {final_voltage:.6f} ± {final_std:.6f} mV")
    print(f"Recovered: {result.percent_recovered:.2f}% of manual best")
    print(f"Duration: {duration_s:.1f} s")

    return result


def main() -> None:
    output_root = now_output_dir()
    print("Output folder:", output_root)

    if len(MANUAL_BEST_POS) != 5:
        raise ValueError(
            "MANUAL_BEST_POS / SERVOS_TEST_POS must be 5D: [m0,m1,m2,m3,z]. "
            f"Got {MANUAL_BEST_POS}"
        )

    save_json(output_root / "experiment_config.json", {
        "manual_best_pos": MANUAL_BEST_POS.tolist(),
        "manual_best_voltage": MANUAL_BEST_VOLTAGE,
        "n_trials": N_TRIALS,
        "random_seed": RANDOM_SEED,
        "perturb_angular": PERTURB_ANGULAR,
        "perturb_z": PERTURB_Z,
        "search_angular_range": SEARCH_ANGULAR_RANGE,
        "search_z_range": SEARCH_Z_RANGE,
        "opt_config": OPT_CONFIG,
        "initial_measurements": INITIAL_MEASUREMENTS,
        "picoscope_range": PICOSCOPE_RANGE,
        "success_thresholds": SUCCESS_THRESHOLDS,
        "save_gp_diagnostic_plots": SAVE_GP_DIAGNOSTIC_PLOTS,
        "run_gp_sample_sweep": RUN_GP_SAMPLE_SWEEP,
        "gp_sample_sweep_trials": GP_SAMPLE_SWEEP_TRIALS,
        "gp_sample_counts": GP_SAMPLE_COUNTS,
    })

    rng = np.random.default_rng(RANDOM_SEED)
    results: List[TrialResult] = []

    for trial in range(1, N_TRIALS + 1):
        initial_pos = random_perturbation(rng)
        try:
            result = run_one_trial(trial, initial_pos, output_root)
            results.append(result)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            break
        except Exception as exc:
            print(f"ERROR in trial {trial}: {exc}")
            error_dir = output_root / f"trial_{trial:03d}"
            error_dir.mkdir(parents=True, exist_ok=True)
            save_json(error_dir / "error.json", {"trial": trial, "error": str(exc)})

        # Save continuously after every trial.
        if results:
            results_df = pd.DataFrame([asdict(r) for r in results])
            results_df.to_csv(output_root / "all_trial_results.csv", index=False)
            save_text_summary(results_df, output_root)
            plot_trial_summary(results_df, output_root)

    if results:
        results_df = pd.DataFrame([asdict(r) for r in results])
        results_df.to_csv(output_root / "all_trial_results.csv", index=False)
        save_text_summary(results_df, output_root)
        plot_trial_summary(results_df, output_root)
        print("\n" + "=" * 80)
        print("RECOVERY BENCHMARK COMPLETE")
        print("=" * 80)
        print("Results CSV:", output_root / "all_trial_results.csv")
        print("Summary:", output_root / "summary_metrics.txt")
        print("Plots saved in:", output_root)
        print(results_df[[
            "trial",
            "initial_voltage",
            "final_voltage",
            "percent_recovered",
            "success_80pct_manual",
            "duration_s",
        ]].tail())
    else:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
