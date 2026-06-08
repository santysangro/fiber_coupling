from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict,  Tuple

import numpy as np

from configuration import SERVOS_TEST_POS
from controller.fiber_coupling import FiberCoupling
from controller.servos import Servos
from controller.picoscope import Picoscope


# ---- User settings ----

# If SERVOS_TEST_POS already stores your manual best, leave this as SERVOS_TEST_POS.
#Otherwise ADD BEST POSITION 
INITIAL_BEST_POS = np.asarray(SERVOS_TEST_POS, dtype=float)


# Search box around MANUAL_BEST_POS used by the optimizer.
# Keep this small if the aim is specifically local recovery.
SEARCH_ANGULAR_RANGE = 150
SEARCH_Z_RANGE = 200

# Optimization budget per trial.
# For local recovery, BO can often be modest; local/z refinement usually matters most.
OPT_CONFIG = {
    "global_samples": 50,
    "bo_iterations": 30,
    "local_step": 15,
    "local_z_step": 10,
    "local_rounds": 5,
    "validation_measurements": 10,
}

# Measurement settings
SETTLE_AFTER_INITIAL_MOVE_S = 1.0
INITIAL_MEASUREMENTS = 10
PICOSCOPE_RANGE = "PS2000_2V"  # change if needed

# Safety bounds for servos
SERVO_MIN = 0
SERVO_MAX = 4095



# =============================================================================
# Helpers
# =============================================================================


def clip_position(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), SERVO_MIN, SERVO_MAX)


def make_bounds(center: np.ndarray, angular_range: float, z_range: float) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float).reshape(-1)
    if len(center) != 5:
        raise ValueError(f"Expected 5D center [m0,m1,m2,m3,z], got {center}")
    delta = np.array([angular_range, angular_range, angular_range, angular_range, z_range], dtype=float)
    return clip_position(center - delta), clip_position(center + delta)


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

def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



# Main optimization loop

def run_one_trial(output_root: Path, initial_best_voltage, initial_pos=INITIAL_BEST_POS, load_json=False):
    if not load_json:
        trial_dir = output_root
        trial_dir.mkdir(parents=True, exist_ok=True)

        min_b, max_b = make_bounds(INITIAL_BEST_POS, SEARCH_ANGULAR_RANGE, SEARCH_Z_RANGE)

        save_json(trial_dir / "trial_config.json", {
            "manual_best_pos": INITIAL_BEST_POS.tolist(),
            "initial_best_voltage": initial_best_voltage,
            "initial_pos": initial_pos.tolist(),
            "min_boundary": min_b.tolist(),
            "max_boundary": max_b.tolist(),
            "opt_config": OPT_CONFIG,
        })

    print("Initial position:", np.round(initial_pos).astype(int))

    # Move to the perturbed state and measure initial coupling.
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
        final_x = np.append(final_x, INITIAL_BEST_POS[4])


    percent_recovered=float(100.0 * final_voltage / initial_best_voltage),
    print(f"Final voltage: {final_voltage:.6f} ± {final_std:.6f} mV")
    print(f"Recovered: {percent_recovered:.2f}% of manual best")
    print(f"Duration: {duration_s:.1f} s")


if __name__ == "__main__":
    initial_best_voltage, _ = measure_voltage(n=INITIAL_MEASUREMENTS)
    run_one_trial(output_root="settings", initial_best_voltage=initial_best_voltage)

