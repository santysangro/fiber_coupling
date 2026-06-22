from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict,  Tuple

import numpy as np

from controller.fiber_coupling import FiberCoupling
from controller.servos import Servos
from controller.picoscope import Picoscope

from configuration import PICOSCOPE_RANGE


# ---- User settings ----

# If SERVOS_TEST_POS already stores your manual best, leave this as SERVOS_TEST_POS.
#Otherwise add None
INITIAL_BEST_POS = np.asarray([1118, 3451, 2082, 683, 0]) #np.asarray([569, 3811, 2097, 770, 206], dtype=float)


# Search box around MANUAL_BEST_POS used by the optimizer.
# Keep this small if the aim is specifically local recovery.
SEARCH_ANGULAR_RANGE = 50
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
#PICOSCOPE_RANGE = "PS2000_2V"  # change if needed

# Safety bounds for servos
SERVO_MIN = 0
SERVO_MAX = 4095




def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------------- Helper functions to measure initial voltage & define your boundaries -------------

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

def clip_position(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), SERVO_MIN, SERVO_MAX)

def make_bounds(center: np.ndarray, radius: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float).reshape(-1)
    radius = np.asarray(radius, dtype=float).reshape(-1)
    if len(center) != 5 or len(radius) != 5:
        raise ValueError("center and radius must both be 5D arrays.")
    return clip_position(center - radius), clip_position(center + radius)
    
# Main optimization loop

def run_fine_tune(output_root: Path, initial_pos=INITIAL_BEST_POS, load_json=False):
    settings_dir = output_root
    settings_dir.mkdir(parents=True, exist_ok=True)

    if load_json:
        config = json.loads((settings_dir / "config.json").read_text())
        min_b = np.asarray(config["min_boundary"], dtype=float)
        max_b = np.asarray(config["max_boundary"], dtype=float)
        initial_pos = np.asarray(config["initial_pos"], dtype=float)
    else:
        if initial_pos is None:
            with Servos() as servos:
                initial_pos = np.array([x[1] for x in servos.read()])

        radius = np.array([
            SEARCH_ANGULAR_RANGE,
            SEARCH_ANGULAR_RANGE,
            SEARCH_ANGULAR_RANGE,
            SEARCH_ANGULAR_RANGE,
            SEARCH_Z_RANGE
        ])
        min_b, max_b = make_bounds(initial_pos, radius)

    print("Initial position:", np.round(initial_pos).astype(int))
    with Servos() as servos:
        servos.write(initial_pos)
        time.sleep(SETTLE_AFTER_INITIAL_MOVE_S)
    initial_voltage, initial_std = measure_voltage(n=INITIAL_MEASUREMENTS)
    print(f"Initial voltage: {initial_voltage:.6f} ± {initial_std:.6f} mV")

    save_json(settings_dir / "config.json", {
                "intial_best_pos": initial_pos.tolist(),
                "initial_best_voltage": initial_voltage,
                "initial_pos": initial_pos.tolist(),
                "min_boundary": min_b.tolist(),
                "max_boundary": max_b.tolist(),
                "opt_config": OPT_CONFIG,
            })

    fc = FiberCoupling(
        csv_path=str(settings_dir / "global_scan_dataset.csv"),
        settle_time=1,
        oversampling=10,
        min_boundary=min_b,
        max_boundary=max_b,
        center=initial_pos,
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


    percent_recovered=float(100.0 * final_voltage / initial_voltage)
    print(f"Final voltage: {final_voltage:.6f} ± {final_std:.6f} mV")
    print(f"Recovered: {percent_recovered:.2f}% of manual best")
    print(f"Duration: {duration_s:.1f} s")


if __name__ == "__main__":
    run_fine_tune(output_root=Path("settings"))

