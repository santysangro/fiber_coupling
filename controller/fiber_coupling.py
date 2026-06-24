from model.data_acquisition import DataAcquisition
from model.gaussian_process import GaussianProcessModel
from controller.servos import Servos
from controller.picoscope import Picoscope
from configuration import SERVOS_INTIAL_POS, PICOSCOPE_RANGE

import numpy as np
import time
from pathlib import Path

IMPROVEMENT_THRESHOLD = 5 #Improvement threshold for FC refinement loop.

class FiberCoupling:
    def __init__(
        self,
        csv_path="Data/fiber_dataset.csv",
        settle_time=1,
        oversampling=10,
        min_boundary=None,
        max_boundary=None,
        center=SERVOS_INTIAL_POS
    ):
        self.csv_path = str(Path(csv_path))
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.settle_time = settle_time
        self.oversampling = oversampling

        self.min_boundary = min_boundary
        self.max_boundary = max_boundary
        self.data_acq = DataAcquisition(
            data_path=self.csv_path,
            search_type="LatinHypercube",
        )
        gp_min_boundary=np.asarray(self.min_boundary)[:4]
        gp_max_boundary=np.asarray(self.max_boundary)[:4]
        self.gp_model = GaussianProcessModel(min_boundary=gp_min_boundary, max_boundary=gp_max_boundary)
        self.center = center

        self.progress = 0
        self.pico = None

        self.best_x_real = None
        self.best_value = -np.inf
        self.history = []

  
    def initialize_hardware(self):
        if self.pico is None:
            self.pico = Picoscope(voltage_range=PICOSCOPE_RANGE)

    def close_hardware(self):
        if self.pico is not None:
            self.pico.close_device()

    # ---------- Main implementation ---------------

    def run_full_optimization(
        self,
        global_samples=500,
        bo_iterations=50,
        local_step=50,
        local_rounds=6,
        local_z_step=100,
        validation_measurements=10,
        load_global_scan=False,
    ):
        """
        Function which runs the entire thing:
            1. Global Latin Hypercube scan.
            2. GP with Bayesian Optimization.
            3. Local coordinate refinement.
            4. Local z-refinement.
            5. Repeated validation of final point.

        Returns:
            best_x, validated_voltage, validated_std
        """

        self.initialize_hardware()
        self.history = []
        self.progress = 0
        print("\n===== STAGE 1: GLOBAL SCAN =====")
        X_global, y_global = self.run_global_scan(
            n_samples=global_samples,
            load_only=load_global_scan,
        )

        print("\n===== STAGE 2: GP BAYESIAN OPTIMIZATION =====")
        best_x_bo, best_y_bo = self.run_bayesian_optimization(
            X_init=X_global,
            y_init=y_global,
            n_iterations=bo_iterations,
        )
        if len(best_x_bo) == 4:
            best_x_bo = np.append(best_x_bo, self.center[4])

        previous_best = best_y_bo
        best_x_current = best_x_bo
        best_x_local = best_x_bo
        best_y_local = best_y_bo

        while local_step >= 2:
            print("\n===== STAGE 3: Z REFINEMENT =====")
            best_x_z, best_y_z = self.run_z_refinement(
                start_x=best_x_current,
                z_dim=4,
                z_step=local_z_step,
                z_points=20,
            )
            print("\n===== STAGE 4: LOCAL COORDINATE REFINEMENT =====")
            best_x_local, best_y_local = self.run_local_refinement(
                start_x=best_x_z,
                initial_step=local_step,
                rounds=local_rounds,
            )
            local_step /= 2
            local_z_step /= 2
            improvement = best_y_local - previous_best
            if improvement < IMPROVEMENT_THRESHOLD:
                print("No significant improvement. Stopping refinement loop.")
                break

            previous_best = best_y_local
            best_x_current = best_x_local

        print("\n===== STAGE 5: REPEATED VALIDATION =====")
        validated_voltage, validated_std = self.validate_position(
            best_x_local,
            n_measurements=validation_measurements,
            move_away=False
        )

        self.best_x_real = np.asarray(best_x_local, dtype=float).copy()
        self.best_value = float(validated_voltage)
        self.progress = 100

        print("\n===== OPTIMIZATION COMPLETE =====")
        print("Final best position:", self.best_x_real)
        print(f"Final validated voltage: {validated_voltage:.6f} ± {validated_std:.6f}")

        self.close_hardware()
        return self.best_x_real, validated_voltage, validated_std

    #  ---- STEP1:  global scan ----

    def generate_dataset(self, load_only=False, n_samples=5000, include_z=False):
        if not load_only:
            print("Starting Latin Hypercube sampling...")

            run_kwargs = {"sample_size": n_samples}
            run_kwargs["picoscope"] = self.pico

            if self.min_boundary is not None:
                run_kwargs["min_boundary"] = self.min_boundary
            if self.max_boundary is not None:
                run_kwargs["max_boundary"] = self.max_boundary

            self.data_acq.run(**run_kwargs)

            print("Dataset generated:", self.csv_path)

        return self.data_acq.load_dataset(include_z=include_z)

    def run_global_scan(self, n_samples=500, load_only=False, include_z=False):
        X, y = self.generate_dataset(
            load_only=load_only,
            n_samples=n_samples,
            include_z=include_z,
        )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(X) == 0 or len(y) == 0:
            raise RuntimeError("Global scan returned an empty dataset.")

        best_idx = int(np.argmax(y))
        self.best_x_real = X[best_idx].copy()
        self.best_value = float(y[best_idx])

        self.history.append(
            {
                "stage": "global_scan",
                "best_x": self.best_x_real.copy(),
                "best_voltage": self.best_value,
                "n_samples": int(len(y)),
            }
        )

        print("Global scan dataset shape:", X.shape, y.shape)
        print("Global scan best position:", self.best_x_real)
        print(f"Global scan best voltage: {self.best_value:.6f}")

        return X, y

    # ---- STEP2: GP Bayesian optimization -----

    def run_bayesian_optimization(self, X_init, y_init, n_iterations=50):
        X_init = np.asarray(X_init, dtype=float)
        y_init = np.asarray(y_init, dtype=float).reshape(-1)

        if X_init.ndim != 2:
            raise ValueError("X_init must be a 2D array: shape (n_samples, n_dimensions).")
        if len(X_init) != len(y_init):
            raise ValueError("X_init and y_init must have the same number of samples.")

        X_norm = self.gp_model.normalize_X(X_init)

        self.gp_model.train(X_norm, y_init)
        self.gp_model.X_data = X_norm
        self.gp_model.y_data = y_init

        best_idx = int(np.argmax(y_init))
        best_x = X_init[best_idx].copy()
        best_y = float(y_init[best_idx])

        self.best_x_real = best_x.copy()
        self.best_value = best_y

        for i in range(n_iterations):
            print(f"\nBO iteration {i + 1}/{n_iterations}")

            t0 = time.time()
            next_x_norm = self.gp_model.suggest_next_point(n_iterations)
            next_x = self.gp_model.denormalize_X(next_x_norm)
            next_x = np.asarray(next_x, dtype=float).reshape(-1)
            t1 = time.time()

            next_x_full = np.append(next_x, self.center[4])
            voltage, voltage_std = self._measure(next_x_full)
            t2 = time.time()

            next_x_norm = self.gp_model.normalize_X(next_x.reshape(1, -1))
            self.gp_model.update(next_x_norm, voltage)
            t3 = time.time()

            if voltage > best_y:
                best_y = float(voltage)
                best_x = next_x.copy()
                self.best_x_real = best_x.copy()
                self.best_value = best_y

            self.history.append(
                {
                    "stage": "bayesian_optimization",
                    "iteration": i + 1,
                    "x_real": next_x.copy(),
                    "x_full": next_x_full.copy(),
                    "voltage": float(voltage),
                    "voltage_std": float(voltage_std),
                    "best_x": best_x.copy(),
                    "best_voltage": best_y,
                    "suggest_time": t1 - t0,
                    "measure_time": t2 - t1,
                    "update_time": t3 - t2,
                }
            )

            self.progress = int((i + 1) / max(n_iterations, 1) * 100)

            print("Suggested position:", next_x)
            print(f"Voltage: {voltage:.6f} ± {voltage_std:.6f}")
            print("Best BO position:", best_x)
            print(f"Best BO voltage: {best_y:.6f}")
            print(f"BO progress: {self.progress}%")

        return best_x, best_y

    # ----- STEP3: z-optimization -----
    def run_z_refinement(
        self,
        start_x,
        z_dim=4, #[m1, m2, m3, m4, z]
        z_step=100,
        z_points=50,
    ):
        """
        Divide z range into equal spaced points and just go over them and measure voltage.
        """
        current_x = np.asarray(start_x, dtype=float).reshape(-1)

        if len(current_x) <= z_dim:
            raise ValueError(f"z_dim={z_dim} but position only has {len(current_x)} dimensions. ")

        z_center = current_x[z_dim]

        offsets = np.linspace(
            -z_step * (z_points // 2),
            z_step * (z_points // 2),
            z_points,
        )

        best_x = current_x.copy()
        best_y = -np.inf
        best_std = 0.0

        print("Starting z refinement around z =", z_center)

        for offset in offsets:
            candidate_x = current_x.copy()
            candidate_x[z_dim] = z_center + offset
            #Important: refinment is allowed to go off manual boundaries but not physical boundaries (0, 4096) 
            voltage, voltage_std = self._measure(candidate_x, settle_time=self.settle_time, refinement=True)
            self.history.append(
                {
                    "stage": "z_scan",
                    "z": float(candidate_x[z_dim]),
                    "voltage": float(voltage),
                    "voltage_std": float(voltage_std),
                    "x_real": candidate_x.copy(),
                    "z_dim": int(z_dim),
                }
            )
            print(
                f"z={candidate_x[z_dim]:.1f}, "
                f"voltage={voltage:.6f} ± {voltage_std:.6f}"
            )

            if voltage > best_y:
                best_y = float(voltage)
                best_std = float(voltage_std)
                best_x = candidate_x.copy()

        self.history.append(
            {
                "stage": "z_refinement",
                "x_real": best_x.copy(),
                "voltage": best_y,
                "voltage_std": best_std,
                "z_dim": int(z_dim),
                "z_step": float(z_step),
                "z_points": int(z_points),
            }
        )

        self.best_x_real = best_x.copy()
        self.best_value = best_y

        print("Best z-refined position:", best_x)
        print(f"Best z-refined voltage: {best_y:.6f} ± {best_std:.6f}")

        return best_x, best_y
    # ---- STEP4: local refinement ------

    def run_local_refinement(
        self,
        start_x,
        initial_step=40,
        rounds=6,
        min_step=1,
    ):
        """
        For each servo dimension, the method tries +/- step. If an improvement is
        found, it moves there. If no dimension improves, step size is halved.
        """
        current_x = np.asarray(start_x, dtype=float).reshape(-1)
        current_y, current_std = self._measure(current_x, settle_time=self.settle_time, refinement=True)
        print("Starting local refinement at:", current_x)
        print(f"Initial local voltage: {current_y:.6f} ± {current_std:.6f}")

        step = float(initial_step)

        for r in range(rounds):
            print(f"\nLocal refinement round {r + 1}/{rounds}")
            print(f"Current step size: {step:.3f}")

            improved_this_round = False

            for dim in range(4):#len(current_x)):
                candidates = []

                for direction in [-1, 1]:
                    candidate_x = current_x.copy()
                    candidate_x[dim] += direction * step
                    voltage, voltage_std = self._measure(candidate_x, settle_time=self.settle_time, refinement=True)
                    candidates.append((candidate_x, voltage, voltage_std, direction))

                    print(
                        f"Dim {dim}, direction {direction:+d}: "
                        f"{voltage:.6f} ± {voltage_std:.6f}"
                    )

                best_candidate_x, best_candidate_y, best_candidate_std, best_direction = max(
                    candidates,
                    key=lambda item: item[1],
                )

                if best_candidate_y > current_y + current_std:
                    current_x = best_candidate_x.copy()
                    current_y = float(best_candidate_y)
                    current_std = float(best_candidate_std)
                    improved_this_round = True

                    print("Improved position:", current_x)
                    print(f"Improved voltage: {current_y:.6f} ± {current_std:.6f}")

                #This will make up for the fact that hysteresis might make moving to the true best result too hard
                if best_candidate_y >= (current_y - current_std): 
                    current_x = best_candidate_x.copy()
                    current_std = float(best_candidate_std)
            self.history.append(
                {
                    "stage": "local_refinement",
                    "round": r + 1,
                    "step": float(step),
                    "x_real": current_x.copy(),
                    "voltage": float(current_y),
                    "voltage_std": float(current_std),
                    "improved": bool(improved_this_round),
                }
            )

            if not improved_this_round:
                step /= 2.0
                print("No improvement. Reducing step to:", step)

            if step < min_step:
                print("Minimum local step reached. Stopping local refinement.")
                break

        self.best_x_real = current_x.copy()
        self.best_value = float(current_y)

        print("\nLocal refinement best position:", self.best_x_real)
        print(f"Local refinement best voltage: {self.best_value:.6f}")

        return current_x, current_y

    # ---- STEP4: validation ----

    def validate_position(self, x, n_measurements=10, move_away=False, away_step=20):
        """
        Repeatedly measure a candidate best position.
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        voltages = []

        for i in range(n_measurements):
            if move_away:
                away_x = x.copy()
                away_x[0] += away_step
                self._move_servos(away_x, settle_time=self.settle_time, use_search_bounds=False)
            voltage, voltage_std = self._measure(x, settle_time=self.settle_time, refinement=True)
            voltages.append(voltage)

            print(
                f"Validation {i + 1}/{n_measurements}: "
                f"{voltage:.6f} ± {voltage_std:.6f}"
            )

        mean_voltage = float(np.mean(voltages))
        std_voltage = float(np.std(voltages))

        self.history.append(
            {
                "stage": "validation",
                "x_real": x.copy(),
                "mean_voltage": mean_voltage,
                "std_voltage": std_voltage,
                "n_measurements": int(n_measurements),
            }
        )

        return mean_voltage, std_voltage


    # Some helper functions 

    def _measure(self, x, oversampling=None, settle_time=None, refinement=False):
        """
        Move servos to position x and measure mean PicoScope voltage.
        Returns:mean_voltage, std_voltage
        """
        if self.pico is None:
            self.initialize_hardware()

        if oversampling is None:
            oversampling = self.oversampling

        if settle_time is None:
            settle_time = self.settle_time

        x = np.asarray(x, dtype=float).reshape(-1)
        if refinement: #For refinement it's not confined to determined boundaries but to physical ones 
            self._move_servos(x, settle_time=settle_time, use_search_bounds=False)
        else:
            self._move_servos(x, settle_time=settle_time)

        voltages = []
        for _ in range(oversampling):
            v, _ = self.pico.get_voltage()
            voltages.append(float(v))

        mean_voltage = float(np.mean(voltages))
        std_voltage = float(np.std(voltages))

        return mean_voltage, std_voltage


    def _move_servos(self, x, settle_time=None, use_search_bounds=True):
        if settle_time is None:
            settle_time = self.settle_time

        x = np.asarray(x, dtype=float).reshape(-1)
        if use_search_bounds:
            x = self._clip_to_boundaries(x)
        else:
            x = np.clip(x, 0, 4095)

        x_write = np.round(x).astype(int).tolist()

        with Servos() as servos:
            servos.write(x_write)
            time.sleep(settle_time)


    def _clip_to_boundaries(self, x, dims=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        if dims is None:
            dims = len(x)
            
        if self.min_boundary is not None:
            min_b = np.asarray(self.min_boundary, dtype=float)[:dims]
            x = np.maximum(x, min_b)

        if self.max_boundary is not None:
            max_b = np.asarray(self.max_boundary, dtype=float)[:dims]
            x = np.minimum(x, max_b)

        return x
