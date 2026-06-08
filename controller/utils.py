from controller.picoscope import Picoscope
from controller.servos import Servos

import numpy as np
import time


def measure(x, pico=Picoscope(), oversampling=1, settle_time=1, refinement=False):
        """
        Move servos to position x and measure mean PicoScope voltage.

        Returns:
            mean_voltage, std_voltage
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if refinement:
            x = np.clip(x, 0, 4095)
        else:
            x = clip_to_boundaries(x)

        move_servos(x, settle_time=settle_time, use_search_bounds=False)
        voltages = []
        for _ in range(oversampling):
            v, _ = pico.get_voltage()
            voltages.append(float(v))

        mean_voltage = float(np.mean(voltages))
        std_voltage = float(np.std(voltages))

        return mean_voltage, std_voltage




def move_servos(self, x, settle_time=1, use_search_bounds=True):

        x = np.asarray(x, dtype=float).reshape(-1)
        if use_search_bounds:
            x = clip_to_boundaries(x)
        else:
            x = np.clip(x, 0, 4095)

        x_write = np.round(x).astype(int).tolist()

        with Servos() as servos:
            servos.write(x_write)
            time.sleep(settle_time)



def make_local_initial_dataset(self, center_x, picoscope=Picoscope(), radius=50):
        """
        Create a small local dataset around the current position:
            center, plus +/- radius in each servo dimension.
        """
        center_x = np.asarray(center_x, dtype=float).reshape(-1)
        center_x = clip_to_boundaries(center_x)

        X_local = []
        y_local = []

        voltage, voltage_std = measure(center_x, picoscope)
        X_local.append(center_x.copy())
        y_local.append(voltage)

        print(f"Center voltage: {voltage:.6f} ± {voltage_std:.6f}")

        for dim in range(len(center_x)):
            for direction in [-1, 1]:
                x_probe = center_x.copy()
                x_probe[dim] += direction * radius
                x_probe = self._clip_to_boundaries(x_probe)

                voltage, voltage_std = self._measure(x_probe)
                X_local.append(x_probe.copy())
                y_local.append(voltage)

                print(
                    f"Local probe dim {dim}, direction {direction:+d}: "
                    f"{voltage:.6f} ± {voltage_std:.6f}"
                )

        return np.asarray(X_local, dtype=float), np.asarray(y_local, dtype=float)

def clip_to_boundaries(self, x, dims=None):
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
