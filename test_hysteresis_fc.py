import numpy as np
from configuration import SERVOS_INTIAL_POS
from controller.picoscope import Picoscope
from controller.servos import Servos
import matplotlib.pyplot as plt
from scservo_sdk import * 
import csv

class HysteresisFCTest:
    def __init__(self):
        self.picoscope = Picoscope(voltage_range='PS2000_50MV')

    def test_single_motor(self, motor_idx, step=1):
        with Servos() as servos:
            filename = f"Data/sweep_data_motor_{motor_idx}_WAVEPLATE.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)

                writer.writerow(["iteration", "motor_pos", "voltage", "direction"])
                base_pos = np.array(SERVOS_INTIAL_POS.copy())
                #base_pos[motor_idx] -= 700
                pos = base_pos.copy()
                
                f_cmd, b_cmd = [], []
                f_meas, b_meas = [], []

                # Forward sweep
                for x in range(4096):
                    print(f'iteration {x}')
                    servos.write(pos)
                    #time.sleep(0.05)

                    voltage, _ = self.picoscope.get_voltage()

                    f_cmd.append(pos[motor_idx])
                    f_meas.append(voltage)

                    writer.writerow([x, pos[motor_idx], voltage, "forward"])
                    pos[motor_idx] += step
                print("Now back!")
                # Backward sweep
                for j in range(4096):
                    print(f'iteration {j}')
                    pos[motor_idx] -= step
                    servos.write(pos)
                    #time.sleep(0.05)
                    voltage, _ = self.picoscope.get_voltage()

                    writer.writerow([x, pos[motor_idx], voltage, "backwards"])

                    b_cmd.append(pos[motor_idx])
                    b_meas.append(voltage)


        return (
            np.array(f_cmd),
            np.array(b_cmd),
            np.array(f_meas),
            np.array(b_meas),
        )

    def plot_hysteresis(self, motor_idx, f_cmd, b_cmd, f_meas, b_meas):
        all_vals = np.concatenate([f_meas, b_meas])
        plt.figure()
        plt.plot(f_cmd, f_meas, label="Forward", marker='o')
        plt.plot(b_cmd, b_meas, label="Backward", marker='o')

        plt.xlabel(f"Motor {motor_idx} Command")
        plt.ylabel("Voltage")
        plt.title(f"Hysteresis - Motor {motor_idx}")
        plt.legend()
        plt.grid()
        plt.savefig(f"Data/full_hysteresis_fc_steps1_motor_{motor_idx}_WAVEPLATE.png", dpi=300)
        plt.close()

    def compute_hysteresis_metrics(self, f_cmd, b_cmd, f_meas, b_meas):
        """Compute max and mean hysteresis (1D voltage)"""

        # Interpolate backward measurements to forward command points
        b_interp = np.interp(f_cmd, b_cmd, b_meas)

        diff = np.abs(f_meas - b_interp)

        return {
            'H_max': diff.max(),
            'H_mean': diff.mean()
        }
    
    def run_all(self):
        num_motors = len(SERVOS_INTIAL_POS)

        for m in range(num_motors):
            print(f"\nTesting motor {m}")
            f_cmd, b_cmd, f_meas, b_meas = self.test_single_motor(m)

            # Plot curves
            self.plot_hysteresis(m, f_cmd, b_cmd, f_meas, b_meas)

            # Compute metrics
            metrics = self.compute_hysteresis_metrics(f_cmd, b_cmd, f_meas, b_meas)
            print(f"Motor {m}: H_max = {metrics['H_max']:.3f}, "
                f"H_mean = {metrics['H_mean']:.3f}")


if __name__ == '__main__':
    test = HysteresisFCTest()
    test.run_all()
    test.picoscope.close_device()