import numpy as np
from configuration import SERVOS_TEST_POS
from controller.picoscope import Picoscope
from controller.servos import Servos
import matplotlib.pyplot as plt
import csv
import time

class Scan2D:
    def __init__(self):
        self.picoscope = Picoscope(voltage_range='PS2000_2V')

    def run_scan(self, motor_x=0, motor_y=2,
                 range_x=1000, range_y=1000,
                 steps_x=5, steps_y=5,
                 settle_time=0.02):

        with Servos() as servos:
            filename = f"Data/scan2D_m{motor_x}_m{motor_y}.csv"

            base_pos = np.array(SERVOS_TEST_POS.copy())

            # Define scan ranges centered around base
            x_vals = np.linspace(base_pos[motor_x] - range_x,
                                 base_pos[motor_x] + range_x,
                                 steps_x)

            y_vals = np.linspace(base_pos[motor_y] - range_y,
                                 base_pos[motor_y] + range_y,
                                 steps_y)

            data = []

            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["motor_0", "motor_1", "voltage"])

                for i, x in enumerate(x_vals):
                    print(f"Row {i+1}/{steps_x}")

                    for j, y in enumerate(y_vals):
                        pos = base_pos.copy()
                        pos[motor_x] = int(x)
                        pos[motor_y] = int(y)

                        servos.write(pos)
                        time.sleep(settle_time)

                        voltage, _ = self.picoscope.get_voltage()

                        writer.writerow([x, y, voltage])
                        data.append((x, y, voltage))

        return np.array(data), x_vals, y_vals

    def plot_heatmap(self, data, x_vals, y_vals, motor_x, motor_y):
        # Reshape into grid
        Z = data[:, 2].reshape(len(x_vals), len(y_vals))

        plt.figure(figsize=(8, 6))

        # Heatmap
        plt.imshow(Z,
                   extent=[y_vals[0], y_vals[-1], x_vals[0], x_vals[-1]],
                   origin='lower',
                   aspect='auto')

        plt.colorbar(label="Voltage (mV)")
        plt.xlabel(f"Motor {motor_y}")
        plt.ylabel(f"Motor {motor_x}")
        plt.title("2D Scan Heatmap")

        plt.savefig(f"Data/heatmap_m{motor_x}_m{motor_y}.png", dpi=300)
        plt.close()

    def plot_contour(self, data, x_vals, y_vals, motor_x, motor_y):
        Z = data[:, 2].reshape(len(x_vals), len(y_vals))

        X, Y = np.meshgrid(y_vals, x_vals)

        plt.figure(figsize=(8, 6))

        plt.contourf(X, Y, Z, levels=30)
        plt.colorbar(label="Voltage (mV)")

        plt.xlabel(f"Motor {motor_y}")
        plt.ylabel(f"Motor {motor_x}")
        plt.title("2D Scan Contour")

        plt.savefig(f"Data/contour_m{motor_x}_m{motor_y}.png", dpi=300)
        plt.close()

    def run(self, motorx=0, motory=1):
        data, x_vals, y_vals = self.run_scan(
            motor_x=motorx,
            motor_y=motory,
            range_x=500,
            range_y=500,
            steps_x=50,
            steps_y=50
        )

        self.plot_heatmap(data, x_vals, y_vals, motorx, motory)
        self.plot_contour(data, x_vals, y_vals, motorx, motory)


if __name__ == "__main__":
    scan = Scan2D()
    scan.run()
    scan.picoscope.close_device()