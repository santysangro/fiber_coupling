import csv
import numpy as np
import time 
from configuration import *
from controller.picoscope import Picoscope
from controller.servos import Servos
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc
import pandas as pd

class DataAcquisition:

    def __init__(self, data_path= f"Data/data_22_4.csv", search_type="LatinHypercube", settle_time=1):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.search_type = search_type
        self.settle_time = settle_time
        self.num_samples = 100



    def search_structure(self, min_boundary, max_boundary):
        """
        Open to implement other search types.
        """
        if self.search_type == "LatinHypercube":
            min_boundary = np.asarray(min_boundary, dtype=float)[:4]
            max_boundary = np.asarray(max_boundary, dtype=float)[:4]

            sampler = LatinHypercube(d=4)
            X = sampler.random(n=self.num_samples)
            X = qmc.scale(X, min_boundary, max_boundary)

            return X.astype(int)



    def run(self, min_boundary=None, max_boundary=None, sample_size=10, picoscope=None):
        self.num_samples = sample_size

        if min_boundary is None:
            min_boundary = [0, 0, 0, 0]
        if max_boundary is None:
            max_boundary = [4095, 4095, 4095, 4095]

        min_boundary = np.asarray(min_boundary, dtype=float)
        max_boundary = np.asarray(max_boundary, dtype=float)

        angular_min = min_boundary[:4]
        angular_max = max_boundary[:4]

        fixed_z = SERVOS_TEST_POS[4]

        try:
            with open(self.data_path, mode="w", newline="") as file:
                writer = csv.writer(file)

                X = self.search_structure(angular_min, angular_max)

                motor_headers = [f"m{i}" for i in range(4)]
                writer.writerow(motor_headers + ["voltage_mV", "std_mV"])

                with Servos() as servos:
                    for i, pos in enumerate(X):
                        print(f"Point {i+1}/{self.num_samples}: {pos}")

                        try:
                            pos = np.clip(pos, angular_min, angular_max)

                            full_pos = np.append(pos, fixed_z)
                            full_pos = np.round(full_pos).astype(int).tolist()

                            servos.write(full_pos)
                            time.sleep(self.settle_time)

                            voltage, std = picoscope.get_voltage()
                            writer.writerow([int(p) for p in pos] + [voltage, std])

                        except Exception as e:
                            print(f"ERROR at point {i}: {e}")
                            writer.writerow([int(p) for p in pos] + [np.nan, np.nan])

                print("Wrote training data to file")

        except Exception as e:
            print("Error writing to file:", e)

        print("Finished generating training data")

    def load_dataset(self):
        """
        Load dataset from CSV
        Assumes: first N (in this case 4) columns = motor positions, last column = voltage
        """
        data = pd.read_csv(self.data_path).values
        X = data[:, :-2]   # all columns except last
        y = data[:, 4]    # second-to-last column (voltage) [last column is sd]

        return X, y

