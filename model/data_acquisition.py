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
            min_boundary = np.asarray(min_boundary, dtype=float)
            max_boundary = np.asarray(max_boundary, dtype=float)

            sampler = LatinHypercube(d=5)
            X = sampler.random(n=self.num_samples)
            X = qmc.scale(X, min_boundary, max_boundary)

            return X.astype(int)



    def run(self, min_boundary=None, max_boundary=None, sample_size=10, picoscope=None):
        self.num_samples = sample_size

        if min_boundary is None:
            min_boundary = [0, 0, 0, 0, 0]
        if max_boundary is None:
            max_boundary = [4095, 4095, 4095, 4095, 4095]

        try:
            with open(self.data_path, mode="w", newline="") as file:
                writer = csv.writer(file)

                X = self.search_structure(min_boundary, max_boundary)

                writer.writerow(["m0", "m1", "m2", "m3", "z", "voltage_mV", "std_mV"])

                with Servos() as servos:
                    for i, pos in enumerate(X):
                        print(f"Point {i+1}/{self.num_samples}: {pos}")

                        try:
                            pos = np.clip(pos, min_boundary, max_boundary)
                            pos = np.round(pos).astype(int).tolist()
                            servos.write(pos)
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

    def load_dataset(self, include_z=False):
        df = pd.read_csv(self.data_path)
        if include_z: 
            X = df[["m0", "m1", "m2", "m3", "z"]].values
        else: 
            X = df[["m0", "m1", "m2", "m3"]].values

        y = df["voltage_mV"].values

        return X, y

