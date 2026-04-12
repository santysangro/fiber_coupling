import csv
import numpy as np
import time 

from configuration import *
from controller.picoscope import Picoscope
from controller.servos import Servos

from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc

N_SAMPLES = 5000
SETTLE_TIME = 0.03

class DataAcquisition:

    def __init__(self, data_path= f"Data/data_12_4.csv", search_type="random"):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.search_type = search_type



    def search_structure(self, min_boundary, max_boundary, n_samples=N_SAMPLES):
        if self.search_type == "LatinHypercube":
            sampler = LatinHypercube(d=4) #I think d is the number of motors?
            X = sampler.random(n=n_samples)

            """
            TO DO: Compute the quality of the sample using the discrepancy criterion.
            Whatever discrepancy is...
            qmc.discrepancy(sample)
            0.0196... # random
            """
            X = qmc.scale(X, min_boundary, max_boundary)

            return X.astype(int)



    def run(self, min_boundary=[0 for _ in range(len(STS_IDS))], max_boundary=[4095 for _ in range(len(STS_IDS))], sample_size=10):
        picoscope = Picoscope()
        
        try:
            with open(self.data_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                X = self.search_structure(min_boundary, max_boundary)
                writer.writerow(["m0", "m1", "m2", "m3","voltage_mV", "std_mV"])

                with Servos() as servos:
                    for i, pos in enumerate(X):
                        print(f"Point {i+1}/{N_SAMPLES}: {pos}")

                        try:
                            pos = np.clip(pos, min_boundary, max_boundary) #JUST IN CASE SOMETHING WENT WRONG
                            servos.write(pos)
                            time.sleep(SETTLE_TIME)
                            voltage, std = picoscope.get_voltage()
                   
                            writer.writerow([
                                        int(pos[0]), int(pos[1]),
                                        int(pos[2]), int(pos[3]),
                                        voltage, std
                                    ])
                        except Exception as e:
                            print(f"ERROR at point {i}: {e}")
                            writer.writerow([
                                int(pos[0]), int(pos[1]),
                                int(pos[2]), int(pos[3]),
                                np.nan, np.nan
                            ])
                    print("Wrote training data to file")

        except Exception as e:
            print("Error writing to file: ", e)

        print("Finished generating training data. Closing picoscope...")

        picoscope.close_device()





import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def grouped_pca_landscape(data):
    """
    data format:
    [m0, m1, m2, m3, voltage, std]
    """

    motors = data[:, :4]
    voltage = data[:, 4]

    # Split into two motor groups
    group_A = motors[:, :2]   # m0, m1
    group_B = motors[:, 2:]   # m2, m3

    # PCA per group (2D -> 1D)
    pca_A = PCA(n_components=1)
    pca_B = PCA(n_components=1)

    A_1d = pca_A.fit_transform(group_A).flatten()
    B_1d = pca_B.fit_transform(group_B).flatten()

    print("Explained variance A:", pca_A.explained_variance_ratio_)
    print("Explained variance B:", pca_B.explained_variance_ratio_)

    # 2D landscape
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(A_1d, B_1d, c=voltage, s=8, cmap="viridis")
    plt.colorbar(sc, label="Voltage (mV)")
    plt.xlabel("PCA(m0, m1)")
    plt.ylabel("PCA(m2, m3)")
    plt.title("Grouped PCA landscape of fiber coupling (voltage range=5V)")
    plt.savefig("Data/PCA.png", dpi=300)

    return A_1d, B_1d, pca_A, pca_B




if __name__ == "__main__":
    #get_data = DataAcquisition(search_type="LatinHypercube")
    #get_data.run()
    data = pd.read_csv("Data/data_12_4.csv").values
    grouped_pca_landscape(data=data)