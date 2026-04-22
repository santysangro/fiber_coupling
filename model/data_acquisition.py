import csv
import numpy as np
import time 

from configuration import *
from controller.picoscope import Picoscope
from controller.servos import Servos
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc

SETTLE_TIME = 0.03

class DataAcquisition:

    def __init__(self, data_path= f"Data/data_22_4.csv", search_type="random"):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.search_type = search_type
        self.num_samples = 100



    def search_structure(self, min_boundary, max_boundary):
        if self.search_type == "LatinHypercube":
            sampler = LatinHypercube(d=4) #I think d is the number of motors?
            X = sampler.random(n=self.num_samples)

            """
            TO DO: Compute the quality of the sample using the discrepancy criterion.
            Whatever discrepancy is...
            qmc.discrepancy(sample)
            0.0196... # random
            """
            X = qmc.scale(X, min_boundary, max_boundary)

            return X.astype(int)



    def run(self, min_boundary=[0 for _ in range(len(STS_IDS))], max_boundary=[4095 for _ in range(len(STS_IDS))], sample_size=10):
        self.num_samples = sample_size
        picoscope = Picoscope()
        try:
            with open(self.data_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                X = self.search_structure(min_boundary, max_boundary)
                writer.writerow(["m0", "m1", "m2", "m3","voltage_mV", "std_mV"])

                with Servos() as servos:
                    for i, pos in enumerate(X):
                        print(f"Point {i+1}/{self.num_samples}: {pos}")

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

    def load_dataset(self):
        """
        Load dataset from CSV
        Assumes: first N columns = motor positions, last column = voltage
        """
        data = pd.read_csv(self.data_path).values

        X = data[:, :-2]   # all columns except last
        y = data[:, 4]    # last column (voltage)

        return X, y




### CHATGPT ###



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def grouped_pca_landscape(csv_path):
    """
    data format:
    [m0, m1, m2, m3, voltage, std]
    """
    data = pd.read_csv(csv_path).values
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
    plt.savefig("Data/focused_PCA.png", dpi=300)

    return A_1d, B_1d, pca_A, pca_B


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata


def pca_full_landscape(csv_path, save_path= None, make_heatmap=True, grid_res=200):
    """
    Full PCA on 4 motors → 2D embedding → voltage landscape

    CSV format:
    m0, m1, m2, m3, voltage, std
    """

    # -------------------
    # 1. Load data
    # -------------------
    data = pd.read_csv(csv_path).values
    motors = data[:, :4]
    voltage = data[:, 4]

    # -------------------
    # 2. PCA (4D → 2D)
    # -------------------
    pca = PCA(n_components=2)
    X = pca.fit_transform(motors)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance captured:", np.sum(pca.explained_variance_ratio_))

    # -------------------
    # 3. Scatter plot (raw landscape)
    # -------------------
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X[:, 0], X[:, 1],
        c=voltage,
        s=8,
        cmap="viridis"
    )
    plt.colorbar(sc, label="Voltage (mV)")
    plt.xlabel("PC1 (dominant motion mode)")
    plt.ylabel("PC2 (secondary motion mode)")
    plt.title("PCA Landscape of 4-motor fiber coupling")
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300)

    # -------------------
    # 4. Smooth interpolation (optional)
    # -------------------
    if make_heatmap:
        xi = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_res)
        yi = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_res)

        XI, YI = np.meshgrid(xi, yi)

        Z = griddata(
            X,
            voltage,
            (XI, YI),
            method="cubic"
        )

        plt.figure(figsize=(8, 6))
        plt.contourf(XI, YI, Z, levels=40, cmap="viridis")
        plt.colorbar(label="Voltage (mV)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Interpolated PCA Fiber Coupling Landscape")
        plt.tight_layout()
        plt.savefig(f"{save_path}_heatmap.png", dpi=300)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
import umap


def umap_landscape(csv_path, n_neighbors=30, min_dist=0.1):

    # -------------------
    # 1. Load data
    # -------------------
    data = pd.read_csv(csv_path).values
    motors = data[:, :4]
    voltage = data[:, 4]

    # -------------------
    # 2. Fit UMAP
    # -------------------
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42
    )

    X_umap = reducer.fit_transform(motors)

    # -------------------
    # 3. Plot landscape
    # -------------------
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=voltage,
        s=8,
        cmap="viridis"
    )

    plt.colorbar(sc, label="Voltage (mV)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP Fiber Coupling Landscape (4 motors → 2D)")
    plt.tight_layout()

    plt.savefig("Data/focused_umap_landscape.png", dpi=300)

    return X_umap, reducer




if __name__ == "__main__":
    get = False
    plot = True
    if get:
        get_data = DataAcquisition(data_path="Data/10000_focused_data_12_4.csv", search_type="LatinHypercube")
        boundary_l = [SERVOS_TEST_POS[i] - 500 for i in range(4)]
        boundary_u = [SERVOS_TEST_POS[i] + 500 for i in range(4)]
        get_data.run(min_boundary=boundary_l, max_boundary=boundary_u)
    if plot:
        pca_full_landscape("Data/10000_focused_data_12_4.csv",save_path="Data/100000_focused_PCA")
        #umap_landscape("Data/focused_data_12_4.csv")
        #grouped_pca_landscape("Data/focused_data_12_4.csv")
        #corr_matrix()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def motor_landscape_heatmap(
    csv_path,
    motor_x=0,
    motor_y=1,
    voltage_col=4,
    bins=50,
    method="mean",   # "mean" or "max"
    title=None,
    save_path=None):

    # --------------------
    # Load data
    # --------------------
    data = pd.read_csv(csv_path).values

    x = data[:, motor_x]
    y = data[:, motor_y]
    z = data[:, voltage_col]

    # --------------------
    # Bin edges
    # --------------------
    xedges = np.linspace(x.min(), x.max(), bins + 1)
    yedges = np.linspace(y.min(), y.max(), bins + 1)

    # --------------------
    # Output grid
    # --------------------
    grid = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))

    # --------------------
    # Fill bins
    # --------------------
    for xi, yi, zi in zip(x, y, z):

        ix = np.searchsorted(xedges, xi) - 1
        iy = np.searchsorted(yedges, yi) - 1

        if 0 <= ix < bins and 0 <= iy < bins:

            if method == "mean":
                grid[ix, iy] += zi
                counts[ix, iy] += 1

            elif method == "max":
                grid[ix, iy] = max(grid[ix, iy], zi)

    # finalize mean if needed
    if method == "mean":
        grid = grid / np.maximum(counts, 1)

    # --------------------
    # Plot
    # --------------------
    plt.figure(figsize=(8, 6))

    plt.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            xedges[0], xedges[-1],
            yedges[0], yedges[-1]
        ]
    )

    plt.colorbar(label="Voltage (mV)")

    plt.xlabel(f"Motor {motor_x}")
    plt.ylabel(f"Motor {motor_y}")

    if title is None:
        title = f"Motor Landscape (m{motor_x} vs m{motor_y}, {method})"

    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


    return grid

"""