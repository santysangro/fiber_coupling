from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import numpy as np

class GaussianProcessModel:
    def __init__(self):
        self.kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) *
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
        self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=9,normalize_y=True)

        pass

    def train(self, x_train, y_train):
        self.gaussian_process.fit(x_train, y_train)
        print("Learned kernel:", self.model.kernel_)
        return self

    def predict(self, X, return_std=True):
        return self.model.predict(X, return_std=return_std)
    

def find_best_location(gp, n_samples=200000):
        X_candidates = np.random.randint(0, 4096, size=(n_samples, 4))

        y_pred, y_std = gp.predict(X_candidates)

        best_idx = np.argmax(y_pred)

        best_point = X_candidates[best_idx]
        best_value = y_pred[best_idx]

        return best_point, best_value









import pandas as pd

# load dataset
data = pd.read_csv("Data/focused_data_12_4.csv").values

gp = GaussianProcessModel()
# split
X_train = data[:, :4]   # m0, m1, m2, m3
y_train = data[:, 4]    # voltage

# optional: remove bad points (NaNs)
mask = ~np.isnan(y_train)

X_train = X_train[mask]
y_train = y_train[mask]
# train
gp.train(X_train, y_train)

# find max
best_pos, best_voltage = find_best_location(gp)

print("Best motor position:", best_pos)
print("Predicted voltage:", best_voltage)