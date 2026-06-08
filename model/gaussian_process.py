from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import numpy as np
from model.run_experiment_fiber_coupling import run_experiment
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


class GaussianProcessModel:
    def __init__(self, min_boundary, max_boundary):
    
        self.kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) *
            Matern(length_scale=1.0, nu=1.5) +
            WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)
        self.min_boundary = np.asarray(min_boundary, dtype=float)
        self.max_boundary = np.asarray(max_boundary, dtype=float)
        self.iteration = 0
        self.motor_num = 4 #len(STS_IDS-1)
        self.best_value = -np.inf
        self.best_x = None
        self.lock_mode = False
        self.lock_threshold = 100
        self.X_data = None
        self.y_data = None
        pass

    def normalize_X(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.min_boundary) / (self.max_boundary - self.min_boundary)
    
    def denormalize_X(self, X):
        X = np.asarray(X, dtype=float)
        real_x = self.min_boundary + X * (self.max_boundary - self.min_boundary)
        return np.clip(real_x, self.min_boundary, self.max_boundary).astype(int)
    
    
    def suggest_next_point(self, max_iterations=100):
        def acquisition_ucb(x):
            x = np.clip(x, 0, 1).reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            
            if self.iteration > 0.9 * max_iterations:
                kappa = 0.001
            else:
                kappa = max(0.05, 1 * (1 - self.iteration / max_iterations))
            return -(mean + kappa * std)[0]

        best_res = None

        for _ in range(20):
            if self.best_x is not None:
                if self.iteration > 0.9 * max_iterations:
                    x0 = self.normalize_X(self.best_x)
                    x0 = x0 + np.random.normal(0, 0.005, size=self.motor_num)
                    x0 = np.clip(x0, 0, 1)
                elif self.iteration > 0.7 * max_iterations:

                    x0 = self.normalize_X(self.best_x)
                    x0 = x0 + np.random.normal(0, 0.01, size=self.motor_num)
                    x0 = np.clip(x0, 0, 1)
                else:
                    x0 = np.random.uniform(0, 1, self.motor_num)
            else:
                x0 = np.random.uniform(0, 1, self.motor_num)

            res = minimize(
                acquisition_ucb,
                x0=x0,
                bounds=[(0, 1)] * self.motor_num,
                method="L-BFGS-B",
            )

            candidate = np.clip(res.x, 0, 1)

            if self.X_data is not None:
                distances = np.linalg.norm(
                    self.X_data - candidate.reshape(1, -1),
                    axis=1,
                )

                if np.min(distances) < 0.01:
                    continue

            if best_res is None or res.fun < best_res.fun:
                best_res = res

        if best_res is None:
            return np.random.uniform(0, 1, self.motor_num)

        return np.clip(best_res.x, 0, 1)
    

    def train(self, X, y):
        self.gp.fit(X, y)
        print("Learned kernel:", self.gp.kernel_) #It's good to print kernel to evaluate if it has learned or not
        #You make a new GP so that it is faster to train with no n_restarts
        self.gp = GaussianProcessRegressor(kernel= self.gp.kernel_, normalize_y=True)
        self.gp.fit(X,y)

        


    def update(self, X_new, y_new):
            """
            Add new observation and retrain GP.
            """
            X_new = np.atleast_2d(X_new)
            y_new = np.atleast_1d(y_new)

            if self.X_data is None:
                self.X_data = X_new
                self.y_data = y_new
            else:
                self.X_data = np.vstack((self.X_data, X_new))
                self.y_data = np.append(self.y_data, y_new)

            # Track best
            if y_new > self.best_value:
                self.best_value = y_new
                self.best_x = self.denormalize_X(X_new[0])

            # Retrain GP
            self.gp.fit(self.X_data, self.y_data)

            self.iteration += 1