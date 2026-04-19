from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import numpy as np
from controller.servos import Servos
from controller.picoscope import Picoscope
from Model.run_experiment_fiber_coupling import run_experiment
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


class GaussianProcessModel:
    def __init__(self):
    
        self.kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) *
            #RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e3)) +
            Matern(length_scale=1.0, nu=1.5) +
            WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-5, 1e1))
        )
        self.gp = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5,normalize_y=True)
        self.bounds = 4095  # motor limits
        self.iteration = 0
        self.motor_num = 4
        self.best_value = -np.inf
        self.best_x = None
        self.lock_mode = False
        self.lock_threshold = 100
        pass

    def normalize_X(self, X):
        return X/self.bounds
    
    def denormalize_X(self, X):
        return np.clip(X * self.bounds, 0, self.bounds).astype(int)
    
    
    def acquisition_ucb(self, gp, X, kappa=2):
        mean, std = gp.predict(X, return_std=True)
        return mean + kappa * std


    def suggest_next_point(self):
        def objective(x):
                x = np.clip(x, 0, 1).reshape(1, -1)
                mean, std = self.gp.predict(x, return_std=True)
                return -(mean + 2.0 * std)
            
        if self.iteration > 74:
            intial = self.normalize_X(self.best_x)
            x0 = intial + np.random.normal(0, 0.005, size=self.motor_num)
            x0 = np.clip(x0, 0,1)
        else:
            x0 = np.random.uniform(0, 1, self.motor_num)

        res = minimize(objective, x0=x0, bounds=[(0,1)]*self.motor_num)
       
        return res.x
    

    def train_gp_from_csv(self):

        data = pd.read_csv(self.csv_path).values

        X = data[:, :self.motor_num]   # motors
        y = data[:, self.motor_num]    # voltage

        # normalize
        X = X / self.normalize_X

        print("Training GP...")
        self.gp_model.train(X, y)

        print("GP training complete.")

    
    def train(self, X, y):
        self.gp.fit(X, y)
        print("Learned kernel:", self.gp.kernel_)

        
        #return self



    """
    def run(self):
        picoscope = Picoscope()
        with Servos() as servos:
            pos = servos.read()
            initial_X = [p[1] for p in pos]
            initial_y, _  = picoscope.get_voltage()
            X_data = np.array(initial_X).reshape(1, -1)
            y_data = np.array([initial_y])
            
        for i in range(100):
            X_data_norm = self.normalize_X(X_data)
            self.train(X_data_norm, y_data)
            next_x_norm = self.suggest_next_point()

            next_x = self.denormalize_X(next_x_norm)
            print(next_x)
            #next_x = [next_x[0], next_x[1]]
            y = -run_experiment(next_x, picoscope)

            if y > self.best_value:
                self.best_value = y
                self.best_x = np.array(next_x.copy())
                print(self.best_x)
            if self.best_value > self.lock_threshold:
                self.lock_mode = True

            X_data = np.vstack([X_data, next_x])
            y_data = np.append(y_data, float(y))
            self.iteration = i
            print(f"Iteration {i}: voltage = {y}")
        print("Best position found: ", self.best_x, self.best_value)
        fixed_x = self.normalize_X(self.best_x[:2].reshape(1, -1))[0]
        #self.attempt_2(X_data, y_data, fixed_x, dim=0)
        #self.attempt_2(X_data, y_data, fixed_x,dim=1)
        self.refine_gradient_descent(self.best_x, picoscope)
        picoscope.close_device()


    def attempt_2(self, X_data, y_data, fixed_x, dim=1):

        # Sweep one dimension
        x_vals = np.linspace(0, 1, 4000)

        X_plot = np.tile(fixed_x, (len(x_vals), 1))
        X_plot[:, dim] = x_vals

        # GP prediction
        print(X_plot)
        y_mean, y_std = self.gp.predict(X_plot, return_std=True)

        # Extract observed points along that dim
        x_obs = X_data[:, dim]
        x_obs = self.normalize_X(x_obs, dim=101)
        # Plot
        plt.figure(figsize=(8, 5))

        plt.plot(x_vals, y_mean, label="GP mean")
        plt.fill_between(
            x_vals,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.3,
            label="± std"
        )

        plt.scatter(x_obs, y_data, color="black", alpha=0.6, label="Observations")

        plt.xlabel(f"Motor {dim} (normalized)")
        plt.ylabel("Voltage")

        plt.title(
            f"1D Slice | Kernel: {self.gp.kernel_}\n"
            f"LogMLL: {self.gp.log_marginal_likelihood(self.gp.kernel_.theta):.2f}"
        )

        plt.legend()
        plt.savefig(f"Data/gp_1d_dim{dim}.png", dpi=300)
        plt.show()
        plt.close()
    """

"""
import pandas as pd
import time
# load dataset
time1= time.time()
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
time2 = time.time()
print("Best motor position:", best_pos)
print("Predicted voltage:", best_voltage)
print("Duration:", time2-time1)
"""


import matplotlib.pyplot as plt

def plot_gp_slice(self, fixed_x=None, grid_size=40):
    if fixed_x is None:
        fixed_x = np.zeros(4)

    # vary first 2 dimensions
    x1 = np.linspace(0, 1, grid_size)
    x2 = np.linspace(0, 1, grid_size)

    X1, X2 = np.meshgrid(x1, x2)

    X_pred = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = fixed_x.copy()
            x[0] = X1[i, j]
            x[1] = X2[i, j]
            X_pred.append(x)

    X_pred = np.array(X_pred)

    mean, std = self.gp.predict(X_pred, return_std=True)

    Z = mean.reshape(grid_size, grid_size)

    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)
    plt.colorbar(label="GP mean (signal)")
    plt.scatter(
        self.normalize_X(self.X_data)[:, 0],
        self.normalize_X(self.X_data)[:, 1],
        c="red",
        s=10,
        label="samples"
    )
    plt.title("GP learned landscape (slice)")
    plt.legend()
    plt.show()




    def plot_gp_3d(self, X, y):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Create grid over input space
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        X1, X2 = np.meshgrid(x1, x2)

        # Stack into GP input
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

        # Predict GP mean + std
        y_mean, y_std = self.gp.predict(X_grid, return_std=True)
        Y_mean = y_mean.reshape(X1.shape)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Surface (GP mean)
        ax.plot_surface(X1, X2, Y_mean, alpha=0.7)

        # Observations
        ax.scatter(X[:, 0], X[:, 1], y, color="black", label="Observations")

        ax.set_xlabel("Motor 1 (normalized)")
        ax.set_ylabel("Motor 2 (normalized)")
        ax.set_zlabel("Voltage")

        ax.set_title(
            f"Kernel: {self.gp.kernel_}\n"
            f"Log-MLL: {self.gp.log_marginal_likelihood(self.gp.kernel_.theta):.2f}"
        )

        plt.legend()
        plt.savefig("Data/gp_3d.png", dpi=300)
        plt.show()
        plt.close()




    def refine_gradient_descent(self, start_x, picoscope, eps=5, lr=0.9, steps=5):
        """
        Real gradient descent using actual experiment measurements.
        
        eps: small motor step (in DAC units)
        lr: step size (in DAC units)
        """

        x = start_x.astype(float).copy()

        def measure(x):
            x_int = np.clip(x, 0, self.bounds).astype(int)
            y = -run_experiment(x_int, picoscope)
            return y

        history = []

        for step in range(steps):
            grad = np.zeros_like(x)

            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()

                x_plus[i] += eps
                x_minus[i] -= eps

                y_plus = measure(x_plus)
                y_minus = measure(x_minus)

                grad[i] = (y_plus - y_minus) / (2 * eps)

            # Gradient ascent (maximize voltage)
            x = x + lr * grad

            # Clip to valid range
            x = np.clip(x, 0, self.bounds)

            y = measure(x)
            history.append(y)

            print(f"REAL GD step {step}: x = {x.astype(int)}, voltage = {y:.4f}")

        return x.astype(int), history