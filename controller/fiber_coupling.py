
from model.data_acquisition import DataAcquisition
from model.gaussian_process import GaussianProcessModel
from controller.servos import Servos
from controller.picoscope import Picoscope
import numpy as np
import time

class FiberCoupling:
    def __init__(self, csv_path="Data/fiber_dataset.csv"):
        self.csv_path = csv_path
        self.data_acq = DataAcquisition(data_path=csv_path, search_type="LatinHypercube")
        self.gp_model = GaussianProcessModel()

        self.progress = 0

    def generate_dataset(self, n_samples=5000):

        print("Starting Latin Hypercube sampling...")

        self.data_acq.run(
            #min_boundary=min_boundary,
            #max_boundary=max_boundary,
            sample_size=n_samples
        )

        print("Dataset generated:", self.csv_path)
        return self.data_acq.load_dataset()



    def run_optimization(self, n_iterations=50, dataset_len=5000):
        """
        Bayesian optimization loop (controller layer)
        """
        X_init, y_init = self.generate_dataset(n_samples=dataset_len)   # <-- must return data

        # INITIAL GP TRAINING 
        X_init = self.gp_model.normalize_X(X_init)

        self.gp_model.train(X_init, y_init)
        self.gp_model.X_data = X_init
        self.gp_model.y_data = y_init

        print("Starting optimization...")

        self.progress = 0
        self.pico = Picoscope()
        for i in range(n_iterations):
            # 1. ask model
            next_x = self.gp_model.suggest_next_point(n_iterations)
            next_x = self.gp_model.denormalize_X(next_x)
            # 2. hardware interaction
            voltage = self._measure(next_x)
            # 3. update model
            n_next_x = self.gp_model.normalize_X(next_x)
            self.gp_model.update(n_next_x, voltage)
            # 4. UI progress
            self.progress = int((i + 1) / n_iterations * 100)

            print(f"Progress: {self.progress}% | Voltage: {voltage}")

        print("Optimization finished.")
        print("Best position found: ", self.gp_model.best_x, self.gp_model.best_value[0])
        return self.gp_model.best_x

    def fine_tune(self, n_iterations=200):

        with Servos() as servos:
            pos = servos.read()
            initial_X = [p[1] for p in pos]
            initial_y, _  = self._measure(initial_X)

        X_init = np.array(initial_X).reshape(1, -1)
        y_init = np.array([initial_y])
        self.gp_model.X_data = X_init
        self.gp_model.y_data = y_init
                
        self.progress = 0

        for i in range(n_iterations):

            # 1. ask model
            next_x = self.gp_model.suggest_next_point(n_iterations)
            next_x = self.gp_model.denormalize_X(next_x)

            # 2. hardware interaction
            voltage = self._measure(next_x)
        
            # 3. update model
            n_next_x = self.gp_model.normalize_X(next_x)
            self.gp_model.update(n_next_x, voltage)

            # 4. UI progress
            self.progress = int((i + 1) / n_iterations * 100)

            print(f"Progress: {self.progress}% | Voltage: {voltage}")

        print("Optimization finished.")
        print("Best position found: ", self.gp_model.best_x, self.gp_model.best_value[0])


    def _measure(self, x, oversampling=10):
        with Servos() as servos:
            servos.write(x)
        
        voltages = []
        for _ in range(oversampling):
            v, _ = self.pico.get_voltage()
            voltages.append(v)
        voltage = np.mean(voltages)
        return voltage