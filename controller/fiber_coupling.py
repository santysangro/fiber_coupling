
from model.data_acquisition import DataAcquisition
from model.gaussian_process import GaussianProcessModel
from controller.servos import Servos
from controller.picoscope import Picoscope
import numpy as np

class FiberCoupling:
    def __init__(self, csv_path="Data/fiber_dataset.csv"):
        self.csv_path = csv_path
        self.data_acq = DataAcquisition(data_path=csv_path, search_type="LatinHypercube")
        self.gp_model = GaussianProcessModel()

        self.progress = 0

    def generate_dataset(self, min_boundary, max_boundary, n_samples=5000):

        print("Starting Latin Hypercube sampling...")

        self.data_acq.run(
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            sample_size=n_samples
        )

        print("Dataset generated:", self.csv_path)

    def run_optimization(self, n_iterations=50):
        """
        Bayesian optimization loop (controller layer)
        """

        print("Starting optimization...")

        self.progress = 0

        for i in range(n_iterations):

            # 1. ask model
            next_x = self.gp_model.suggest_next_point()
            next_x = self.gp_model.denormalize_X(next_x)

            # 2. hardware interaction
            voltage = self._measure(next_x)

            # 3. update model
            nnext_x = self.gp_model.normalize_X(next_x)
            self.gp_model.update(nnext_x, voltage)

            # 4. UI progress
            self.progress = int((i + 1) / n_iterations * 100)

            print(f"Progress: {self.progress}% | Voltage: {voltage}")

        print("Optimization finished.")


    def _measure(self, x, oversampling=1):
        with Servos() as servos:
            servos.write(x)

        pico = Picoscope()
        voltages = []
        for _ in range(oversampling):
            v, _ = pico.get_voltage()
            voltages.append(v)
        pico.close_device()
        voltage = np.mean(voltages)
        return voltage