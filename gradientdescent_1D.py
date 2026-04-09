import numpy as np
import time
from configuration import SERVOS_TEST_POS
from fibber_coupling.picoscope import Picoscope
from servos import Servos

class GradientDescent1D:
    def __init__(self, motor_id,
                 alpha=100, delta=10, avg_samples=5,
                 step_clip=100, settle_time=0.05):
        
        self.motor_id = motor_id
        self.picoscope = Picoscope()
        
        self.alpha = alpha
        self.delta = delta
        self.avg_samples = avg_samples
        self.step_clip = step_clip
        self.settle_time = settle_time

    def measure_avg(self):
        values = []
        for _ in range(self.avg_samples):
            voltage, _ = self.picoscope.get_voltage()
            values.append(voltage)
        return np.mean(values)

    def move_and_wait(self, servos, position):
        servos.write(position)
        time.sleep(self.settle_time)

    def estimate_gradient(self, theta, servos):
        # Measure at current position
        self.move_and_wait(servos, theta)
        V0 = self.measure_avg()

        # Measure at theta + delta
        new_theta = theta.copy()
        new_theta[0] += self.delta
        self.move_and_wait(servos, new_theta)
        V1 = self.measure_avg()
        grad = (V1 - V0) / self.delta
        return grad, V0

    def optimize(self, start_pos, iterations=20, verbose=True):
        theta = start_pos
        history = []
        decay = 0.9
        with Servos() as servos:
            for i in range(iterations):
                grad, V = self.estimate_gradient(theta, servos)

                # Compute step
                self.alpha = self.alpha * (decay ** i)
                step = self.alpha * grad
                #self.delta = int(self.delta * (decay ** i)) + 1
                step = np.clip(step, -self.step_clip, self.step_clip)

                if verbose:
                    print(f"Iter {i}: θ={theta:}, V={V:.4f}, grad={grad:.4f}, step={step:.2f}")

                theta[0] += int(step)
                history.append([theta[0], V])

        # Final move to best position
            self.move_and_wait(servos, theta)
        self.picoscope.close_device()
        return theta, history
    
gd = GradientDescent1D(
    motor_id=0,
    alpha=50,
    delta=5,
    avg_samples=5
)
start_pos = SERVOS_TEST_POS
random_v = np.random.randint(SERVOS_TEST_POS[0] - 100, SERVOS_TEST_POS[0] + 100)
start_pos[0] = random_v
print("START :", start_pos)
best_pos, history = gd.optimize(start_pos, iterations=10)
print(history)