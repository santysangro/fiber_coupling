from matplotlib import pyplot as plt
import numpy as np
import time

from configuration import SERVOS_TEST_POS, STS_IDS
from controller.picoscope import Picoscope
from controller.servos import Servos

from scservo_sdk import *  #  type: ignore


class ServosTest:
    def __init__(self):
        self.picoscope = Picoscope()
        self.counter = 1

    def collect_test_points(self):
        print(self.counter)
        self.counter += 1

        sts_goal_positions = SERVOS_TEST_POS.copy()
        sts_goal_positions[0] = np.random.randint(SERVOS_TEST_POS[0]-50, SERVOS_TEST_POS[0]+50)
        

        with Servos() as servos:
            servos.write(sts_goal_positions)
            time.sleep(0.5)

            # Move back to the original position after one second
            servos.write(SERVOS_TEST_POS)
            time.sleep(0.5)
            
        voltages = []
        for i in range(5):
            voltage, _ = self.picoscope.get_voltage()
            voltages.append(voltage)
        f_voltage = np.mean(voltages)
        print("Voltage after moving back: ", f_voltage)

        return f_voltage

    def run_test(self, iterations, filename='Data/small_servo_quality_0.png'):
        positions = np.array([self.collect_test_points()
                             for _ in range(iterations)])

        stats = [np.mean(positions), np.std(positions)]
        print(f"mean={stats[0]:.2f}, std={stats[1]:.2f}")

        plt.hist(positions, bins=20, alpha=0.7, color='b')
        plt.title(f'Noise')# Distribution of Servo 0 within a 1000 step range')
        plt.xlabel('Voltage')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    test = ServosTest()
    test.run_test(50)
    test.picoscope.close_device()