
from controller.servos import Servos
from controller.picoscope import Picoscope
from configuration import *

def test_servos(manual_input=False, write=False, dummy=[3030, 102, 980, 627]):
    # Read and write
    with Servos() as servos:
        positions = servos.read()
        for pos in positions:
            print("Servo_ID: %03d, Pos: %03d, Speed: %03d" %
                  (pos[0], pos[1], pos[2]))

        goal_positions = []
        if write:
            if manual_input:
                for sts_id in STS_IDS:
                    angle = int(input("Enter angle of servo %03d: " % sts_id))
                    goal_positions.append(angle)

                servos.write(goal_positions)
            else:
                servos.write(dummy)

def test_picoscope():
    pico = Picoscope(voltage_range='PS2000_5V')
    vol, sd = pico.get_voltage(CHANNEL='A')
    print(vol, sd)
    #pico.get_voltage(CHANNEL='B') 
    return vol


from controller.fiber_coupling import FiberCoupling
import numpy as np

if __name__ == '__main__':

    test_servos(write=True, dummy=[1880])
    test_picoscope()
    """
    min_bound = np.subtract(SERVOS_TEST_POS, [1500, 1500, 1500, 1500, 1000])
    max_bound = np.add(SERVOS_TEST_POS, [1500, 1500, 1500, 1500, 1000])

    f = FiberCoupling(min_boundary=min_bound, max_boundary=max_bound)
    time_0 = time.time()
    best_pos = f.run_full_optimization(global_samples=250, 
                                       bo_iterations=200, 
                                       local_step=30, 
                                       local_rounds=5, 
                                       validation_measurements=10, 
                                       load_global_scan=False)
    time_1 = time.time()
    print("Duration: ", time_1 - time_0)
    """