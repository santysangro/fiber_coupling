
from controller.servos import Servos
from controller.picoscope import Picoscope
from configuration import *
import time

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
    pico = Picoscope(voltage_range='PS2000_2V')
    vol = pico.get_voltage(CHANNEL='A')
    pico.get_voltage(CHANNEL='B') 
    return vol


from controller.fiber_coupling import FiberCoupling
if __name__ == '__main__':
    f = FiberCoupling()
    time_0 = time.time()
    best_pos = f.run_optimization(n_iterations=300, dataset_len=700)
    time_1 = time.time()
    print("Duration: ", time_1 - time_0)
    test_servos(write=True, dummy=[best_pos[0], best_pos[1], best_pos[2], best_pos[3]])
    x = test_picoscope()
    print(x)

""" GP:
Learned kernel: 0.727**2 * RBF(length_scale=100) + WhiteKernel(noise_level=0.119)
Best motor position: [2604  812  649  334]
Predicted voltage: 347.8436954633137
ACTUAL VALUES: 
Servo_ID: 005, Pos: 2604, Speed: 000
Servo_ID: 006, Pos: 812, Speed: 000
Servo_ID: 007, Pos: 649, Speed: 000
Servo_ID: 008, Pos: 334, Speed: 000
Voltage: 23.68541520432142 mV

NEW ONE:
(.venv) santy@MacBook-Pro-de-Santiago fiber_coupling % python gaussian_process.py 
Learned kernel: 0.773**2 * RBF(length_scale=116) + WhiteKernel(noise_level=0.143)
Best motor position: [2291 1105  588  469]
Predicted voltage: 367.4722594468197
ACTUAL: 
Servo_ID: 005, Pos: 2291, Speed: 000
Servo_ID: 006, Pos: 1105, Speed: 000
Servo_ID: 007, Pos: 588, Speed: 000
Servo_ID: 008, Pos: 469, Speed: 000
Voltage: 1779.474166081728 mV
Voltage: 2000.0 mV
"""