
from controller.servos import Servos
from controller.picoscope import Picoscope
from configuration import *


def test_servos(manual_input=False):
    # Read and write
    with Servos() as servos:
        positions = servos.read()
        for pos in positions:
            print("Servo_ID: %03d, Pos: %03d, Speed: %03d" %
                  (pos[0], pos[1], pos[2]))

        goal_positions = []
        
        if manual_input:
            for sts_id in STS_IDS:
                angle = int(input("Enter angle of servo %03d: " % sts_id))
                goal_positions.append(angle)

            servos.write(goal_positions)
        else:
            servos.write([3080, 60, 1529, 2884])

def test_picoscope():
    pico = Picoscope()
    pico.get_voltage(CHANNEL='A')
    pico.get_voltage(CHANNEL='B')

if __name__ == '__main__':
    test_picoscope()