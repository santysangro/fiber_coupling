from scservo_sdk import *  # type: ignore
import math
import time

from controller.servos import Servos


def run_experiment(params, picoscope, target_position):
    yaw_1, pitch_1, yaw_2, pitch_2 = params
    sts_goal_positions = [
        math.floor(yaw_1),
        math.floor(pitch_1),
        math.floor(yaw_2),
        math.floor(pitch_2),
    ]

    with Servos() as servos:
        servos.write(sts_goal_positions)

    time.sleep(0.2)
    voltage, _ = picoscope.get_voltage()
    cost = abs(target_position - voltage) #squared error
    return cost
