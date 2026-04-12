from scservo_sdk import *  # type: ignore
import math
import time

from controller.servos import Servos


def run_experiment(params, picoscope):
    yaw_1, yaw_2 = params
    sts_goal_positions = [
        math.floor(yaw_1),
        math.floor(yaw_2),
        1094,#math.floor(pitch_2),
        921#math.floor(pitch_1), #[2654, 515, 1094, 921]
    ]

    with Servos() as servos:
        servos.write(sts_goal_positions)

    time.sleep(0.2)
    voltage, _ = picoscope.get_voltage()
    cost = - voltage #abs(target_position - voltage) #squared error
    return cost
