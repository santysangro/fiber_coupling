import time
import numpy as np

from configuration import *
from scservo_sdk import *


def update_configuration(servo_turns):
    with open("configuration.py", "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("SERVO_TURNS"):
            lines[i] = f"SERVO_TURNS = {servo_turns}\n" 

    with open("configuration.py", "w") as file:
        file.writelines(lines)

class Servos:
    """
    A class to manage and control servos using a specified port and packet handler.
    Methods
    -------
    __init__():
        Initializes the port handler and packet handler.
    __enter__():
        Opens the port and sets the baud rate.
    read_single_servo(sts_id):
        Reads the position and speed of a single servo.
    read():
        Reads the positions and speeds of all servos.
    write_single_servo(sts_id, sts_goal_position):
        Writes the goal position to a single servo and enables control parameters.
    write(sts_goal_positions):
        Writes the goal positions to all servos.
    __exit__(exc_type, exc_val, exc_tb):
        Closes the port.
    """

    def __init__(self):
        self.port_handler = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_END)

    def __enter__(self):
        self.port_handler.openPort()
        self.port_handler.setBaudRate(BAUDRATE)

        return self

    def read_single_servo(self, sts_id):
        scs_present_position_speed, scs_comm_result, scs_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_PRESENT_POSITION)
        if scs_comm_result != COMM_SUCCESS:
            print(self.packet_handler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print(self.packet_handler.getRxPacketError(scs_error))

        scs_present_position = SCS_LOWORD(scs_present_position_speed)
        scs_present_speed = SCS_HIWORD(scs_present_position_speed)

        servo_position = (sts_id, scs_present_position, SCS_TOHOST(scs_present_speed,
                                                                   15))
        scs_comm_result, scs_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_TORQUE_ENABLE, 0)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(scs_error))

        return servo_position

    def read(self):
        """
        Reads the positions of all servos specified in STS_IDS.
        Returns:
            list: A list of positions for each servo.
        """

        servo_positions = []

        for i, sts_id in enumerate(STS_IDS):
            pos = list(self.read_single_servo(sts_id))
            #pos[1] += SERVO_TURNS[i] * 4096
            servo_positions.append(pos)

        return servo_positions

    def write_single_servo(self, sts_id, sts_goal_position):
        sts_comm_result,  sts_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_TORQUE_ENABLE, 1)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

        # Enable STServo Acc
        sts_comm_result,  sts_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_GOAL_ACC, SCS_MOVING_ACC)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

        # Enable STServo Speed
        sts_comm_result,  sts_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_GOAL_SPEED, SCS_MOVING_SPEED)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

        # Write SCServo goal position
        sts_comm_result,  sts_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_GOAL_POSITION, sts_goal_position)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

        # enable P-control
        sts_comm_result,  sts_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_P_CONTROL, SCS_P_FACTOR)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

        # enable I-control
        sts_comm_result,  sts_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, sts_id, ADDR_STS_I_CONTROL, SCS_I_FACTOR)
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(sts_comm_result))
        elif sts_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(sts_error))

    def write(self, sts_goal_positions):
        max_deviation = max(abs(np.array(self.read())[:, 1] - np.array(sts_goal_positions)))
        sleep_time = max(0.1, max_deviation / 700 * 0.15)

        for i, (sts_id, sts_goal_position) in enumerate(zip(STS_IDS, sts_goal_positions)):
            self.write_single_servo(sts_id, sts_goal_position)

            if sts_goal_position > 4096 and sts_goal_position <= 8192:
                SERVO_TURNS[i] = 1
            elif sts_goal_position > 8192:
                SERVO_TURNS[i] = 2
            else:
                SERVO_TURNS[i] = 0
        
        update_configuration(SERVO_TURNS)

        time.sleep(sleep_time)

    def precise_write(self, sts_goal_positions):
        """
        Writes goal positions to servos in two steps with a delay in between.
        Args:
            sts_goal_positions (list): A list of goal positions for the servos.
        The method first writes each servo's goal position reduced by 50 units,
        waits for 0.7 seconds, and then writes the original goal positions.
        """

        for sts_id, sts_goal_position in zip(STS_IDS, sts_goal_positions):
            self.write_single_servo(sts_id, sts_goal_position-50)

        time.sleep(0.7)

        for i, (sts_id, sts_goal_position) in enumerate(zip(STS_IDS, sts_goal_positions)):
            self.write_single_servo(sts_id, sts_goal_position)

            if sts_goal_position > 4096 and sts_goal_position <= 8192:
                SERVO_TURNS[i] = 1
            elif sts_goal_position > 8192:
                SERVO_TURNS[i] = 2
            else:
                SERVO_TURNS[i] = 0
        
        update_configuration(SERVO_TURNS)

        time.sleep(0.7)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.port_handler.closePort()

        return False


if __name__ == '__main__':
    # Read and write
    with Servos() as servos:
        positions = servos.read()
        for pos in positions:
            print("Servo_ID: %03d, Pos: %03d, Speed: %03d" %
                  (pos[0], pos[1], pos[2]))

        goal_positions = []
        servos.write([3080, 60, 1529, 2884])#SERVOS_TEST_POS) # THE GOOD ONE [2135, 840, 1990, 950]

        #for sts_id in STS_IDS:
            #angle = int(input("Enter angle of servo %03d: " % sts_id))
            #goal_positions.append(angle)

        #servos.write(goal_positions)
