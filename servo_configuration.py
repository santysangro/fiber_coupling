import serial
import time
"""
This is a file to change Feetech Servo configurations, you use the ID address in their documentation to know what you wanna change.
At the moment it is set at 5 to change servo name cause you need to give them a name and make it permanent to be able to use more than one at a time.
"""# ==== CONFIG ====
PORT = "COM3"    
BAUD = 1000000
OLD_ID = 1
NEW_ID = 4

ser = serial.Serial(PORT, BAUD, timeout=0.1)

def write_register(servo_id, address, value):
    """
    Generic write command for Feetech TTL servos
    """
    # Example packet (adjust if your protocol differs)
    packet = bytearray([
        0xFF, 0xFF,
        servo_id,
        0x05,          # length
        0x03,          # WRITE instruction
        address,
        value & 0xFF,
        (value >> 8) & 0xFF
    ])
    
    checksum = (~sum(packet[2:]) & 0xFF)
    packet.append(checksum)

    ser.write(packet)
    time.sleep(0.02)
def save_eeprom(ser, servo_id):
    packet = bytearray([
        0xFF, 0xFF,
        servo_id,
        0x02,      # length
        0x06       # SAVE/EEPROM instruction (common Feetech)
    ])
    checksum = (~sum(packet[2:]) & 0xFF)
    packet.append(checksum)

    ser.write(packet)
    time.sleep(0.05)
    
# 1. Change ID
write_register(OLD_ID, 5, NEW_ID)   # (common ID address = 5)
save_eeprom(ser, NEW_ID)
time.sleep(0.1)


print("Configuration sent.")