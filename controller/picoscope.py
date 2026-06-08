from ctypes import byref, c_byte, c_int16, c_int32, sizeof
from time import sleep
import numpy as np
import time
from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok, adc2mV
from picosdk.PicoDeviceEnums import picoEnum
from configuration import PICOSCOPE_RANGE


class Picoscope:

    def __init__(self, samples=2000, oversampling=1, voltage_range=PICOSCOPE_RANGE):
        self.samples = samples
        self.oversampling = oversampling
        self.voltage_range = voltage_range
        self.device = ps2000.open_unit()
        self.res = ps2000.ps2000_set_channel(
        self.device.handle,
        picoEnum.PICO_CHANNEL['PICO_CHANNEL_A'],
        True,
        picoEnum.PICO_COUPLING['PICO_DC'],
        ps2000.PS2000_VOLTAGE_RANGE[self.voltage_range],
        )
        print("Result:", self.res)
        assert_pico2000_ok(self.res)

        self.res = ps2000.ps2000_set_channel(
        self.device.handle,
        picoEnum.PICO_CHANNEL['PICO_CHANNEL_B'],
        True,
        picoEnum.PICO_COUPLING['PICO_DC'],
        ps2000.PS2000_VOLTAGE_RANGE[self.voltage_range],
        )
        assert_pico2000_ok(self.res)


    def get_timebase(self, device, wanted_time_interval):
        current_timebase = 1

        old_time_interval = None
        time_interval = c_int32(0)
        time_units = c_int16()
        max_samples = c_int32()

        while ps2000.ps2000_get_timebase(
            device.handle,
            current_timebase,
            2000,
            byref(time_interval),
            byref(time_units),
            1,
            byref(max_samples)) == 0 \
            or time_interval.value < wanted_time_interval:

            current_timebase += 1
            old_time_interval = time_interval.value

            if current_timebase.bit_length() > sizeof(c_int16) * 8:
                raise Exception('No appropriate timebase was identifiable')

        return current_timebase - 1, old_time_interval

    def get_voltage(self, CHANNEL='A'):

            timebase, interval = self.get_timebase(self.device, 1_00)

            collection_time = c_int32()

            res = ps2000.ps2000_run_block(
                self.device.handle,
                self.samples,
                timebase,
                self.oversampling,
                byref(collection_time)
            )
            assert_pico2000_ok(res)

            while ps2000.ps2000_ready(self.device.handle) == 0:
                sleep(0.1)

            times = (c_int32 * self.samples)()

            buffer = (c_int16 * self.samples)()

            overflow = c_byte(0)
            if CHANNEL == 'A':
                self.res = ps2000.ps2000_get_times_and_values(
                    self.device.handle,
                    byref(times),
                    byref(buffer),
                    None,
                    None,
                    None,
                    byref(overflow),
                    2,
                    self.samples,
                )
            elif CHANNEL == 'B':
                self.res = ps2000.ps2000_get_times_and_values(
                    self.device.handle,
                    byref(times),
                    None,
                    byref(buffer),
                    None,
                    None,
                    byref(overflow),
                    2,
                    self.samples,
                )

            assert_pico2000_ok(self.res)

            channel_overflow = (overflow.value & 0b0000_0001) != 0

            ps2000.ps2000_stop(self.device.handle)

            channel_mv = adc2mV(buffer, ps2000.PS2000_VOLTAGE_RANGE[self.voltage_range], c_int16(32767))

            if channel_overflow:
                print("OVERSATURATED!! Increase voltage range :(")
                return 
            
            average = np.average(channel_mv)
            print(f"Voltage: {average} mV")
            std_dev = np.std(channel_mv)  # per-acquisition SD
            return average, std_dev
    
    def close_device(self):
         self.device.close()