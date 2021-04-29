import time

import serial
import numpy as np
import matplotlib.pyplot as plt

CURRENT_READING = b"?C\r\n"
AUTO_TRANSMIT = b"AOUT250\r\n"
TIME_RESOLUTION = 250
BUFFER_SIZE = 1024


def read_current(ser):
    """
    Reads and returns current reading from serial object until interrupted
    Parameters
    ----------
    ser: serial object

    Returns
    -------
    list of times of results, list of readings
    """
    results = []
    times = []
    t0 = time.time()

    # Test
    # ser.write(bytes(AUTO_TRANSMIT_FORMAT.format(250)))
    try:
        while True:
            ser.write(CURRENT_READING)
            ser.flush()
            x = ser.readline()
            try:
                results.append(float(x.split()[0]))
                times.append(time.time() - t0)
            except ValueError:
                continue
    except KeyboardInterrupt:
        return times, results


def read_auto(ser):
    # Transmission with units
    # ser.write(b"FULL\r\n")
    # ser.flush()
    # Transimission without units
    ser.write(b"NUM\r\n")
    ser.flush()

    # Auto transmit reading
    ser.write(AUTO_TRANSMIT)
    ser.flush()
    # Wait for some data to accumulate
    time.sleep(1)
    raw_results = bytearray()
    try:
        while True:
            current_buffer = ser.read(BUFFER_SIZE)
            print(f"Read {len(current_buffer)} bytes...")
            raw_results += current_buffer
    except KeyboardInterrupt:
        pass

    results = []
    for value in raw_results.split():
        try:
            results.append(float(value))
        except ValueError:
            continue
    times = np.arange(0, len(results) / TIME_RESOLUTION, 1/TIME_RESOLUTION)
    return times, results


def main():
    with serial.Serial('COM3', 115200, timeout=1) as ser:
        ser.reset_output_buffer()
        ser.reset_input_buffer()

        # Use Newton units
        ser.write(b"N\r\n")
        ser.flush()
        # times, results = read_current(ser)
        times, results = read_auto(ser)
        plt.plot(times, results)
        plt.xlabel("Time [sec]")
        plt.ylabel("Force [N]")
        plt.xlim(1,2)
        plt.show()


if __name__ == "__main__":
    main()


