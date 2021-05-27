import time

import pandas
import scipy.integrate
from scipy.fft import fft, fftfreq
import serial
import numpy as np
import matplotlib.pyplot as plt

CURRENT_READING = b"?C\r\n"
AUTO_TRANSMIT = b"AOUT250\r\n"
TIME_RESOLUTION = 250
BUFFER_SIZE = 1024 * 16


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


def read_auto(ser, record_time=5):
    # Transmission with units
    # ser.write(b"FULL\r\n")
    # ser.flush()
    # Transimission without units
    ser.write(b"NUM\r\n")
    ser.flush()

    # Auto transmit reading
    ser.write(AUTO_TRANSMIT)
    # ser.write(CURRENT_READING)
    ser.flush()

    raw_results = bytearray()
    times = []
    results = []
    t = time.time()

    try:
        # Ignore first reading, it tends to be problematic
        ser.readline()
        while time.time() - t < record_time:
            # current_buffer = ser.read(BUFFER_SIZE)
            # print(f"Read {len(current_buffer)} bytes...")
            # raw_results += current_buffer

            value = ser.readline()
            try:
                result = float(value)
                results.append(result)
                times.append(time.time() - t)
            except ValueError:
                continue
    except KeyboardInterrupt:
        pass
    return times, results


def main():
    with serial.Serial('COM3', 115200, timeout=1) as ser:
        ser.reset_output_buffer()
        ser.reset_input_buffer()

        # Use Newton units
        ser.write(b"N\r\n")
        ser.flush()
        # times, results = read_current(ser)
        times, results = read_auto(ser, 60*5)

        fourier = fft(results)
        freq = fftfreq(len(results), d=1/TIME_RESOLUTION)
        plt.plot(freq, fourier)
        plt.xlim(0)
        plt.title("Fourier Transform of Force")
        plt.xlabel("Frequency [Hz]")
        plt.show()

        results = np.array(results)
        times = np.array(times)
        positive_indices = np.where(results >= 0)
        results = results[positive_indices]
        times = times[positive_indices]

        df = pandas.DataFrame(zip(times, results), columns=["Time", "Results"])
        name = "7_2"
        df.to_csv(f"forces\\{name}.csv")

        plt.plot(times, results)
        plt.xlabel("Time [sec]")
        plt.ylabel("Force [N]")
        plt.title("Force of Hexbug vs. Wall")
        plt.savefig(f"forces\\force_{name}_cm.png")
        plt.show()

        area = scipy.integrate.trapz(results, times)
        average = area / times[-1] - times[0]
        print(f"Area is: {area:.3f} [N*s]")
        print(f"Average force is {average:.3f} [N]")

        plt.hist(results, density=True, bins=100)
        plt.vlines(average, 0, 1, colors="black")
        plt.xlabel("Force [N]")
        plt.title("Forces distribution")
        plt.savefig(f"forces\\dist_{name}_cm.png")
        plt.show()

        print(average, np.mean(results))


if __name__ == "__main__":
    main()


