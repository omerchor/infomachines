import time

import serial
import numpy as np
import matplotlib.pyplot as plt

CURRENT_READING = b"?C\r\n"


def main():
    with serial.Serial('COM3', 9600, timeout=1) as ser:
        results = []
        times = []
        while len(results) < 5000:
            if len(results) % 100 == 0:
                print(len(results))
            ser.write(CURRENT_READING)
            x = ser.readline()
            try:
                results.append(float(x.split()[0]))
                times.append(time.time())
            except ValueError:
                continue
    plt.plot(times, results, ".")
    plt.xlabel("Time [sec]")
    plt.ylabel("Force [mN]")
    plt.show()


if __name__ == "__main__":
    main()

