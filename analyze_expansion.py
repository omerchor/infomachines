import collections

import numpy as np
import matplotlib.pyplot as plt
import os

PIXEL_TO_METER_720P = (59/900)*(10**(-2))
PIXEL_TO_METER_360P = PIXEL_TO_METER_720P*2


def main():
    videos_path = r"C:\Users\OmerChor\Pictures\Camera Roll\wheels"
    times = [np.load(os.path.join(videos_path, f)) for f in os.listdir(videos_path) if f.endswith("times.npy")]
    widths = [np.load(os.path.join(videos_path, f)) * PIXEL_TO_METER_360P for f in os.listdir(videos_path) if f.endswith("widths.npy")]
    velocities = []
    velocities_by_width = collections.defaultdict(list)
    for i in range(len(widths)):
        velocities.append(np.diff(widths[i]) / np.diff(times[i]))
        plt.plot(widths[i][:-1], velocities[i], ".")
        for j, width in enumerate(widths[i][:-1]):
            velocities_by_width[width].append(velocities[i][j])

    plt.xlabel("Width [m]")
    plt.ylabel("Velocity [m/sec]")

    widths = list(velocities_by_width.keys())
    widths.sort()
    average_velocities = [np.average(velocities_by_width[w]) for w in widths]
    average_velocities_err = np.array([np.std(velocities_by_width[w]) / np.sqrt(len(velocities_by_width[w])) for w in widths])
    plt.plot(widths, average_velocities,
             color="black", label="average")
    plt.vlines(widths, average_velocities - average_velocities_err, average_velocities + average_velocities_err, color="red",)
               # alpha=0.5, color="red")
    plt.legend()
    plt.title("Barrier velocity as function of width")

    plt.show()


if __name__ == "__main__":
    plt.clf()
    main()
