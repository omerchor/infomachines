import collections

import numpy as np
import matplotlib.pyplot as plt
import os

import lab

PIXEL_TO_METER_720P = (59/900)*(10**(-2))
PIXEL_TO_METER_360P = PIXEL_TO_METER_720P*2


def main():
    videos_path = r"C:\Users\OmerChor\Pictures\Camera Roll\wheels"
    times = [np.load(os.path.join(videos_path, f)) for f in os.listdir(videos_path) if f.endswith("times.npy")]
    widths = [np.load(os.path.join(videos_path, f)) * PIXEL_TO_METER_360P for f in os.listdir(videos_path) if f.endswith("widths.npy")]
    pulses = [np.diff(video) for video in widths]

    total_positive = np.sum([np.count_nonzero(np.array(p) > 0) for p in pulses])
    total_negative = np.sum([np.count_nonzero(np.array(p) < 0) for p in pulses])

    total = np.sum([len(p) for p in pulses])
    print(total_positive / total, total_negative / total)

    probabilities_by_width = collections.defaultdict(list)
    for i, video in enumerate(widths):
        current_pulses = pulses[i]
        pulses_by_width = collections.defaultdict(list)
        for j, w in enumerate(video[:-1]):
            pulses_by_width[w].append(current_pulses[j])

        for w, current_pulses in pulses_by_width.items():
            # Find ratio of positive pulse to total number of pulses
            probabilities_by_width[w].append(np.count_nonzero(np.array(current_pulses) > 0) / len(current_pulses))

    # velocities = []
    # velocities_by_width = collections.defaultdict(list)
    # for i in range(len(widths)):
    #     velocities.append(np.diff(widths[i]) / np.diff(times[i]))
    #     # plt.plot(widths[i][:-1], velocities[i], ".")
    #     for j, width in enumerate(widths[i][:-1]):
    #         velocities_by_width[width].append(velocities[i][j])

    # unique_widths = list(velocities_by_width.keys())
    # unique_widths.sort()
    # For each width: number of frames in which there was a movement divided by total number of frames for that width
    ws = list(probabilities_by_width.keys())
    pulse_probability = np.array([np.average(probabilities_by_width[w]) for w in ws])
    pulse_probability_err = np.array([np.std(probabilities_by_width[w]) / np.sqrt(len(probabilities_by_width[w])) for w in ws])
    # indices_to_show = np.where((pulse_probability_err != 0) & (pulse_probability_err < 0.05))[0]
    indices_to_show = np.where((pulse_probability_err != 0))[0]
    print((len(pulse_probability_err) - len(indices_to_show)) / len(pulse_probability_err))
    # indices_to_show = np.where((pulse_probability_err != 0))[0]
    ws = np.array(ws)
    plt.errorbar(ws[indices_to_show], pulse_probability[indices_to_show],
                 pulse_probability_err[indices_to_show], fmt=".")


    # average_velocities = [np.average(velocities_by_width[w]) for w in widths]
    # average_velocities_err = np.array([np.std(velocities_by_width[w]) / np.sqrt(len(velocities_by_width[w])) for w in widths])
    # plt.plot(widths, average_velocities,
    #          color="black", label="average")
    # plt.vlines(widths, average_velocities - average_velocities_err, average_velocities + average_velocities_err, color="red",)
    #            # alpha=0.5, color="red")
    # plt.legend()
    # plt.title("Barrier velocity as function of width")

    plt.xlabel("Width [m]")
    # plt.ylabel("Velocity [m/sec]")
    plt.ylabel("Pulse probability per unit of time")
    plt.show()


if __name__ == "__main__":
    plt.clf()
    main()
