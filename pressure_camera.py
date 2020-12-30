import collections
import os
import time
from multiprocessing import Pool
import lab
import logging

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.signal
from contextlib import contextmanager

# Channel indices
BLUE = 0
GREEN = 1
RED = 2

Y_INDEX = 0
X_INDEX = 1

# videos_path = r"C:\Users\OmerChor\Pictures\Camera Roll"
VIDEOS_PATH = r"C:\Users\OmerChor\Pictures\Camera Roll\weights"


@np.vectorize
def line(B, x):
    return B[0]*x + B[1]
line.excluded.add(0)


def analyze_frame(frame):
    # Crop frame [y, x]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # barrier_color_lower = np.array([120, 100, 100])
    # barrier_color_upper = np.array([130, 255, 255])
    barrier_color_lower = np.array([120, 100, 100])
    barrier_color_upper = np.array([130, 255, 255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, barrier_color_lower, barrier_color_upper)
    blurred = cv2.blur(mask, (5, 5), 0)

    # Find maximal x value of the barrier
    barrier_position = np.max(blurred.nonzero()[X_INDEX])
    arena_width = frame.shape[X_INDEX] - barrier_position

    return arena_width, barrier_position, blurred


def analyze_video(path, show=False):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    widths = []
    try:
        ret = True
        while ret:
            # Take each frame
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame [y, x]
            # frame = frame[:230, 150:500]
            RESOLUTION_FACTOR = 2
            frame = frame[30:480, 310:980]
            frame = cv2.flip(frame, 1)
            try:
                arena_width, barrier_position, mask = analyze_frame(frame)
            # This happens due to the mask finding no pixels at all. Might happen in a few frames
            except ValueError:
                print(f"--> Empty frame in {os.path.basename(path)}")
                continue
            # Ignore very large barrier movements - they are probably just a mistake
            # if len(widths) > 1 and (widths[-1] - arena_width > 5 or widths[-1] - arena_width < -5):
            if len(widths) > 1 and (np.abs(widths[-1] - arena_width) > 100):
                print(f"{path}: skipped frame")
                if show:
                    cv2.putText(frame, "skipped",
                                (barrier_position - 20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                    cv2.imshow('frame', frame)
                    cv2.imshow('res', res)
                    time.sleep(0.1)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break
                continue
            widths.append(arena_width)

            res = cv2.bitwise_and(frame, frame, mask=mask)
            # Bitwise-AND mask and original image
            cv2.putText(frame, str(arena_width),
                        (barrier_position - 20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            if show:
                cv2.imshow('frame', frame)
                cv2.imshow('res', res)
                time.sleep(0.1)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
    except Exception:
        print(f"Error in {os.path.basename(path)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"{os.path.basename(path)}: {len(widths) / fps:.2f} sec")
    return np.arange(0, len(widths)) / fps, widths


def get_average_expansion(results):
    # Find video with minimum length for averaging
    min_num_of_frames = np.inf
    min_times = []
    for (times, widths) in results:
        if len(times) < min_num_of_frames:
            min_num_of_frames = len(times)
            min_times = times

    # This dict maps arena width to time
    widths_dict = collections.defaultdict(list)
    for times, widths in results:
        plt.plot(times, widths)

        # Find inverse (time as function of width)
        for i, w in enumerate(widths):
            widths_dict[w].append(times[i])

    # The error the time of each width is statistical error of times with same width
    time_errs_by_width = {width: np.std(times) / np.sqrt(len(times) - 1) for width, times in widths_dict.items()
                          if len(times) > 1}

    # Compute average
    averages = np.zeros(min_num_of_frames)
    averages_errs = np.zeros(min_num_of_frames)
    time_errs = np.zeros(min_num_of_frames)
    for t in range(min_num_of_frames):
        widths_at_time = [r[1][t] for r in results]
        averages[t] = np.average(widths_at_time)
        averages_errs[t] = np.std(widths_at_time) / np.sqrt(len(widths_at_time) - 1)
        try:
            rounded_width = round(averages[t])
            time_errs[t] = time_errs_by_width[rounded_width]
        # If the average value was not measured, find error from neighbours
        except KeyError:
            err_before = None
            err_after = None
            difference = 0
            while err_before is None and err_after is None:
                difference += 1
                if rounded_width - difference in time_errs_by_width:
                    err_before = time_errs_by_width[rounded_width - difference]
                if rounded_width + difference in time_errs_by_width:
                    err_after = time_errs_by_width[rounded_width + difference]
            if err_before is not None and err_after is not None:
                time_errs[t] = (err_before + err_after) / 2
            elif err_before is not None:
                time_errs[t] = err_before
            elif err_after is not None:
                time_errs[t] = err_after
    plt.plot(min_times, averages,
             label="Average", color="black")
    plt.fill_between(min_times, averages + averages_errs, averages - averages_errs)
    plt.fill_betweenx(averages, min_times + time_errs, min_times - time_errs)

    plt.xlabel("Time [sec]")
    plt.ylabel("Arena width [pixels]")
    plt.title(f"Free expansion of hexbugs (ensemble size: {len(files)})")
    plt.legend()
    plt.show()

    plt.plot(min_times, averages,
             label="Average", color="black")
    plt.fill_between(min_times, averages + averages_errs, averages - averages_errs)
    plt.fill_betweenx(averages, min_times + time_errs, min_times - time_errs)
    plt.show()

    np.save(os.path.join(VIDEOS_PATH, "averages.npy"), averages)
    np.save(os.path.join(VIDEOS_PATH, "min_times.npy"), min_times)
    np.save(os.path.join(VIDEOS_PATH, "averages_errs.npy"), averages_errs)
    np.save(os.path.join(VIDEOS_PATH, "time_errs.npy"), time_errs)


def get_terminal_velocity(times, widths, position_plot=None,
                          velocity_plot=None, acceleration_plot=None,
                          acceleration_vs_velocity=None,
                          filename=None):
    """
    Linear fit to width as function of time, when it becomes constant

    Returns
    -------

    """
    # smooth positons
    widths = np.array(widths)
    velocity = np.diff(widths) / np.diff(times)
    acceleration = np.diff(velocity) / np.diff(times)[:-1]

    if acceleration_vs_velocity:
        acceleration_vs_velocity.plot(velocity[:-1], acceleration, ".")

    # movement_indices = np.where((velocity[:-1] > 50) &
    #                             (acceleration < 200) & (acceleration > -200))
    # movement_indices = np.arange(np.min(movement_indices), np.max(movement_indices))
    movement_indices = np.arange(np.where(widths > 200)[0][0], np.where(widths > 500)[0][0])
    times = times[movement_indices]
    widths = widths[movement_indices]

    params, param_errs, reduced_chi_squared, p_value = lab.fit(line, times, widths, None, (1, 1))

    velocity = velocity[movement_indices]
    acceleration = acceleration[movement_indices]

    # Find region where acceleration is approximately 0
    if position_plot:
        position_plot.plot(times, widths, ".",
                           label=os.path.basename(filename)[13:-8])
        position_plot.plot(times, line(params, times), "--", color="black", alpha=0.5)
        # position_plot.text(times[0], widths[0], f"v={params[0]:.2f}$\\pm{param_errs[0]:.2f}\\frac{{pixels}}{{s}}$",
        #                    fontsize=10)
    if velocity_plot:
        velocity_plot.plot(times, velocity, ".")
    if acceleration_plot:
        acceleration_plot.plot(times, acceleration, ".")

    return params[0], param_errs[0]


def main():
    weigths_dirs = os.listdir(VIDEOS_PATH)
    files = {}
    for d in weigths_dirs:
        current_dir = os.path.join(VIDEOS_PATH, d)
        files[float(d)] = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith(".mp4")]
    # files = [os.path.join(VIDEOS_PATH, w, f) for f in os.listdir(VIDEOS_PATH) if f.endswith(".mp4")]
    # analyze_video(r"C:\Users\OmerChor\Pictures\Camera Roll\weights\30.45\WIN_20201230_14_56_51_Pro.mp4",
    #                   True)
    # return
    #
    #
    # results = []
    with Pool(8) as p:
        # results = p.map(analyze_video, files)
        results = {}
        terminal_velocities = {}
        weights = []
        terminal_velocities_list = []
        terminal_velocities_errs = []

        t = time.time()
        for weight, weight_files in files.items():
            results[weight] = p.map(analyze_video, weight_files)

        for weight, weight_results in results.items():
            if len(weight_results) == 0:
                continue

            fig, axs = plt.subplots(3, 1, sharex=True)
            # fig2, ax_acceleration_velocity = plt.subplots()

            terminal_velocities[weight] = [get_terminal_velocity(times, widths,
                                                                 axs[0], axs[1], axs[2],
                                                                 #ax_acceleration_velocity,
                                                                 filename=files[weight][i])
                                           for i, (times, widths) in enumerate(weight_results)]
            velocities = np.array([x[0] for x in terminal_velocities[weight]])
            errs = [1/x[1] for x in terminal_velocities[weight]]
            weights.append(weight)
            terminal_velocity = np.average(velocities, weights=errs)
            terminal_velocities_list.append(terminal_velocity)
            terminal_velocities_errs.append(np.sqrt(np.average(velocities**2 - terminal_velocity**2,
                                            weights=errs)) / len(velocities))

            axs[0].set_title("Width")
            axs[1].set_title("Velocity")
            axs[2].set_title("Acceleration")
            axs[2].set_xlabel("Time [sec]")
            axs[0].set_ylabel("Width [pixels]")
            axs[1].set_ylabel(r"Velocity [$\frac{pixels}{sec}$]")
            axs[2].set_ylabel(r"Acceleration [$\frac{pixels}{sec^2}$]")
            # fig.legend()

            fig.suptitle(f"Expansion with {weight:.2f}g weight")
            print("plotting", weight)
            fig.show()

            # ax_acceleration_velocity.set_xlabel("Velocity [pixels / sec]")
            # ax_acceleration_velocity.set_ylabel("Acceleration [pixels / sec$^2$]")
            # fig2.suptitle(f"Acceleration vs. Velocity with {weight:.2f}g weight")
            # fig2.show()

        print(f"Elapsed {time.time() - t:.2f} seconds")

        plt.clf()
        plt.errorbar(weights, terminal_velocities_list, terminal_velocities_errs, fmt=".")
        plt.xlabel("Weight [gram]")
        plt.ylabel("Terminal Velocity [pixel/sec]")
        plt.show()

    # get_average_expansion(results)


if __name__ == "__main__":
    plt.clf()
    main()
