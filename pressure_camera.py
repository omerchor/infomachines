import collections
import os
import time
from multiprocessing import Pool
import logging

import numpy as np
import cv2
from matplotlib import pyplot as plt
from contextlib import contextmanager


# Channel indices
BLUE = 0
GREEN = 1
RED = 2

Y_INDEX = 0
X_INDEX = 1

videos_path = r"C:\Users\OmerChor\Pictures\Camera Roll"


def power_law(B, x):
    return B[0] + B[1]*x


def analyze_frame(frame):
    # Crop frame [y, x]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # barrier_color_lower = np.array([140, 150, 150])
    # barrier_color_upper = np.array([150, 255, 255])
    barrier_color_lower = np.array([120, 180, 120])
    barrier_color_upper = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, barrier_color_lower, barrier_color_upper)

    # Find maximal x value of the barrier
    barrier_position = np.max(mask.nonzero()[X_INDEX])
    arena_width = frame.shape[X_INDEX] - barrier_position

    return arena_width, barrier_position, mask


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
            frame = frame[:230, 150:500]
            try:
                arena_width, barrier_position, mask = analyze_frame(frame)
            # This happens due to the mask finding no pixels at all. Might happen in a few frames
            except ValueError:
                print(f"--> Empty frame in {os.path.basename(path)}")
                continue
            # Ignore very large barrier movements - they are probably just a mistake
            if len(widths) > 1 and (widths[-1] - arena_width > 5 or widths[-1] - arena_width < -5):
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


def main():
    files = [os.path.join(r"C:\Users\OmerChor\Pictures\Camera Roll", f) for f in os.listdir(videos_path) if f.endswith(".mp4")]
    # analyze_video(r"C:\Users\OmerChor\Pictures\Camera Roll\WIN_20201216_12_09_41_Pro.mp4",
    #               True)
    # return
    #
    #
    # results = []
    with Pool(8) as p:
        t = time.time()
        results = p.map(analyze_video, files)
        print(f"Elapsed {time.time() - t:.2f} seconds")

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
                    err_after = time_errs_by_width[rounded_width - difference]
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

    # diffs = [np.diff(w) for t, w in results]
    # plt.hist(diffs, bins=np.arange(-2.5, 7.5))
    # plt.xlabel("Step size")
    # plt.ylim(0, 500)
    # plt.show()


if __name__ == "__main__":
    plt.clf()
    main()
