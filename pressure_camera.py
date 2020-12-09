import os
import time
from multiprocessing import Pool

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


def analyze_frame(frame):
    # Crop frame [y, x]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    barrier_color_lower = np.array([140, 150, 150])
    barrier_color_upper = np.array([150, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, barrier_color_lower, barrier_color_upper)

    # Find maximal x value of the barrier
    barrier_position = np.max(mask.nonzero()[X_INDEX])
    arena_width = frame.shape[X_INDEX] - barrier_position

    return arena_width, barrier_position, mask


def analyze_video(path, show=False):
    print("Analyzing", path)
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

            arena_width, barrier_position, mask = analyze_frame(frame)
            # Ignore very large barrier movements - they are probably just a mistake
            if len(widths) > 1 and widths[-1] - arena_width > 20:
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
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return np.arange(0, len(widths)) / fps, widths


def main():
    # analyze_video(os.path.join(r"C:\Users\OmerChor\Pictures\Camera Roll",
    #                            "WIN_20201209_16_09_58_Pro.mp4"),
    #               True)
    # return

    files = [os.path.join(r"C:\Users\OmerChor\Pictures\Camera Roll", f) for f in os.listdir(videos_path) if f.endswith(".mp4")]
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

    # for i, f in enumerate(files):
    #     start_time = time.time()
    #     print(f"{i+1}/{len(files)}: Analyzing {os.path.split(f)[-1]}...",)
    #     times, widths = analyze_video(f)
    #     results.append((times, widths))
    #     if len(times) < min_num_of_frames:
    #         min_num_of_frames = len(times)
    #         min_times = times
    #     print(f"Elapsed {time.time() - start_time:.2f} seconds")

    for times, widths in results:
        plt.plot(times, widths)

    # Compute average
    averages = np.zeros(min_num_of_frames)
    for t in range(min_num_of_frames):
        widths_at_time = [r[1][t] for r in results]
        averages[t] = np.average(widths_at_time)
    plt.plot(min_times, averages, label="Average", color="black")

    plt.xlabel("Time [sec]")
    plt.ylabel("Arena width [pixels]")
    plt.title(f"Free expansion of {len(files)} hexbugs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
