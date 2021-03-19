<<<<<<< HEAD
import pprint

=======
>>>>>>> origin/master
import cv2
import time
import numpy as np
import pandas
import trackpy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn


def detect_stickers(frame, hsv_min=(40, 50, 0), hsv_max=(85, 255, 255)):
    """
    Detects regions within HSV colors in range of hsv_min and hsv_max.
    Default values are for Green stickers. Try changing only hue (first index)
    for different colors.

    Parameters
    ----------
    frame image to look for objects in
    hsv_min minimal values of h, s, and v in a tuple (h,s,v).
    hsv_max maximal values of h, s, and v in a tuple (h,s,v).

    Returns
    -------
    List of object centers
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    foreground = np.zeros_like(frame, np.uint8)
    imask = mask > 0
    foreground[imask] = frame[imask]

    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 113, 255, 0)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = [cv2.moments(cnt) for cnt in contours]
    centers = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) for m in moments]
    return centers


def centers_dataframe(centers, frame_number):
    """
    Create a pandas dataframe from centers locations

    Parameters
    ----------
    centers list of object center locations
    frame_number frame number of data in centers list

    Returns
    -------
    A dataframe with centers organized as expected for trackpy
    """
    df = pandas.DataFrame.from_records(centers, columns=["x", "y"])
    df["frame"] = frame_number
    return df


def show_centers(frame, centers, sleep_time=0.0):
    """
    Show the frame with detected centers drawn on top
    Parameters
    ----------
    frame
    centers
    sleep_time

    Returns
    -------

    """
    for cnt in centers:
        cv2.circle(frame, cnt, 10, (255, 255, 255), cv2.FILLED)
    cv2.imshow("output", frame)
    if sleep_time > 0:
        time.sleep(sleep_time)


def main():
    cap = cv2.VideoCapture(r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\1.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret = True
    frames = []
    frame_num = 0
    try:
        # while ret and frame_num < 10000:
        with tqdm(total=frame_count) as pbar:
            while ret:
                # ret, frame = cap.read(1)
                ret, frame = cap.read()
                if ret:
                    frame_num += 1
                    pbar.update(1)
                    centers = detect_stickers(frame,
                                              hsv_min=(30, 35, 0),
                                              hsv_max=(90, 255, 255)
                                              )
                    if len(centers) == 0:
                        # print(f"Nothing found in frame #{frame_num}")
                        # cv2.imshow("output", frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     ret = False
                        # time.sleep(1)
                        continue
                    frames.append(centers_dataframe(centers, frame_num))
                    # show_centers(frame, centers, 0.01)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     ret = False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    np.save("frames", np.asarray(frames))

    # predictor = trackpy.predict.NearestVelocityPredict()
    tr = pandas.concat(trackpy.link_df_iter(frames, 20, memory=0,
                                            pos_columns=['y', 'x'],
                                            t_column='frame'))
    trackpy.plot_traj(tr, label=True, superimpose=frame)


def load():
    frames = list(np.load("frames.npy", allow_pickle=True))
    predictor = trackpy.predict.NearestVelocityPredict()
    tr = pandas.concat(predictor.link_df_iter(frames, 100, memory=0,
                                              pos_columns=['y', 'x'],
                                              t_column='frame'))
    tr.to_pickle("trajectories")
    return tr


def analyze_velocity():
    """
    Loads trajectories files and returns a new dataframe with velocities at each position
    """
    tr = pandas.read_pickle("trajectories")
    subs = []
    particles = set(tr.particle)
    for i, item in enumerate(particles):
        sub = tr[tr.particle == item]
        dvx = np.diff(sub.x)
        dvy = np.diff(sub.y)

        sub = sub[:-1]
        sub['dx'] = dvx
        sub['dy'] = dvy
        # sub['velocity'] = np.sqrt(dvx**2 + dvy**2)
        sub['velocity'] = np.hypot(dvx, dvy)

        subs.append(sub)

    data = pandas.concat(subs)
    data.to_pickle("trajectories_with_velocity")
    return data


def plot_velocity_hist(trajectories=None):
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    plt.hist(trajectories.velocity, bins=200)
    plt.xlabel("Velocity (pixels / frame)")
    plt.ylabel("Count")
    plt.title("Velocities distribution")
    plt.ylim(0, 4000)
    plt.xlim(0, 40)
    plt.show()


@np.vectorize
def get_average_velocity(x, y):
    return np.average(trajectories[(trajectories.x == x) & (trajectories.y == y)].velocity)


def plot_velocity_heatmap(trajectories=None):
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    heatmap = np.zeros((trajectories.y.max() - trajectories.y.min() + 1,
                        trajectories.x.max() - trajectories.x.min() + 1,), dtype=float)

    for i in set(trajectories.x):
        for j in set(trajectories.y):
            heatmap[j - trajectories.y.min()][i - trajectories.x.min()] = np.average(trajectories[(trajectories.x == i)
                                                                        & (trajectories.y == j)].velocity)

    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()


def plot_velocity_direction(trajectories=None):
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    # Remove zero velocity points
    filtered_trajectories = trajectories[(trajectories.dx != 0) | (trajectories.dy != 0)]

    angles = np.arctan2(filtered_trajectories.dy, filtered_trajectories.dx)

    ax = plt.subplot(111, polar=True)
    bars = ax.hist(angles, bins=90)
    plt.title("Distribution of velocity direction (with respect to x axis)")
    plt.show()

    # heatmap = trajectories.pivot_table("velocity", "y", "x", aggfunc="mean", fill_value=-1)
    # seaborn.heatmap(heatmap, robust=True)
    # plt.title("Mean velocity as function of position")
    # plt.show()

    # Use cartesian to polar
    # heatmap by regions
    # Average autocorrelations (on all trajectories)


if __name__ == "__main__":
    trackpy.enable_numba()
    # Analyze
    # main()

    # Load previous analysis
    # tr = load()
    # trackpy.plot_traj(tr, label=True)

    # Load trajectories directly and analyze
    tr = analyze_velocity()
    plot_velocity_hist(tr)
    # plot_velocity_heamap(tr)
