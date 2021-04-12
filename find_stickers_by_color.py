import itertools

import cv2
import time
import numpy as np
import pandas
import trackpy
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
from analyze_stk import correlation_peaks


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
    # cap = cv2.VideoCapture(r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\1.mp4")
    cap = cv2.VideoCapture(r"C:\Users\OmerChor\Pictures\Camera Roll\vid_1080p.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
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
                    # Cut edges (outside arena)
                    frame = frame[100:900,300:1500]
                    frame_num += 1
                    pbar.update(1)
                    centers = detect_stickers(frame,
                                              hsv_min=(90, 150, 150),
                                              hsv_max=(120, 255, 255)
                                              )
                    if len(centers) == 0:
                        # print(f"Nothing found in frame #{frame_num}")
                        # cv2.imshow("output", frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     ret = False
                        # time.sleep(0.01)
                        continue
                    frames.append(centers_dataframe(centers, frame_num))
                    # show_centers(frame, centers)#, 0.01)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        ret = False
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
        sub['velocity'] = np.hypot(dvx, dvy)

        # Velocity exactly 0 probably due to lag in video (?)
        sub = sub[sub['velocity'] != 0]
        if len(sub) == 0:
            continue

        # # Where frames with velocity=0 were removed (due to lag in video), cut velocity by half because the distance
        # # was travelled during two frames, not one
        # sub.set_index("frame", drop=False, inplace=True)
        # prev = sub.index.values[0]
        # for j in sub.index.values[1:]:
        #     if j - prev > 1:
        #         sub.at[j, "velocity"] /= (j - prev)
        #     prev = j

        subs.append(sub)

    data = pandas.concat(subs)
    data.to_pickle("trajectories_with_velocity")
    return data


def plot_velocity_hist(trajectories=None):
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    h = plt.hist(trajectories.velocity, bins=200)
    plt.xlabel("Velocity (pixels / frame)")
    plt.ylabel("Count")
    plt.title("Velocities distribution")
    plt.ylim(0, 4000)
    plt.xlim(0, 40)
    plt.show()

    return h


def average_velocity(trajectories=None):
    """
    Return a matrix containing the average velocity at each position

    """
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    heatmap = trajectories.pivot_table("velocity", "y", "x", aggfunc="mean").to_numpy()

    # Interpolate missing values
    x = np.arange(0, heatmap.shape[1])
    y = np.arange(0, heatmap.shape[0])
    heatmap = np.ma.masked_invalid(heatmap)

    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~heatmap.mask]
    y1 = yy[~heatmap.mask]
    newarr = heatmap[~heatmap.mask]

    return griddata((x1, y1), newarr.ravel(), (xx, yy))


    # xmin = trajectories.x.min()
    # ymin = trajectories.y.min()
    # for row in trajectories.iterrows():
    #     velocity_sum[row['y'] - ymin][row['x'] - xmin] += row['velocity']
    #     velocity_sum[row['y'] - ymin][row['x'] - xmin] += 1
    #
    # return velocity_sum / velocity_counts

    # for x in set(trajectories.x):
    #     for y in set(trajectories.y):
    #         values = trajectories[(trajectories.x == x) & (trajectories.y == y)]
    #         if len(values) > 0:
    #             heatmap[y - trajectories.y.min()][x - trajectories.x.min()] = np.average(values.velocity)
    #         else:
    #             heatmap[y - trajectories.y.min()][x - trajectories.x.min()] = np.nan

    # return heatmap


def plot_velocity_heatmap(trajectories=None):
    velocities = average_velocity(trajectories)
    plt.matshow(velocities, vmin=0, vmax=35)
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label("velocity [px/s]")
    plt.title("Mean velocity as function of position\n(interpolated)")
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()


def plot_velocity_direction(trajectories=None):
    if trajectories is None:
        trajectories = pandas.read_pickle("trajectories_with_velocity")

    # Remove zero velocity points
    filtered_trajectories = trajectories[(trajectories.dx != 0) | (trajectories.dy != 0)]

    angles = np.arctan2(filtered_trajectories.dy, filtered_trajectories.dx)

    ax = plt.subplot(111, polar=True)
    ax.hist(angles, bins=90)
    plt.title("Distribution of velocity direction (with respect to x axis)")
    plt.show()


def velocity_autocorrelation(trajectories):
    velocity_lags = []
    velocity_autocorrelations = []
    angle_lags = []
    angle_autocorrelations = []
    # Each "particle" is a new trajectory (i.e. bug capsizes and reappears as a new object)
    for particle in set(trajectories["particle"]):
        current = trajectories[trajectories.particle == particle]
        current_velocities = current.velocity
        if len(current_velocities) > 100:
            lags, autocorrs, _, _ = plt.acorr(current_velocities, maxlags=None, usevlines=False, alpha=0.2)
            velocity_lags.append(lags)
            velocity_autocorrelations.append(autocorrs)

            angles = np.arctan2(current.dy, current.dx)
            lags, autocorrs, _, _ = plt.acorr(angles, maxlags=None, usevlines=False, alpha=0.2)
            angle_autocorrelations.append(autocorrs)
            plt.clf()

    # Plot velocities
    cmap = plt.cm.get_cmap("hsv", len(set(trajectories["particle"])) + 1)
    for i, particle in enumerate(set(trajectories["particle"])):
        current_frames = trajectories[trajectories.particle == particle]
        plt.plot(current_frames.frame, current_frames.velocity, ".", markersize=1, color=cmap(i))
    plt.ylim(top=35)
    plt.xlim(0)
    plt.title("Velocity as function of time")
    plt.ylabel("Velocity [px/s]")
    plt.xlabel("Time [frame]")
    plt.show()


    # Plot velocity autocorrelations
    for i in range(len(velocity_lags)):
        plt.plot(velocity_lags[i], velocity_autocorrelations[i], linewidth=0.5, color=cmap(i))

    plt.xlim(0, 200)
    plt.minorticks_on()
    plt.title("Autocorrelation of velocities in different executions")
    plt.xlabel("Lag [frames]")
    plt.ylabel("Autocorrelation")
    plt.show()

    # Plot angle autocorrelations
    for i in range(len(velocity_lags)):
        plt.plot(velocity_lags[i], angle_autocorrelations[i], linewidth=0.5, color=cmap(i))

    normed_length = list(itertools.zip_longest(*angle_autocorrelations, fillvalue=np.nan))
    # mean_autocorrelation = [np.nanmean(i) for i in normed_length]
    # plt.plot(np.linspace(0, 400, 400), mean_autocorrelation[:400], color="black", label="Average")
    # plt.legend()

    plt.xlim(0, 400)
    plt.minorticks_on()
    plt.title("Autocorrelation of velocity direction in different executions")
    plt.xlabel("Lag [frames]")
    plt.ylabel("Autocorrelation")
    plt.show()


def velocity_by_region(trajectories):
    xmin = trajectories.x.min()
    ymin = trajectories.y.min()
    # margins_rule = (trajectories.x - xmin < 30) | (trajectories.x - xmin > 220) | (trajectories.y - ymin < 40) | (trajectories.y - ymin > 270)
    margins_rule = (trajectories.x - xmin < 40) | (trajectories.x - xmin > 240) | (trajectories.y - ymin < 50) | (trajectories.y - ymin > 260)
    velocities_out = trajectories[margins_rule].velocity
    velocities_in = trajectories[~margins_rule].velocity

    fig, ax = plt.subplots(2, sharex=True)

    n, bins, patches = ax[0].hist(velocities_out, bins=200)
    ax[0].set_ylabel("Count")
    ax[0].set_title("Velocities near edges")
    ax[0].set_xlim(0, 40)

    df2 = pandas.DataFrame({'count': n, 'bins': bins[:-1]})
    df2.to_excel("edges_velocities.xlsx", sheet_name='sheet1', index=False)


    ax[1].hist(velocities_in, bins=200)
    ax[1].set_xlabel("Velocity (pixels / frame)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Velocities inside")
    ax[1].set_xlim(0, 40)

    df2 = pandas.DataFrame({'count': n, 'bins': bins[:-1]})
    df2.to_excel("inside_velocities.xlsx", sheet_name='sheet1', index=False)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trackpy.enable_numba()
    # Analyze
    main()

    # Load previous analysis
    # tr = load()
    # trackpy.plot_traj(tr, label=True)

    # Load trajectories directly and analyze
    tr = analyze_velocity()
    plot_velocity_hist(tr)
    # plot_velocity_heatmap(tr)
    # plot_velocity_direction(tr)
    velocity_autocorrelation(tr)
    velocity_by_region(tr)


