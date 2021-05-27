import numpy as np
import pandas
import matplotlib.pyplot as plt


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

        # Where frames with velocity=0 were removed (due to lag in video), cut velocity by half because the distance
        # was travelled during two frames, not one
        sub.set_index("frame", drop=False, inplace=True)
        prev = sub.index.values[0]
        for j in sub.index.values[1:]:
            if j - prev > 1:
                sub.at[j, "velocity"] /= (j - prev)
            prev = j

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


def velocity_by_region(trajectories):
    xmin = trajectories.x.min()
    ymin = trajectories.y.min()
    margins_rule = (trajectories.x - xmin < 40) | (trajectories.x - xmin > 240) | (trajectories.y - ymin < 50) | (trajectories.y - ymin > 260)
    velocities_out = trajectories[margins_rule].velocity
    velocities_in = trajectories[~margins_rule].velocity

    fig, ax = plt.subplots(2, sharex=True)

    n, bins, patches = ax[0].hist(velocities_out, bins=200)
    ax[0].set_ylabel("Count")
    ax[0].set_title("Velocities near edges")
    ax[0].set_xlim(0, 40)

    ax[1].hist(velocities_in, bins=200)
    ax[1].set_xlabel("Velocity (pixels / frame)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Velocities inside")
    ax[1].set_xlim(0, 40)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tr = analyze_velocity()
    plot_velocity_hist(tr)
    velocity_by_region(tr)


