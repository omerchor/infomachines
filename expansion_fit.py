import os

import lab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


BARRIER_MASS = 58.12 * 10**-3
ARENA_WIDTH = 29.5 * 10**-2
GAMMA = 0.524


def as_si(x, ndp=2):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


PIXEL_TO_METER_720P = (59/900)*(10**(-2))
PIXEL_TO_METER_360P = PIXEL_TO_METER_720P*2


def power_law(B, t):
    return B[0] + B[2] * (t**B[1])


def inverse_power_law(B, x):
    """
    Returns time as function of position
    """
    return ((x - B[0])/B[2])**(1/B[1])


@np.vectorize
def velocity(B, t):
    return B[2] * B[1] * t**(B[1]-1)
velocity.excluded.add(0)


@np.vectorize
def acceleration(B, t):
    return B[2] * B[1] * (B[1] - 1) * t**(B[1] - 2)
acceleration.excluded.add(0)


@np.vectorize
def pressure(B, x):
    t = inverse_power_law(B, x)
    return (BARRIER_MASS * acceleration(B, x) +  GAMMA * velocity(B, t)) / ARENA_WIDTH
pressure.excluded.add(0)


@np.vectorize
def work(B, x):
    """
    Work that has been applied up to x
    """
    return (GAMMA * B[2] * B[1]**2 / (2*B[1] - 1)) * \
           ((x - B[0]) / B[2]) ** (2-1/B[1])
work.excluded.add(0)


files_path = r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project"


def main():
    averages = np.load(os.path.join(files_path, "averages.npy"))
    min_times = np.load(os.path.join(files_path, "min_times.npy"))
    averages_errs = np.load(os.path.join(files_path, "averages_errs.npy"))
    time_errs = np.load(os.path.join(files_path, "time_errs.npy"))

    averages *= PIXEL_TO_METER_360P
    averages_errs *= PIXEL_TO_METER_360P

    # accelerations = np.diff(np.diff(averages))
    # average = np.average(accelerations)
    # plt.plot(min_times[:-2], accelerations)
    # plt.hlines(average, min_times[0], min_times[-1], label=f"Average={average:.2f}")
    # plt.legend()
    # plt.xlabel("Time [sec]")
    # plt.ylabel("Acceleration [$m/{s}^2$]")
    # plt.show()

    # diffs = np.diff(averages)
    # min_index = 0# np.min(np.where(diffs > 5 * PIXEL_TO_METER_720P))
    # max_index = len(averages) - 1  # np.max(np.where(diffs > 5 * PIXEL_TO_METER_720P))
    # print(min_index, max_index)
    # print(diffs)
    # indices = np.arange(min_index, max_index)
    # print(diffs[indices])
    # averages = averages[indices]
    # min_times = min_times[indices]
    # averages_errs = averages_errs[indices]
    # time_errs = time_errs[indices]

    # plt.plot(min_times, averages)
    # plt.show()
    # return


    params, param_errs, reduced_chi_squared, p_value = lab.fit(power_law, min_times, averages,
                                                               averages_errs, x_errs=time_errs,
                                                               params_guess=(1, 1, 1))
    # mass = float(os.path.basename(files_path))
    lab.plot(power_law, params, min_times, averages, averages_errs, time_errs,
             xlabel="time [sec]", ylabel="arena width [m]",
             title=f"$width=({as_si(params[0])}) + ({as_si(params[2])})t^{{{params[1]:.2f}}}$"
                   f"\t($\chi^2_{{red}}={reduced_chi_squared:.2f}$)\n"
                   f"$acceleration={as_si(params[2]*params[1]*(params[1]-1))}t^{{{params[1]-2:.2f}}}$")
    plt.show()

    plt.plot(averages, pressure(params, averages))
    plt.title("Pressure ($\\frac{ma+\\gamma v}{A}$) as function of arena width")
    plt.xlabel("Arena width [m]")
    plt.ylabel("1D-Pressure [N/m]")
    plt.show()

    print(averages)
    plt.plot(averages, work(params, averages))
    plt.title("Total work as function of arena width")
    plt.xlabel("Arena width [m]")
    plt.ylabel("Work (J)")
    plt.show()


if __name__ == "__main__":
    main()

