import lab
import numpy as np
import matplotlib.pyplot as plt


def power_law(B, x):
    return B[0] + B[2] * (x**B[1])


def main():
    averages = np.load("averages.npy")
    min_times = np.load("min_times.npy")
    averages_errs = np.load("averages_errs.npy")
    time_errs = np.load("time_errs.npy")

    params, param_errs, reduced_chi_squared, p_value = lab.fit(power_law, min_times, averages,
                                                               averages_errs, x_errs=time_errs,
                                                               params_guess=(1, 0.5, 1))
    lab.plot(power_law, params, min_times, averages, averages_errs, time_errs,
             xlabel="time [sec]", ylabel="arena width [pixels]",
             title=f"$width={params[0]:.2f} + {params[2]:.2f}t^{{{params[1]:.2f}}}$"
                   f"\t($\chi^2_{{red}}={reduced_chi_squared:.2f}$)\n"
                   f"$acceleration={params[2]*params[1]*(params[1]-1):.2f}t^{{{params[1]-2:.2f}}}$")
    plt.show()


if __name__ == "__main__":
    main()

