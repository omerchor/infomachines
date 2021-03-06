import glob
from multiprocessing import Pool

from analyze_stk import *

# %%
BOARD_WIDTH = 29.186
HEX_LENGTH = 4.086
HEX_WIDTH = 1.459


def calc_free_perimeter(length, num_hexbugs):
    """
    Proportions where measured in reference to 25.1 cm board
    """
    # Perimeter
    #return 2 * length + 2 * BOARD_WIDTH - num_hexbugs * HEX_LENGTH

    # Area
    return (length * BOARD_WIDTH) - (num_hexbugs * (HEX_WIDTH * HEX_LENGTH))

# %%
# Execute whole script from here
if __name__ == "__main__":
    # %%
    # Last 4 measurements (39 and 40 cm) seem a little outlying in terms of length to pixels (old measurements, maybe
    # the camera moved)
    # files = glob.glob("stks/*.mat")
    # xtitle = "Board length (pixels)"
    # files = glob.glob("num_hexbugs_stks/*.mat")
    # xtitle = "Number of hexbugs"
    # lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")) for filename
    #            in files]
    # # %%
    # # Work with densities (combine number of hexbugs and board sizes)
    board_sizes_files = glob.glob("stks/*.mat")
    num_hexbugs_files = glob.glob("num_hexbugs_stks/*.mat")
    board_size_densities = [6.0 / calc_free_perimeter(
        float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")), 6)
                            for filename in board_sizes_files]
    num_hexbugs_densities = [
        (n_hexs := float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", ""))) \
                  / calc_free_perimeter(25.1, n_hexs)
        for filename in num_hexbugs_files]
    files = board_sizes_files + num_hexbugs_files
    lengths = board_size_densities + num_hexbugs_densities
    xtitle = "modified_2d_density"
    # xtitle = "Density"
    # %%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
        # Save board sizes in pixels rather than centimeters, it's more natural for the way data is analyzed
        # lengths = [a[1] for a in analysis_results]
        analysis_results = [a[0] for a in analysis_results]
        # Find cell occupied probabilities in each file
        occupied_probabilities = p.map(get_probabilities, analysis_results)
    #     # Analyze frame correlations in each file
    #     # Result is list of [lags, values] ###, peaks, peak_values, peak_widths]
    #     # correlations = p.map(autocorrelations, analysis_results)
    # For debugging: non multithreaded version
    # analysis_results = [parse_file(f) for f in files]
    # occupied_probabilities = [get_probabilities(a) for a in analysis_results]
    # correlations = [autocorrelations(a) for a in analysis_results]
    # %%
    # params, param_errs, _, _ = lab.fit(lab.line, lengths, widths, None, [0,0])
    # lab.plot(lab.line, params, lengths, widths, None, title=f"y={params[0]:.2f}x{params[1]:.2f}",
    #          xlabel="Board length (cm)", ylabel="Board length (pixels)")
    # plt.show()
    # %%
    probabilities = [prob[0] for prob in occupied_probabilities]
    cumulative_counts = [prob[1] for prob in occupied_probabilities]
    infos = [information(prob) for prob in probabilities]
    # Find fit for interpolation of info to length ratio
    params, param_errs, _, _ = lab.fit(lab.line, lengths, infos, None, (0, 0))
    length_to_infos = partial(lab.line, params)
    # lab.plot(lab.line, params, lengths, infos, None, xlabel=xtitle, ylabel="Information",
    #          title=f"y={params[0]:.3e}x{params[1]:.3f}", fmt=".")
    plot_information(lengths, infos, xtitle,
                     groups=["Board Size", "Num Hexbugs"],
                     group_sizes=[len(board_sizes_files), len(num_hexbugs_files)])
    plt.show()
    # %%
    # plot_probabilities(lengths, probabilities, xtitle, show_p0=False,
    #                    is_num_hexbugs=(xtitle == "Number of hexbugs"))
    # for i, c in enumerate(cumulative_counts):
    #     plot_cumulative_probabilities(lengths[i], c)
    # x = plot_probabilities(lengths, probabilities, xtitle, show_p0=False,
    #                        params_guess=(0.1, 0.1, 0.01))
    # # for i, c in enumerate(cumulative_counts):
    # #     plot_cumulative_probabilities(lengths[i], c)
    # %%
    # board_lengths, info_per_step, distance_diffs, diff_errs = simulate_experiment(length_to_infos, lengths,
    #                                                                               probabilities)
    # cumulative_errors = np.cumsum(diff_errs)
    # steps = np.arange(0, len(board_lengths))
    #
    # fig, axs = plt.subplots(2, sharex=True)
    # axs[0].plot(steps, board_lengths, ".")
    # max_vals = np.minimum(board_lengths + cumulative_errors, max(board_lengths))
    # min_vals = np.maximum(board_lengths - cumulative_errors, CELL_WIDTH)
    # axs[0].fill_between(steps, min_vals, max_vals, alpha=0.5)
    # axs[0].set_ylabel("Board length (pixels)")
    # axs[0].set_title("Average board length as function of number of movements")
    #
    # axs[1].plot(steps, np.cumsum(info_per_step), ".", label="Total info")
    # axs[1].set_ylabel("Information")
    # axs[1].set_xlabel("Step number")
    # axs[1].set_title("Total information as function of number of movements")
    # plt.show()
# %%
# Fit the first peak of each autocorrelation graph to a Gaussian
# correlation_fits = [fourier_peak_fit(c[1], False, files[i], 2) for i, c in enumerate(correlations)]
# %%
# correlation_fit_trends(correlation_fits, lengths, xtitle)
# %%
# # Plot blocked probabilities as function of board size
# blocked_probabilities = [probability[0] for probability in occupied_probabilities]
# plt.plot(lengths, blocked_probabilities, ".")
# plt.title("Blocked probability as function of board length")
# plt.xlabel(xtitle)
# plt.ylabel("Occupied probability")
# plt.savefig("blocked_probabilities.png", bbox_inches='tight')
# plt.show()
# %%
# unique_lengths, durations, durations_avg, durations_stdev = first_passages(analysis_results, lengths)
# %%
# Blocked duration histograms
# titles = [f"{length} hexbugs" for length in unique_lengths]
# multi_plot("hist", len(durations), list(durations.values()),
#            main_title="Blocked durations histogram", titles=titles,
#            xtitle=xtitle, density=True)
# %%
#   # Average duration as function of board length
#     plt.errorbar(unique_lengths, durations_avg, durations_stdev, fmt=".")
#     plt.title("Average blocked duration")
#     plt.xlabel(xtitle)
#     plt.ylabel("Average duration (sec)")
#     plt.savefig("average_blocked_duration.png")
#     plt.show()
# %%
# split_correlations(analysis_results[5], lengths[5])
# sub_experiments = np.array_split(analysis_results[-12], 10)
# correlations = [autocorrelations(r) for r in sub_experiments]
# correlation_fits = [fourier_peak_fit(c[1], True, f"{i}", 2,
#                                      smoothing_window_size=31) for i, c in enumerate(correlations)]
# correlation_fit_trends(correlation_fits, range(len(correlation_fits)), "Length (cm)")
