import glob
from multiprocessing import Pool
from analyze_stk import *

# %%
# Execute whole script from here
if __name__ == "__main__":
    # %%
    #     files = glob.glob("stks/*.mat")
    #     xtitle = "Board length (cm)"
    files = glob.glob("num_hexbugs_stks/*.mat")
    xtitle = "Number of hexbugs"
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")) for filename
               in files]
    # # %%
    # # Work with densities (combine number of hexbugs and board sizes)
    # board_sizes_files = glob.glob("stks/*.mat")
    # num_hexbugs_files = glob.glob("num_hexbugs_stks/*.mat")
    # board_size_densities = [6.0 / float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", ""))
    #                         for filename in board_sizes_files]
    # num_hexbugs_densities = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")) / 25.1
    #                          for filename in num_hexbugs_files]
    # files = board_sizes_files + num_hexbugs_files
    # lengths = board_size_densities + num_hexbugs_densities
    # # xtitle = "Hexbug 1D density (1 / cm)"
    # xtitle = "Density"
    # %%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
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
    probabilities = [prob[0] for prob in occupied_probabilities]
    cumulative_counts = [prob[1] for prob in occupied_probabilities]
    infos = [information(prob) for prob in probabilities]
    plot_information(lengths, infos, xtitle)
                     # groups=["Board Size", "Num Hexbugs"],
                     # group_sizes=[len(board_sizes_files), len(num_hexbugs_files)])
    #%%
    plot_probabilities(lengths, probabilities, xtitle, show_p0=False,
                       is_num_hexbugs=(xtitle == "Number of hexbugs"))
    for i, c in enumerate(cumulative_counts):
        plot_cumulative_probabilities(lengths[i], c)
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
    #%%
    # split_correlations(analysis_results[5], lengths[5])
    # sub_experiments = np.array_split(analysis_results[-12], 10)
    # correlations = [autocorrelations(r) for r in sub_experiments]
    # correlation_fits = [fourier_peak_fit(c[1], True, f"{i}", 2,
    #                                      smoothing_window_size=31) for i, c in enumerate(correlations)]
    # correlation_fit_trends(correlation_fits, range(len(correlation_fits)), "Length (cm)")