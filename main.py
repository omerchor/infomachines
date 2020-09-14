import os.path
import glob
from multiprocessing import Pool
from analyze_stk import *
#%%
# Execute whole script from here
if __name__ == "__main__":
# %%
    files = glob.glob("stks/*.mat")
    xtitle = "Board length (cm)"
    # files = glob.glob("num_hexbugs_stks/*.mat")
    # xtitle = "Number of hexbugs"
    infos = np.zeros(len(files))
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")) for filename in files]
#%%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
        # Find cell occupied probabilities in each file
        occupied_probabilities = p.map(get_probabilities, analysis_results)
        # Analyze frame correlations in each file
        # Result is list of [lags, values] ###, peaks, peak_values, peak_widths]
        correlations = p.map(autocorrelations, analysis_results)
    # For debugging: non multithreaded version
    # analysis_results = [parse_file(f) for f in files]
    # occupied_probabilities = [get_probabilities(a) for a in analysis_results]
    # correlations = [autocorrelations(a) for a in analysis_results]
#%%
    # Fit the first peak of each autocorrelation graph to a Gaussian

    correlation_fits = [fourier_peak_fit(c[1], False, files[i], 2) for i, c in enumerate(correlations)]
#%%
    fitted_lengths = []
    centers = []
    centers_errs = []
    widths = []
    widths_errs = []
    for i, f in enumerate(correlation_fits):
        params, param_errs = f
        if params is None:
            continue
        fitted_lengths.append(lengths[i])
        centers.append(params[1])
        centers_errs.append(param_errs[1])
        widths.append(params[2])
        widths_errs.append(param_errs[2])
    plt.errorbar(fitted_lengths, centers, centers_errs, fmt=".")
    plt.title("Gaussian center as function of length")
    plt.xlabel(xtitle)
    plt.ylabel("Gaussian center frequency (Hz)")
    plt.savefig("First_peak_center.png")
    plt.show()

    plt.errorbar(fitted_lengths, widths, widths_errs, fmt=".")
    plt.title("Gaussian width as function of length")
    plt.xlabel(xtitle)
    plt.ylabel("Gaussian widths (Hz)")
    plt.savefig("peak_width.png")
    plt.show()
#%%
#     # plt.plot(lengths, slopes, ".")
#     # plt.title("Autocorrelations first decay slope (semilog) as function of board size")
#     # plt.xlabel("Board length (cm)")
#     # plt.ylabel("Decay semilog slope")
#     # plt.show()
#%%
    # Plot blocked probabilities as function of board size
    blocked_probabilities = [probability[0] for probability in occupied_probabilities]
    plt.plot(lengths, blocked_probabilities, ".")
    plt.title("Blocked probability as function of board length")
    plt.xlabel(xtitle)
    plt.ylabel("Occupied probability")
    plt.savefig("blocked_probabilities.png", bbox_inches='tight')
    plt.show()
#%%
    # Merge results that belong to the same board size (multiple files)
    unique_lengths = np.unique(lengths)
    durations = {}
    durations_avg = np.zeros(len(unique_lengths))
    durations_stdev = np.zeros(len(unique_lengths))

    for i, length in enumerate(unique_lengths):
        # Find indices of length in original list
        indices = np.where(lengths == length)[0]
        results = []
        for index in indices:
            experiment_durations, _ = analyze_first_passage(analysis_results[index])
            results.extend(experiment_durations)

        # Put it in a list for unpacking by multi_plot
        durations[length] = [np.array(results)]
        durations_avg[i] = np.average(results)
        durations_stdev[i] = np.std(results)
#%%
    # Blocked duration histograms
    titles = [f"{length} hexbugs" for length in unique_lengths]
    multi_plot("hist", len(durations), list(durations.values()),
               main_title="Blocked durations histogram", titles=titles,
               xtitle=xtitle, density=True)
# %%
#   # Average duration as function of board length
    plt.errorbar(unique_lengths, durations_avg, durations_stdev, fmt=".")
    plt.title("Average blocked duration")
    plt.xlabel(xtitle)
    plt.ylabel("Average duration (sec)")
    plt.savefig("average_blocked_duration.png")
    plt.show()
