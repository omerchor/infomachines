import os.path
import glob
from multiprocessing import Pool
from analyze_stk import *
#%%
# Execute whole script from here
if __name__ == "__main__":
# %%
    files = glob.glob("stks/*.mat")
    infos = np.zeros(len(files))
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2", "").replace("_1", "")) for filename in files]
#%%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
        # Find cell occupied probabilities in each file
        occupied_probabilities = p.map(get_probabilities, analysis_results)
        # Analyze frame correlations in each file
        # Result is list of [lags, values, peaks, peak_values, peak_widths]
        correlations = p.map(autocorrelations, analysis_results)
        #%%
    # with Pool(8) as p:
    correlation_fits = [fourier_peak_fit(c[1], False, files[i], 3) for i, c in enumerate(correlations)]
#%%
        # Fit the first peak of each autocorrelation graph to a Gaussian
        #correlation_fits = [fourier_peak_fit(v, True, str(lengths[i])) for i, v in enumerate(values)]
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
    plt.xlabel("Board length (cm)")
    plt.ylabel("Gaussian center frequency (Hz)")
    plt.show()

    plt.errorbar(fitted_lengths, widths, widths_errs, fmt=".")
    plt.title("Gaussian width as function of length")
    plt.xlabel("Board length (cm)")
    plt.ylabel("Gaussian widths (Hz)")
    plt.show()
#%%
    # plt.plot(lengths, slopes, ".")
    # plt.title("Autocorrelations decay slope (semilog) as function of board size")
    # plt.xlabel("Board length (cm)")
    # plt.ylabel("Decay semilog slope")
    # plt.show()
# %%
#     blocked = [a[:, 0].astype("float") for a in analysis_results]
#     blocked = [[b - b.mean()] for b in blocked]
#     peaks = [(c[2], c[3]) for c in correlations]
#     # multi_plot("stem", len(peaks), peaks,
#     #            main_title="Correlation peaks", titles=lengths,
#     #            setp_kwargs={"xlim":[0,10000], "ylim":[0,0.2]})
#     multi_plot("acorr", len(blocked), blocked, main_title="Correlations_full",
#                titles=lengths, is_wide=True, num_columns=2,
#                setp_kwargs={"xlim": 0, "ylim": [-0.075, 0.075]},
#                maxlags=None, usevlines=True)

#%%
    # Distance between peaks
    # peaks = [c[2] / FPS for c in correlations]
    # peak_distances = [np.diff(p) for p in peaks]
    # average_peak_distances = [p.mean() for p in peak_distances]
    # peak_distances_stdev = [p.std() for p in peak_distances]
    # plt.errorbar(lengths, average_peak_distances, peak_distances_stdev, fmt=".")
    # plt.title("Average time difference between correlation peaks")
    # plt.xlabel("Board length (cm)")
    # plt.ylabel("Time difference (sec)")
    # plt.show()
# %%
    # Peak width (seconds)
    # widths = [c[4] / FPS for c in correlations]
    # average_peak_width = [w.mean() for w in widths]
    # peak_width_stdev = [p.std() for p in widths]
    # plt.errorbar(lengths, average_peak_width, peak_width_stdev, fmt=".")
    # plt.title("Average correlation peak width")
    # plt.xlabel("Board length (cm)")
    # plt.ylabel("Peak width (sec)")
    # plt.show()
#%%
    # Plot blocked probabilities as function of board size
    # blocked_probabilities = [probability[0] for probability in occupied_probabilities]
    # plt.plot(lengths, blocked_probabilities, ".")
    # plt.title("Blocked probability as function of board length")
    # plt.xlabel("Board length (cm)")
    # plt.ylabel("Occupied probability")
    # plt.savefig("blocked_probabilities.png", bbox_inches='tight')
    # plt.show()
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
    # titles = [f"{length} cm" for length in unique_lengths]
    # multi_plot("hist", len(durations), list(durations.values()),
    #            main_title="Blocked durations histogram", titles=titles,
    #            xtitle="Blocked Duration (sec)", density=True)
#%%
    # Average duration as function of board length
    # plt.errorbar(unique_lengths, durations_avg, durations_stdev, fmt=".")
    # plt.title("Average blocked duration as function of board length")
    # plt.xlabel("Board Length (cm)")
    # plt.ylabel("Average duration (sec)")
    # plt.show()
#%%
    # Find correlations and peaks in correlations
    fig = plt.figure(figsize=(30, 5))
    lags, values, peaks, peak_values, peak_widths = autocorrelations(analysis_results[0])
    # plt.ylim(-0.075, 0.075)
    plt.xlim(0, 150)
    plt.title("Correlations for " + str(lengths[0]) + "cm")
    plt.xlabel("Frame num")
    plt.ylabel("Autocorrelation (logscale)")
    # plt.semilogy()
    plt.show()
#%%
    # first_decay_slope(lags, values, True)
# %%
    fourier_peak_fit(values, True)
