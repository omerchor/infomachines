import os.path
import glob
from multiprocessing import Pool
from analyze_stk import *
#%%
# Execute whole script from here
if __name__ == "__main__":
# %%
    files = glob.glob("stks/*.mat")[24:25]
    infos = np.zeros(len(files))
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2","").replace("_1","")) for filename in files]
#%%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
        # Find cell occupied probabilities in each file
        occupied_probabilities = p.map(get_probabilities, analysis_results)
        # Analyze frame correlations in each file
        # Result is list of [lags, values, peaks, peak_widths]
        correlations = p.map(autocorrelations, analysis_results)
#%%
    peaks = [c[2] for c in correlations]
    peak_distances = [np.diff(p) for p in peaks]
    # average_distance =
#%%
    # Plot blocked probabilities as function of board size
    blocked_probabilities = [probability[0] for probability in occupied_probabilities]
    plt.plot(lengths, blocked_probabilities, ".")
    plt.title("Blocked probability as function of board length")
    plt.xlabel("Board length (cm)")
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
    fig = plt.figure(figsize=(20, 5))
    lags, values, peaks, peak_widths = autocorrelations(analysis_results)
    plt.ylim(-0.075, 0.075)
    # plt.xlim(0)
    plt.xlim(22000, 23000)
    plt.show()
# %%
    # Fourier transform of correlations graph
    xf = np.fft.fftfreq(len(values))
    fourier = np.fft.fft(values)
    plt.plot(xf, fourier)
    plt.xlim(-0.05, 0.05)
    plt.show()
# %%
