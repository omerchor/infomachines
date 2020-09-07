import os.path
import glob
import matplotlib as mpl
from multiprocessing import Pool
from scipy import signal
from analyze_stk import *

#%%
if __name__ == "__main__":
    files = glob.glob("stks/*.mat")[24:25]
    infos = np.zeros(len(files))
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2","").replace("_1","")) for filename in files]
#%%
    with Pool(8) as p:
        # Analyze the files
        analysis_results = p.map(parse_file, files)
        occupied_probabilities = p.map(get_probabilities, analysis_results)

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
    # titles = [f"{length} cm" for length in unique_lengths]
    # multi_plot("hist", len(durations), list(durations.values()),
    #            main_title="Blocked durations histogram", titles=titles,
    #            xtitle="Blocked Duration (sec)", density=True)
#%%
    # plt.errorbar(unique_lengths, durations_avg, durations_stdev, fmt=".")
    # plt.title("Average blocked duration as function of board length")
    # plt.xlabel("Board Length (cm)")
    # plt.ylabel("Average duration (sec)")
    # plt.show()
#%%
    blocked = analysis_results[0][:, 0].astype("float")
    blocked -= blocked.mean()
    fig = plt.figure(figsize=(20, 5))
    corrs = plt.acorr(blocked, maxlags=None, usevlines=True)
    envelope = signal.hilbert(corrs[1])
    # plt.xlim(10000,20000)
    plt.xlim(0)
    plt.ylim(-0.075, 0.075)
    plt.plot(corrs[0], envelope)
    plt.show()
# %%
    xf = np.fft.fftfreq(len(corrs[1]))
    fourier = np.fft.fft(corrs[1])
    plt.plot(xf, fourier)
    plt.xlim(-0.05, 0.05)
    plt.show()
