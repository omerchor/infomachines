#!/usr/bin/env python
# coding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal
import lab
import collections
import pandas

# %%
CELL_WIDTH = 50     # Virtual cell width in pixels (should be barrier blocked size)
FPS = 24            # Video frames per second


def parse_stk(filename):
    """
    Loads and parses a .mat stk file
    Returns the file's data as a pandas dataframe
    """
    return sio.loadmat(filename)["stk"]


def group_by_frame(stk):
    """
    Returns dict containing list of entries for each frame in an stk file,
    as well as minimum and maximum x values
    """
    #   stk file format:
    #    x y Br Ec Frame
    groups = {}
    current = []
    current_frame = 1
    
    min_x = 0
    max_x = 0
    
    for row in stk:
        new_frame = int(row[4])
        # New frame. Save previous and update temporary variables
        if new_frame != current_frame:
            # Save x positions as a numpy array for faster manipulation later
            groups[current_frame] = np.array(current)
            current = []
            current_frame = new_frame
            
        new_x = row[0]
        min_x = min(min_x, new_x)
        max_x = max(max_x, new_x)
        # Add the x field to the current list
        current.append(new_x)
    return groups, min_x, max_x, new_frame


def analyze_frame(data, lower_x_boundary):
    """
    Return indices of occupied cells in frames
    """
    return np.unique((data - lower_x_boundary) // CELL_WIDTH).astype('int')


def analyze_positions(frames, boundaries, num_frames):
    """
    Extracts position distribution from frames dict
    Returns a list of cell occupations (boolean - is occupied or not) for each frame
    """
    # Calculate number of cells for analysis and prepare results matrix
    num_of_cells = np.ceil((boundaries[1] - boundaries[0]) / CELL_WIDTH).astype('int')
    
    results = np.zeros((num_frames, num_of_cells), dtype=bool)
    for frame_num, frame_data in frames.items():
        occupied_indices = analyze_frame(frame_data, boundaries[0])
        results[frame_num - 1][occupied_indices] = True
        
    return results


def parse_file(filename):
    """
    Parses a .mat file containing an stk object.
    Returns list of locations by frame.
    """
    t = time.time()
    stk = parse_stk(filename)
    frames, min_x, max_x, num_frames = group_by_frame(stk)
    boundaries = (min_x, max_x)
    res = analyze_positions(frames, boundaries, num_frames)
    print(f'{filename}:\telapsed {time.time() - t:.2f} seconds')
    return res


def get_probabilities(analysis_res):
    """Calculate the probabilities of the first cell being blocked,
    first free but second blocked, first and second blocked but third
    free, etc.

    Parameters
    ----------
    analysis_res : list of lists of booleans returned from parse_file indicating blocked positions in
                    each frame

    Returns
    -------
    List of p0, p1, ... pN - probabilities of each case as explained above
    """
    # Find indices of first occupied (True value which is the maximum value for a binary variable)
    # cell in each frame
    first_occupied = np.argmax(analysis_res, axis=1)
    _, counts = np.unique(first_occupied, return_counts=True)
    # Return normalized counts of each cell being the first occupied one
    return counts / first_occupied.size


def multi_plot(plot_method_name, num_plots, plot_args, num_columns=3,
               main_title=None, titles=None, xtitle=None, ytitle=None,
               show=True, setp_kwargs=None, is_wide=False,
               **kwargs):
    """Draw many plots on same figure and save to file named main_title

    Parameters
    ----------
    plot_method_name: name of the plotting function to call on each subfigure
    num_plots: number of subplot to draw
    plot_args: list of arguments to pass to each plot method
    num_columns: number of columns to draw plot in
    main_title: name of the whole figure
    titles: list of titles for each subplot
    xtitle: title for x-axis
    ytitle: title for y-axis
    setp_kwargs : dictionary of kwargs to set on all subplot objects
    show : whether to show the plot (otherwise, user should call plt.show() manually)
    is_wide : Whether subplots should be wide (default is square)
    kwargs: keyword args to pass to plotting function
    """
    num_rows = int(np.ceil(num_plots / num_columns))
    width = 8 * num_columns if is_wide else 4 * num_columns
    fig, axs = plt.subplots(num_rows, num_columns,
                            figsize=[width, 4 * num_rows + 1],
                            constrained_layout=True,
                            sharex=True)
    # Plot each result in its place
    for i, item in enumerate(plot_args):
        print("Plotting", titles[i], "...")
        plot_func = getattr(axs[i // num_columns, i % num_columns], plot_method_name)
        plot_func(*item, **kwargs)
        if any(titles):
            axs[i // num_columns, i % num_columns].set_title(titles[i])
        if xtitle:
            axs[i // num_columns, i % num_columns].set_xlabel(xtitle)
        if ytitle:
            axs[i // num_columns, i % num_columns].set_ylabel(ytitle)
    # fig.suptitle(main_title)
    if setp_kwargs:
        plt.setp(axs, **setp_kwargs)
    fig.suptitle(main_title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig("heatmaps.png", bbox_inches='tight')
    if main_title:
        fig.savefig(f"{main_title}.png", bbox_inches='tight')
    if show:
        plt.show()


def analyze_first_passage(analysis_res):
    """
    Finds consecutive frames in analysis_res which are blocked.
    The constant-volume experiment is divided into parts, each starting
    when the first cell is occupied and ending when it is cleared from
    hexbugs. The distribution of part lengths, as well as first free cell,
    is returned.
    """
    # Amount of frames it took the bugs between filling the first cell until
    # freeing it
    experiment_lengths = []
    # The index of the first occupied cell (how far could the barrier have been moved)
    first_occupied_cells = []
    # short_frames = []

    current_duration = 0
    current_frame = 0
    # Iterate through frames
    while current_frame < analysis_res.shape[0]:
        # Check if the first cell is occupied
        if analysis_res[current_frame, 0] != 0:
            current_duration += 1
        # No longer occupied
        else:
            # Ignore too short occupied durations - assuming it is a capsized bug
            # or similar issue
            if current_duration > 2:
                experiment_lengths.append(current_duration)
                # Add index of first nonzero cell (non occupied) to list
                occupied = np.flatnonzero(analysis_res[current_frame])
                if len(occupied) > 0:
                    first_occupied = np.min(occupied)
                    first_occupied_cells.append(first_occupied)
            # For debugging: analyze short frames to see what wen wrong
            # elif current_duration != 0:
            #     short_frames.append(i)

            # Reset current duration
            current_duration = 0
        current_frame += 1

    return np.array(experiment_lengths) / FPS, first_occupied_cells


def correlation_peaks(lags, values):
    """Finds and plots peaks in data returned from correlation function.

    Parameters
    ----------
    lags : np.ndarray
        Array of lags of type 'int32'
    values : np.ndarray
        Array of correlation values of type 'float64'

    Returns
    -------
    peak_indices : np.ndarray
        An array containing indices of peak values
    properties : dict
        Dictionary returned from find_peaks that contains peaks' properties
    """
    # Only look for peaks in positive correlations - this line replaces negative values with zeros (only for analysis)
    positive_corrs = np.clip(values, 0, None)
    peak_indices, properties = scipy.signal.find_peaks(positive_corrs, prominence=0.02, width=0)

    # Plot peaks
    plt.plot(lags[peak_indices], values[peak_indices], "x")
    # Plot peak heights
    plt.vlines(x=lags[peak_indices], ymin=0,
               ymax=values[peak_indices], color="C1")
    # Plot peak widths
    plt.hlines(y=properties['width_heights'], xmin=lags[np.floor(properties["left_ips"]).astype('int')],
               xmax=lags[np.floor(properties["right_ips"]).astype('int')], color="C1")
    return peak_indices, properties


def autocorrelations(analysis_res):
    """
    Finds and plots autocorrelations in engine blocked/unblocked data.
    Returns full correlation results (list of lags and list of correlation values for each lag), as well as list of
    positive peaks and their widths
    """
    # Extract first column from results
    blocked = analysis_res[:, 0].astype("float")
    # Normalize data
    blocked -= blocked.mean()

    # Find correlations.
    # Returns lists of:
    # [lags (frames)] [correlation value] ...
    corrs = plt.acorr(blocked, maxlags=None, usevlines=True)
    lags = corrs[0]
    values = corrs[1]

    # # Find peaks in autocorrelation results
    # peak_indices, properties = correlation_peaks(lags, values)
    #
    # # Return only positive lags (acorr result is symmetrical)
    # positive_peak_indices = np.where(lags[peak_indices] >= 0)
    # # Get only positive peaks out of full peaks list
    # peaks = lags[peak_indices][positive_peak_indices]
    # peak_values = values[peak_indices][positive_peak_indices]
    # peak_widths = properties['widths'][positive_peak_indices]
    return lags, values#, peaks, peak_values, peak_widths


def fourier_peak_fit(data, should_plot=False, title=None, prominence=2,
                     rel_height=0.5, smoothing_window_size=301):
    """Calculates Fourier transform of data. Smooths and finds first peak, fits it to a gaussian
    and returns fit parameters and their errors.

    Parameters
    ----------
    data : data to operate on
    should_plot : whether results should be plotted

    Returns
    -------
    Fit parameters (to pass to lab.gaussian) and a list of their errors
    """
    if title:
        print(f"Fitting peaks for {title} peaks...")

    # Fourier transform of correlations graph. Sampling rate is 1/frames per second
    xf = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/FPS))
    fourier = np.fft.fftshift(np.abs(np.fft.fft(data)))

    # Trim frequencies below zero
    indices_to_keep = np.where(xf >= 0)
    xf = xf[indices_to_keep]
    fourier = fourier[indices_to_keep]

    env_peaks = []
    while not any(env_peaks):
        envelope = scipy.signal.savgol_filter(np.abs(fourier), smoothing_window_size, 1)
        env_peaks, properties = scipy.signal.find_peaks(envelope, prominence=prominence,
                                                        width=0, rel_height=rel_height, distance=10000)
        # Maybe data is very fuzzy. Try smoothing it using a bigger window size
        smoothing_window_size += 100
        # Prevent an infinite loop by limiting the smooth amount
        if smoothing_window_size >= 1000:
            break

    plt.clf()

    if len(env_peaks) == 0:
        if should_plot:
            plt.plot(xf, envelope, label=f"Rolling average (window size {smoothing_window_size - 100})")
            plt.xlim(0, 1)
            if title:
                plt.title(f"No peaks found in {title}")
                plt.savefig(f"{title}.png")
            else:
                plt.title("No peaks found here")
            plt.show()
        return None, None

    width = [xf[properties["right_ips"].astype("int")] - xf[properties["left_ips"].astype("int")]][0][0]
    params, param_errs, _, _ = lab.fit(lab.gaussian,
                                       xf[properties["left_ips"].astype("int")[0]:properties["right_ips"].astype("int")[
                                           0]],
                                       envelope[
                                       properties["left_ips"].astype("int")[0]:properties["right_ips"].astype("int")[
                                           0]],
                                       None,
                                       params_guess=(properties["width_heights"][0], xf[env_peaks][0], width / 2))

    plt.plot(xf, envelope, label=f"Rolling average (window size {smoothing_window_size - 100})")
    plt.plot(xf[env_peaks], envelope[env_peaks], "x", label="Peaks")
    plt.xlim(0, 1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Autocorrelations Fourier transform")
    plt.vlines(x=xf[env_peaks], ymin=envelope[env_peaks] - properties["prominences"],
               ymax=envelope[env_peaks], color="C1")
    harmonies = np.array([2, 3, 4]) * xf[env_peaks][0]
    plt.vlines(x=harmonies, ymin=0, ymax=envelope[env_peaks][0], color="grey", alpha=0.5)
    plt.hlines(y=properties["width_heights"], xmin=xf[properties["left_ips"].astype("int")],
               xmax=xf[properties["right_ips"].astype("int")], color="C1")
    fit_range = xf[properties["left_ips"].astype("int")[0]:properties["right_ips"].astype("int")[0]]
    fit_curve = [lab.gaussian(params, x) for x in fit_range]
    plt.plot(fit_range, fit_curve, "-", label="Gaussian fit")
    plt.legend()
    if title:
        plt.title(title)
        plt.savefig(f"{title}.png")
    if should_plot:
        plt.show()
    return params, param_errs


def first_decay_slope(lags, values, plot=False):
    """Finds the slope of the decay of the first maximum in the autocorrelation
    graph in a semilog plot

    Parameters
    ----------
    lags : lags returned from autocorrelations
    values : autocorrelation values
    plot : bool
        Whether the fit should be plotted

    Returns
    -------
    The slope of the linear fit of the semilog plot
    """
    # Find first decay slope
    # Return only positive lags (acorr result is symmetrical)
    positive_indices = np.where(lags >= 0)
    lags = lags[positive_indices]
    values = values[positive_indices]
    # Decay seems to become less reliable right before reaching zero, so taking all but
    # the last two results
    first_negative_index = np.where(values <= 0)[0].min() - 2

    decay_frames = lags[:first_negative_index]
    decay_values = np.log(values[:first_negative_index])
    params, param_errs, reduced_chi_squared, p_value = lab.fit(lab.line, decay_frames, decay_values, None, params_guess=[0,0])
    if plot:
        lab.plot(lab.line, params, decay_frames, decay_values, None, fmt=".",
                 title=f"y={params[0]:.2f}x+{params[1]:.2f}")
        plt.show()
    return params[0]


def correlation_fit_trends(correlation_fits, lengths, xtitle):
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


def first_passages(analysis_results, lengths):
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
    return unique_lengths, durations, durations_avg, durations_stdev


def information(occupied_probabilities):
    """Calculates the information contained in a list of probabilities

    Parameters
    ----------
    occupied_probabilities : The probability of each cell being the first occupied

    Returns
    -------
    The sum of p*ln(p) for probabilities in the input list
    """
    return - np.sum(occupied_probabilities * np.log(occupied_probabilities))


def plot_information(controlled_variable, infos, xtitle):
    """Plots information as function of a controlled variable

    Parameters
    ----------
    controlled_variable : List of values of the variable being controlled in the experiment
    infos : list of information for each value in controlled_variable
    xtitle : Name of the controlled variable to be shown on the plot
    """
    plt.plot(controlled_variable, infos, ".")
    plt.xlabel(xtitle)
    plt.ylabel("Information")
    plt.title(f"Information as function of {xtitle}")
    plt.savefig("info_to_size.png")
    plt.show()


def plot_probabilities(controlled_variable, probabilities, xtitle,
                       show_p0=False):
    """ Plots probabilities for first cell being blocked, first free but second blocked
    etc. as function of controlled variable. See plot_information for full documentation

    Parameters
    ----------
    probabilities : list of lists of probabilities of cells being occupied in
                    each execution
    show_p0 : whether to draw probability that wall cennot be moved at all or not
    """
    probabilities_dict = collections.defaultdict(list)
    controlled_var_dict = collections.defaultdict(list)
    # j indexes the different executions
    for j, distribution in enumerate(probabilities):
        # i indexes probabilities inside a specific execution
        for i, probability in enumerate(distribution):
            probabilities_dict[i].append(probability)
            controlled_var_dict[i].append(controlled_variable[j])

    for key in probabilities_dict.keys():
        if not show_p0 and key == 0:
            continue
        plt.plot(controlled_var_dict[key], probabilities_dict[key], ".",
                 label=f"p{key}")
    plt.legend()
    plt.title("Probabilities of amount of first consecutive free cells")
    plt.ylabel("Probability")
    plt.xlabel(xtitle)
    plt.savefig("Probabilities_to_sizes.png")
    plt.show()
