#!/usr/bin/env python
# coding: utf-8
import collections
import os.path
import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal
import scipy.interpolate

import lab

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
    Returns list of locations by frame, as well as board size in pixels
    """
    t = time.time()
    stk = parse_stk(filename)
    frames, min_x, max_x, num_frames = group_by_frame(stk)
    boundaries = (min_x, max_x)
    res = analyze_positions(frames, boundaries, num_frames)
    print(f"{filename}:\t\t {max_x - min_x} pixels")
    return res, max_x - min_x


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
    normalized_counts : list
        List of p0, p1, ... pN - probabilities of each case as explained above
    cumulative_count : dict
        Dictionary of non-normalized cumulative counts of each first occupied cell
        indices
    """
    # Find indices of first occupied (True value which is the maximum value for a binary variable)
    # cell in each frame
    first_occupied = np.argmax(analysis_res, axis=1)
    unique_vals, counts = np.unique(first_occupied, return_counts=True)

    # Normalized counts of each cell being the first occupied one
    normalized_counts = counts / first_occupied.size

    # Cumulative counts of each value in movie
    cumulative_count = {}
    for value in unique_vals:
        cumulative_count[value] = np.cumsum(first_occupied == value)

    return normalized_counts, cumulative_count


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


def autocorrelations(analysis_res, plot=False):
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
    if plot:
        plt.show()
    else:
        plt.clf()
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
    """Plot peak locations and widths for all gaussians fitted to
    different results in fourier_peak_fit

    Parameters
    ----------
    correlation_fits : fits returned from fourier_peak_fit
    lengths : list of lengths of board for each correlation_fits
    xtitle : how to name the x axis
    """
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
    """Analyzes sub-experiments in which the engine remains blocked and returns
    their statistics

    Parameters
    ----------
    analysis_results : list of stk results
    lengths : the dependent variable matching each item in analysis_results

    Returns
    -------
    unique_lengths
        list of unique titles
    durations
        durations of each unique experiment
    durations_avg
        average duration for each unique experiment
    durations_stdev
        standard deviation of the experiment durations
    """
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


def plot_information(controlled_variable, infos, xtitle, groups=None, group_sizes=None):
    """Plots information as function of a controlled variable

    Parameters
    ----------
    controlled_variable : List of values of the variable being controlled in the experiment
    infos : list of information for each value in controlled_variable
    xtitle : Name of the controlled variable to be shown on the plot
    """
    if not groups:
        plt.plot(controlled_variable, infos, ".")
    else:
        for i, group in enumerate(groups):
            previous_group = group_sizes[i - 1] if i >= 1 else 0
            plt.plot(controlled_variable[previous_group:previous_group + group_sizes[i]],
                     infos[previous_group:previous_group + group_sizes[i]], ".",
                     label=group)
        plt.legend()
    plt.xlabel(xtitle)
    plt.ylabel("Information")
    plt.title(f"Information as function of {xtitle}")
    plt.savefig(f"info_to_{xtitle}.png")
    plt.show()


# def probabilities_fit_function(B, x):
#     return scipy.interpolate.barycentric_interpolate


def interpolate_probability(controlled_variables, probabilities, params_guess=(0.1, 0.1, 0.01)):
    """Finds a fit for probabilities as function of the controlled variable.

    Parameters
    ----------
    controlled_variables : list
        list of values of the controlled variable in the experiment (x-axis)
    probabilities : list
        list of experiment results (probability) for each value in controlled_variables (y-axis)

    Returns
    -------
    The interpolation function for the x,y data
    """
    try:
        return scipy.interpolate.interp1d(controlled_variables, probabilities, bounds_error=False, fill_value=0,
                                          kind='slinear')#, fill_value="extrapolate")
    except ValueError:
        def zero_func(x):
            return 0
        zero_func = np.vectorize(zero_func)
        return zero_func
    new_guess = (max(probabilities), params_guess[1], params_guess[2], min(controlled_variables))
    # params, param_errs, _, _ = lab.fit(probabilities_fit_function, controlled_variables, probabilities, None,
    #                                    (max(probabilities), params_guess[1], params_guess[2]))
    #
    # # Since extrapolation may return negative values, fix it by replacing them with zero
    # def fit_func(x):
    #     result_probability = probabilities_fit_function(params, x)
    #     if result_probability > 0:
    #         return result_probability
    #     return 0
    # fit_func = np.vectorize(fit_func)
    #
    # return fit_func


def organize_probabilities(controlled_variable, probabilities):
    """Organizes probabilities by event (p0, p1, p2...) rather than by experiment (board size)


    Returns
    -------
    Dictionary of board sizes and probability value by probability indices
    """
    probabilities_dict = collections.defaultdict(list)
    controlled_var_dict = collections.defaultdict(list)

    # Extract data from all executions to lists indexed by the probability (rather than by experiment which is how the
    # raw data is sorted)
    # j indexes the different executions
    for j, distribution in enumerate(probabilities):
        # i indexes probabilities inside a specific execution
        for i, probability in enumerate(distribution):
            probabilities_dict[i].append(probability)
            controlled_var_dict[i].append(controlled_variable[j])

    return controlled_var_dict, probabilities_dict


def fit_probabilities(controlled_variable, probabilities):
    """Finds a fit for probabilities of each event (p0, p1, p2...) as function of controlled variable

    Returns
    -------
    List of functions returning estimated probability at a certain controlled variable value:
    [p_0, p_1, p_2...] - p_i(controlled_value) returns the probability of i-th event at controlled_value
    """
    # Find fits for the different events as function of board size
    controlled_var_dict, probabilities_dict = organize_probabilities(controlled_variable, probabilities)
    probability_funcs = []
    for probability_index in probabilities_dict.keys():
        func = interpolate_probability(controlled_var_dict[probability_index],
                                               probabilities_dict[probability_index])
        probability_funcs.append(func)
    return probability_funcs


def plot_probabilities(controlled_variable, probabilities, xtitle, show_p0=True, is_num_hexbugs=False,
                       params_guess=(0.1, 0.1, 0.01)):
    """ Plots probabilities for first cell being blocked, first free but second blocked
    etc. as function of controlled variable. See plot_information for full documentation.

    Parameters
    ----------
    probabilities : list of lists of probabilities of cells being occupied in
                    each execution
    show_p0 : whether to draw probability that wall cennot be moved at all or not
    is_num_hexbugs : whether controlled_variable is number of hexbugs (or another parameter
                     such as board size). If True, the expected value if hexbugs were independent
                     will be plotted)

    Returns
    ----------
    List of functions of interpolated probabilities as function of controlled variable (p0, p1, p2...)
    """
    controlled_var_dict, probabilities_dict = organize_probabilities(controlled_variable, probabilities)

    # Keys here are p0, p1, p2, ...
    interpolated_funcs = []
    for probability_index in probabilities_dict.keys():
        if not show_p0 and probability_index == 0:
            continue
        p = plt.plot(controlled_var_dict[probability_index], probabilities_dict[probability_index], ".",
                     label=f"p{probability_index}")
        interpolated = interpolate_probability(controlled_var_dict[probability_index],
                                               probabilities_dict[probability_index],
                                               params_guess)
        # Plot interpolation
        data_range = np.linspace(100,
                                 max(controlled_var_dict[probability_index]),
                                 1000)
        interpolated = interpolated(data_range)
        interpolated_funcs.append(interpolated)
        plt.plot(data_range, interpolated, "--", color=p[0].get_color(), label=f"p{probability_index} (fit)")
        plt.plot()

        if is_num_hexbugs:
            # Plot expected probability if hexbugs were independent
            # Formula is explained here:
            # https://trello-attachments.s3.amazonaws.com/5f55ec77266d3d5d8c7f40dc/5f55ed343da4ba75b16c11c0/0abd5aa3c50e4832edfd54174957961e/image.png
            allowed_area = 1 - ((probability_index + 1) / len(probabilities_dict))
            # This is the probability of bugs not being in the
            x_data = np.linspace(min(controlled_var_dict[probability_index]), max(controlled_var_dict[probability_index]),
                                 1000)
            expected = [(allowed_area ** (c - 1)) * (c / len(probabilities_dict)) for c in x_data]
            plt.plot(x_data,
                     expected,
                     "--+", color=p[0].get_color(),
                     label=f"p{probability_index} (naive)")

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title("Probabilities of amount of first consecutive free cells")
    plt.ylabel("Probability")
    plt.xlabel(xtitle)
    # plt.semilogy()
    plt.tight_layout()
    plt.savefig(f"Probabilities_to_{xtitle}.png")
    plt.show()

    return interpolated_funcs


def plot_cumulative_probabilities(controlled_variable, cumulative_counts,
                                  show_p0=True):
    """Plots the cumulative probability of each case as function of time as well as differences series of the probabilities

    Parameters
    ----------
    cumulative_counts
    """
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10))

    for key, cumulative_counts in cumulative_counts.items():
        if not show_p0 and key == 0:
            continue
        num_frames = len(cumulative_counts)
        times = np.linspace(0, num_frames / FPS, num_frames)
        probabilities = [count / (frame_index + 1) for frame_index, count in enumerate(cumulative_counts)]
        diffs = np.abs(np.diff(probabilities))
        axs[0].plot(times, probabilities, label=f"p{key}")
        axs[0].set_ylabel("Probability")
        # The differences series has one item less
        axs[1].plot(times[1:], diffs, ".", label=f"p{key} differences", markersize=5)
        axs[1].set_ylabel("Differences")

    axs[0].legend(bbox_to_anchor=(1.05, 1))
    axs[1].legend(bbox_to_anchor=(1.05, 1))
    axs[1].semilogy()
    plt.xlabel("Time (sec)")
    fig.suptitle(f"Cumulative probabilities for {controlled_variable}")

    path = os.path.join("cumulative", f"{str(controlled_variable)}.png")
    # Don't override existing
    if os.path.isfile(path):
        path = os.path.join("cumulative", f"{str(controlled_variable)}_2.png")
    plt.savefig(path)
    plt.show()


def split_correlations(analysis_results, title):
    """Finds and plots autocorrelations for a certain experiment after dividing it into
    shorter experiments.
    If the correlation are real, they should appear in the shorter experiments too. This
    is one way to overcome problems with initial conditions.

    Parameters
    ----------
    analysis_results : positions of bugs in the experiment
    title : name of the experiment

    Returns
    -------

    """
    sub_experiments = np.array_split(analysis_results, 10)
    correlations = [autocorrelations(r) for r in sub_experiments]
    correlation_fits = [fourier_peak_fit(c[1], True, f"{title} {i}", 2) for i, c in enumerate(correlations)]
    correlation_fit_trends(correlation_fits, range(len(correlation_fits)), "Length (cm)")


def simulate_experiment(length_to_info, lengths, probabilities, covergence_limit=10**-3):
    """Calculates the expected average information for a full compression of the board, starting with the largest length
    and continuing to smaller boards sizes in steps of CELL_WIDTH pixels up to the smallest size in lengths.
    The weighted average information gained per board size is calculated recursively in a dynamic-programming (bottom-up)
    manner.

    Parameters
    ----------
    length_to_info : function
        a function receiving length and returning the interpolated information for that length.
    lengths : list
        measured lengths
    probabilities : list
        list of distributions matching the measurements in lengths
    covergence_limit : float
        smallest barrier movement distance to be considered. Board will stop being compressed when distance differences
        reach this value.

    Returns
    -------
    The total information that is expected to be measured in a single experiment.
    """
    total_info = 0
    # Average distance barrier is moved per step until reaching the minimal distance
    distance_diffs = []
    distance_diffs_stdevs = []
    # Information at each position of the barrier
    info_per_step = []
    # Board sizes at each step
    board_lengths = []

    probability_functions = fit_probabilities(lengths, probabilities)
    move_distances = np.arange(len(probability_functions)) * CELL_WIDTH

    current_length = max(lengths)
    current_diff = 100
    while current_diff > covergence_limit and current_length > CELL_WIDTH:
        # Calculate the average distance the board will now move
        current_probabilities = np.array([f(current_length) for f in probability_functions])
        # Only allow to move the board to positive board sizes
        allowed_indices = np.where(move_distances < current_length - CELL_WIDTH)

        # Reached size in which all probabilities were extrapolated to be 0 (or board too small to move)
        if len(allowed_indices) == 0 or sum(current_probabilities) == 0:
            break

        board_lengths.append(current_length)
        info_per_step.append(length_to_info(current_length))

        current_probabilities = current_probabilities[allowed_indices]
        current_move_distances = move_distances[allowed_indices]
        current_diff = np.average(current_move_distances, weights=current_probabilities)
        distance_diffs.append(current_diff)
        diff_stdev = np.sqrt(np.average((current_move_distances-current_diff)**2, weights=current_probabilities))
        distance_diffs_stdevs.append(diff_stdev)
        current_length -= current_diff

    return board_lengths, info_per_step, distance_diffs, distance_diffs_stdevs
