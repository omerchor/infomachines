#!/usr/bin/env python
# coding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time

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
    return groups, min_x, max_x


def analyze_frame(data, lower_x_boundary):
    """
    Return indices of occupied cells in frames
    """
    return np.unique((data - lower_x_boundary) // CELL_WIDTH).astype('int')


def analyze_positions(frames, boundaries):
    """
    Extracts position distribution from frames dict
    Returns a list of cell occupations (boolean - is occupied or not) for each frame
    """
    # Calculate number of cells for analysis and prepare results matrix
    num_of_cells = np.ceil((boundaries[1] - boundaries[0]) / CELL_WIDTH).astype('int')
    
    results = np.zeros((len(frames), num_of_cells), dtype=bool)
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
    frames, min_x, max_x = group_by_frame(stk)
    boundaries = (min_x, max_x)
    res = analyze_positions(frames, boundaries)
    print(f'{filename}:\telapsed {time.time() - t:.2f} seconds')
    return res


def get_probabilities(analysis_res):
    # Sum over columns and divide by number of rows (frames)
    return analysis_res.sum(axis=0) / analysis_res.shape[0]


def multi_plot(plot_method_name, num_plots, plot_args, num_columns=3,
               main_title=None, titles=None, xtitle=None, ytitle=None,
               **kwargs):
    """
    Draw many plots on same figure.
    @param plot_method_name: name of the plotting function to call on each subfigure
    @param num_plots: number of subplot to draw
    @param plot_args: list of arguments to pass to each plot method
    @param num_columns: number of columns to draw plot in
    @param main_title: name of the whole figure
    @param titles: list of titles for each subplot
    @param xtitle: title for x-axis
    @param ytitle: title for y-axis
    @param kwargs: keyword args to pass to plotting function
    """
    num_rows = int(np.ceil(num_plots / num_columns))
    fig, axs = plt.subplots(num_rows, num_columns,
                            figsize=[12, 4 * num_rows + 1],
                            constrained_layout=True,
                            sharex=True)
    # Plot each result in its place
    for i, item in enumerate(plot_args):
        plot_func = getattr(axs[i // num_columns, i % num_columns], plot_method_name)
        plot_func(*item, **kwargs)
        if any(titles):
            axs[i // num_columns, i % num_columns].set_title(titles[i])
        if xtitle:
            axs[i // num_columns, i % num_columns].set_xlabel(xtitle)
        if ytitle:
            axs[i // num_columns, i % num_columns].set_ylabel(ytitle)
    # fig.suptitle(main_title)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    if main_title:
        fig.savefig(f"{main_title}.png", bbox_inches='tight')
    plt.show()


def plot_probabilities(probabilities, titles):
    cells = [range(1, len(item) + 1) for item in probabilities]
    args = zip(cells, probabilities)
    multi_plot("bar", len(probabilities), args,
               main_title="Occupied probabilities", titles=titles)
    # fig, axs = plt.subplots(int(np.ceil(len(probabilities) / num_columns)), num_columns)
    #
    # # Plot each result in its place
    # for i, item in enumerate(probabilities):
    #     p = axs[i // num_columns, i % num_columns].bar(range(1, len(item) + 1), item)
    #     axs[i // num_columns, i % num_columns].set_title("Length: " + str(titles[i]))
    # fig.tight_layout()
    # plt.title("Occupied probabilities")
    # plt.show()


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
    # The index of the last free cell (how far could the barrier have been moved)
    first_free_cells = []
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
                first_free_cells.append(np.min(analysis_res[current_frame].nonzero()))
            # For debugging: analyze short frames to see what wen wrong
            # elif current_duration != 0:
            #     short_frames.append(i)

            # Reset current duration
            current_duration = 0
        current_frame += 1

    return np.array(experiment_lengths) / FPS, first_free_cells


def correlations(analysis_res):
    """
    Finds correlations between times engine is blocked
    """
    # Extract first column from results
    blocked = analysis_res[:, 0]
    cor = np.correlate(blocked, blocked, mode='same')
    plt.acorr(cor)
    plt.show()
    return cor
