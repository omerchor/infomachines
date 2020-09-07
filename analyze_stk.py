#!/usr/bin/env python
# coding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os.path
import glob
import time
from multiprocessing import Pool

CELL_WIDTH = 50


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
    t = time.time()
    stk = parse_stk(filename)
    frames, min_x, max_x = group_by_frame(stk)
    boundaries = (min_x, max_x)
    res = analyze_positions(frames, boundaries)
    print(filename, "- elapsed", time.time() - t, "seconds")
    return res


def get_probabilities(analysis_res):
    # Sum over columns and divide by number of rows (frames)
    return analysis_res.sum(axis=0) / analysis_res.shape[0]


if __name__ == "__main__":
    files = glob.glob("stks/*.mat")
    infos = np.zeros(len(files))
    lengths = [float(os.path.basename(os.path.splitext(filename)[0]).replace("_2","").replace("_1","")) for filename in files]
    durations = {}  # cell(numel(files), 1)
    autocorrelations = {}  # cell(numel(files), 1);

    with Pool(8) as p:
        analysis_res = p.map(parse_file, files)
        occupied_probabilities = p.map(get_probabilities, analysis_res)
        fig, axs = plt.subplots(int(np.ceil(len(occupied_probabilities) / 3)), 3)
