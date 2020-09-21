#!/usr/bin/env python
# coding: utf-8
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import skimage.transform

# %%
CELLS_RATIO = 15
#%%
DIRNAME = "stks"  # "num_hexbugs_stks"
# DIRNAME = "num_hexbugs_stks"
files = os.listdir(DIRNAME)
files.sort()


#%%
def load_stk(filename):
    # Load .mat file
    print(filename)
    f = sio.loadmat(os.path.join(DIRNAME, filename))
    return f['stk']


def find_densities(filename):
    stk = load_stk(filename)

    x_boundaries = [stk[:, 0].min(), stk[:, 0].max()]
    y_boundaries = [stk[:, 1].min(), stk[:, 1].max()]
    # Figure image size from the min/max positions a hexbug was found at
    resolution = np.ceil(np.array([y_boundaries[1] - y_boundaries[0],
                                   x_boundaries[1] - x_boundaries[0]])).astype("int")
    pixels = np.zeros(resolution)
    # Update pixels matrix frame by frame
    current_pixels = np.zeros(resolution)
    current_frame = 1
    for entry in stk:
        # If we have advanced a frame, update the full table and reset
        if entry[4] != current_frame:
            current_frame = entry[4]
            pixels += current_pixels
            # Reset the current pixels matrix for the next frame
            current_pixels.fill(0)
        # Only include bright enough (at least 5) dots
        if entry[2] >= 5:
            x = np.floor(entry[0] - x_boundaries[0]).astype("int")
            y = np.floor(entry[1] - y_boundaries[0]).astype("int")
            current_pixels[y, x] = 1

    # Add last frame
    pixels += current_pixels
    return pixels, current_frame


#%%
if __name__ == "__main__":
    with Pool(8) as p:
        results = p.map(find_densities, files)
#%%
    small_results = [skimage.transform.downscale_local_mean(res[0], (CELLS_RATIO, CELLS_RATIO)) for res in results]

    # Join results for same board size
    unique_filenames = list(set([float(f.replace("_1", "").replace("_2", "").replace(".mat", ""))
                                 for f in files]))
    unique_filenames.sort()
    unique_results = []
    for filename in unique_filenames:
        indices = [i for i, f in enumerate(files) if
                   float(f.replace("_1", "").replace("_2", "").replace(".mat", "")) == filename]
        total_frames = np.sum([results[i][1] for i in indices])
        new_res = None
        for i in indices:
            # Normalize result by dividing by num of frames
            if new_res is None:
                new_res = small_results[i] * (results[i][1] / total_frames)
            # Add additional file with same length
            else:
                # Normalize new image with its num of frames
                second_img = skimage.transform.resize(small_results[i], new_res.shape)
                new_res += (second_img / (results[i][1] / total_frames))
        unique_results.append(new_res)
    #%%
    # Create multiple figures for all results (two in a row - adds as many rows needed)
    num_columns = 3
    num_rows = int(np.ceil(len(unique_results) / num_columns))
    # resimgs = [r[0] for r in results]
    fig, axs = plt.subplots(num_rows, num_columns,
                            figsize=(4 * num_columns, 4 * num_rows))
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])

    # Plot each result in its place
    for i, item in enumerate(unique_results):
        # Normalize values
        vmin = item.min()
        vmax = item.max()
        item = (item - vmin)/(vmax - vmin)

        # Show the heatmap
        # vmax can be used to add a maximal value - the laser is sometimes detected in the
        # image, making the area near the moving wall appear significantly busier
        im = axs[i // num_columns, i % num_columns].imshow(item, interpolation='gaussian',
                                                           cmap='inferno', vmax=0.7)
        axs[i // num_columns, i % num_columns].set_title(unique_filenames[i])

    # Add colorbar
    cb = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Heatmaps of hexbug positions for different board size (with 6 hexbugs)",
                 fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig("heatmaps.png", bbox_inches='tight')
    fig.show()
