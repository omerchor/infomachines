#!/usr/bin/env python
# coding: utf-8
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.transform
from multiprocessing import Pool
import analyze_stk
# %%
CELLS_RATIO = 19
#%%
# DIRNAME = "stks"  # "num_hexbugs_stks"
DIRNAME = "num_hexbugs_stks"
files = os.listdir(DIRNAME)


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
    # results = [find_densities(f) for f in files]
#%%
    small_results = [skimage.transform.downscale_local_mean(res[0], (CELLS_RATIO, CELLS_RATIO)) for res in results]

    # Join results for same board size
    unique_filenames = list(set([f.replace("_1", "").replace("_2", "") for f in files]))
    unique_results = []
    for filename in unique_filenames:
        indices = [i for i, f in enumerate(files) if f.replace("_1", "").replace("_2", "") == filename]
        new_res = None
        for i in indices:
            # Normalize result by dividing by num of frames
            if new_res is None:
                new_res = small_results[i] / results[i][1]
            # Add additional file with same length
            else:
                # Normalize new image with its num of frames
                second_img = skimage.transform.resize(small_results[i], new_res.shape)
                new_res += second_img / results[i][1]

        unique_results.append(new_res)


    #%%
    # Create multiple figures for all results (two in a row - adds as many rows needed)
    num_columns = 3
    # resimgs = [r[0] for r in results]
    fig, axs = plt.subplots(int(np.ceil(len(unique_results) / num_columns)), num_columns,
                            figsize=(12, 11))
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
        axs[i // num_columns, i % num_columns].set_title("Number of hexbugs: " + str(i+1))

    # Add colorbar
    cb = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Heatmaps of hexbug positions for different amounts of bugs", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig("heatmaps.png", bbox_inches='tight')
    fig.show()
