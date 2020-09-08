#!/usr/bin/env python
# coding: utf-8
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import analyze_stk
# %%
IMAGE_RESOLUTION = np.array([350,290])
CELLS_RATIO = 19
#%%
files = os.listdir("num_hexbugs_stks")
#%%
results = []
previous = ""
for filename in files:
    # If the file is a new number, reset the pixels matrix
    if filename != previous and filename.replace("_2", "") != previous:
        # Create an empty matrix for results
        pixels = np.zeros(IMAGE_RESOLUTION // CELLS_RATIO)
        previous = filename
    # Otherwise, the existing pixels map will be used. Remove the last items
    # from the results as it will be updated here
    else:
        results.pop()
    
    # Load .mat file
    print(filename)
    f = sio.loadmat("num_hexbugs_stks\\" + filename)
    stk = f['stk']
    
    # Update pixels matrix frame by frame
    current_pixels = np.zeros(IMAGE_RESOLUTION // CELLS_RATIO)
    current_frame = 1
    for entry in stk:
        # If we have advanced a frame, update the full table and reset
        if entry[4] != current_frame:
            current_frame = entry[4]
            pixels += current_pixels
            current_pixels = np.zeros(IMAGE_RESOLUTION // CELLS_RATIO)
        x = int(np.floor(entry[0] / CELLS_RATIO))
        y = int(np.floor(entry[1] / CELLS_RATIO))
        current_pixels[y,x] = 1
        
    # Add last frame
    pixels += current_pixels
    results.append(pixels)
#%%
# Create multiple figures for all results (two in a row - adds as many rows needed)
num_columns = 3
fig, axs = plt.subplots(int(np.ceil(len(results) / num_columns)), num_columns,
                        figsize=(12, 11))
cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])

# Plot each result in its place
for i, item in enumerate(results):
    # Normalize values
    vmin = item.min()
    vmax = item.max()
    item = (item - vmin)/(vmax - vmin)

    # Show the heatmap
    # vmax can be used to add a maximal value - the laser is sometimes detected in the
    # image, making the area near the moving wall appear significantly busier
    im = axs[i // num_columns, i % num_columns].imshow(item, interpolation='gaussian',
                                                       cmap='inferno', vmax=0.5,
                                                       label=i+1)
    axs[i // num_columns, i % num_columns].set_title("Number of hexbugs: " + str(i + 1))
    
# Add colorbar
cb = fig.colorbar(im, cax=cbar_ax)
fig.suptitle("Heatmaps of hexbugs positions for different amounts of bugs", fontsize=16)
fig.subplots_adjust(top=0.98)
fig.tight_layout()
fig.savefig("heatmaps.png", bbox_inches='tight')
fig.show()
