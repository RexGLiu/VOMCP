#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:41:09 2017

@author: Rex

Script for running multiple episodes and outputting summary of results.
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


source_path = "clean_results/20k_particles/aggregate_results"

folder = Path(source_path)
filenames = [file.stem for file in folder.glob("*_n_laser=1_gamma=0.95.npy")]
filenames.sort()
# filenames = filenames[2:] + [filenames[1], filenames[0]]
# filenames[1:-2] = []
print(filenames)

# labels = ["VOMCP", "POMCP", "EA"]
# labels = ["VOMCP 1", "VOMCP 2", "VOMCP 3", "VOMCP 4", "VOMCP 5", "POMCP", "EA"]
labels = ["EA", "VOMCP 1", "VOMCP 2", "VOMCP 3"]
n_datapts = len(labels)

return_rate = [None]*n_datapts
return_err = [None]*n_datapts

for ii, file in enumerate(filenames):
    print(ii, file)

    filepath = os.path.join(source_path,file)
    with open(filepath + ".npy", "rb") as f:
        return_data = np.load(f)
        nruns = return_data.size

    return_rate[ii] = np.mean(return_data)
    return_err[ii] = np.sqrt(np.var(return_data) / nruns)
    print(f"nruns: {nruns}")


fig = plt.figure(figsize=(4.5,3.75))
plt.bar(range(n_datapts), return_rate, yerr=return_err, color=['tab:red', 'tab:orange', 'tab:blue'])
# plt.bar(range(n_datapts), return_rate, yerr=return_err)
plt.xticks(ticks=range(n_datapts), labels=labels, size=13)
plt.yticks(size=12)
plt.ylim((-19.5,-16))
plt.yticks(ticks=range(-19,-15), size=13)
fig.tight_layout()


save_folder = "graphs"
file_name = f"laser_tag_results"
save_path = os.path.join(save_folder, file_name)
# plt.savefig(save_path+".png", format="png", dpi=250)
# plt.savefig(save_path+".pdf", format="pdf")


plt.show()

