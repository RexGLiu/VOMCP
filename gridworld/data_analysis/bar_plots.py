#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:41:09 2017

@author: Rex

Script for running multiple episodes and outputting summary of results.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

labels = ["VOMCP\n$n_{vo}=1$", "EAF", "EAF\nmodified", "EA", "POMCP"]

nsamples = 30000
#
datapath = f"../results/hard_env/{nsamples}"
title = f"Maze II (budget={nsamples})"

# labels = ["VOMCP\n$n_{vo}=1$", "EAF", "EAF\nmodified"]
# datapath = f"../results/hard_env2/{nsamples}"
# title = f"Maze I modified (budget={nsamples})"


g=0.9
datapath = f"../results/gamma={g}/{nsamples}"
title = f"Maze I, gamma={g} (budget={nsamples})"

# labels = ["VOMCP\n$n_{vo}="+f"{ii}$" for ii in range(1,9)] #+ ["VOMCP\n$n_{vo}=10$", "VOMCP\n$n_{vo}=15$"]
# datapath = f"../results/hard_env/30000/VOMCP"
# title = f"Maze II (budget=30000)"



folder = Path(datapath)
filenames = [file.stem for file in folder.glob("*.npy")]
filenames.sort()
filenames = [filenames[-1]] + filenames[:-1]
# filenames = filenames[2:] #+ filenames[:2]
print(filenames)


n_datapts = len(labels)
return_rate = [None] * n_datapts
return_err = [None] * n_datapts

for jj, file in enumerate(filenames):
    print(file)

    filepath = os.path.join(datapath, file + ".npy")
    with open(filepath, "rb") as f:
        obs_count = np.load(f)
        visit_count = np.load(f)
        success_array = np.load(f)
        pathlen_array = np.load(f)
        tot_rew_array = np.load(f)
        R_array = np.load(f)

    nruns = R_array.shape[0]

    return_rate[jj] = np.mean(R_array)
    return_err[jj] = np.sqrt(np.var(R_array) / nruns)

print(return_rate)

fig = plt.figure(figsize=(4,2))
plt.bar(range(n_datapts), return_rate, yerr=return_err, color=['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:purple'])
plt.xticks(ticks=range(n_datapts), labels=labels, size=10)
plt.yticks(size=10)
# plt.ylim([30,65])
# plt.ylim(yrange)
plt.title(title)
fig.tight_layout()
plt.show()


