#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:41:09 2017

@author: Rex

Script for running multiple episodes and outputting summary of results.
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def get_nsamples(filename):
    nsamples = int(filename.split('nsamples=')[1].split(',_')[0])
    return nsamples

model_list = ["VOMCP", "EAF", "EAF_modified", "EA"]
for model in model_list:
    model_path = os.path.join("../results/explore_p/", model, "30000")
    file_list = glob.glob(model_path + "/*.npy")
    file_list.sort(key=get_nsamples)

    n_datapts = len(file_list)
    return_rate = [None] * n_datapts
    return_err = [None] * n_datapts
    p_vals = [None] * n_datapts

    for ii, file in enumerate(file_list):
        print(file)

        p_vals[ii] = float(file.split('pval_thresh=')[1].split(',_c=')[0])

        with open(file, "rb") as f:
            obs_count = np.load(f)
            visit_count = np.load(f)
            success_array = np.load(f)
            pathlen_array = np.load(f)
            tot_rew_array = np.load(f)
            R_array = np.load(f)
            obs_count_array = np.load(f)

        nruns = R_array.shape[0]
        return_rate[ii] = np.mean(R_array)
        return_err[ii] = np.sqrt(np.var(R_array) / nruns)

    return_rate = [r for _, r in sorted(zip(p_vals, return_rate))]
    return_err = [r for _, r in sorted(zip(p_vals, return_err))]
    p_vals.sort()

    print(model, p_vals)

    plt.figure(1, figsize=(4.75, 4))
    plt.errorbar(p_vals, return_rate, yerr=return_err, label=model, linewidth=1, alpha=0.9)

plt.figure(1)
plt.xlabel("p-val threshold")
plt.ylabel("average return")
plt.title("Performance vs p-val threshold")

plt.legend(loc="lower left", prop={"size": 10})
plt.tight_layout()

# save_folder = "rerun/graphs"
# file_name = f"{model} - perf_vs_cfactor"
# save_path = os.path.join(save_folder, file_name)
# plt.savefig(save_path+".png", format="png", dpi=250)
# plt.savefig(save_path+".pdf", format="pdf")
plt.show()