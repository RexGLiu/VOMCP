#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:41:09 2017

@author: Rex

Script for running multiple episodes and outputting summary of results.
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

model_path = "../results/explore_split_old"

x_axis = range(5000,30000,5000)
splits = [ f"0.{ii}" for ii in range(1,6) ]


def get_nsamples(filename):
    nsamples = int(filename.split('nsamples=')[1].split(',_')[0])
    return nsamples

for ii, split in enumerate(splits):
    file_folder = os.path.join(model_path, f"split={split}")
    file_list = glob.glob(file_folder + "/*.npy")
    file_list.sort(key=get_nsamples)
    if len(file_list) > len(x_axis):
        file_list = file_list[:-1]

    n_datapts = len(file_list)
    return_rate = [None] * n_datapts
    return_err = [None] * n_datapts

    for jj, file in enumerate(file_list):
        print(file)
        with open(file, "rb") as f:
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
    plt.figure(1, figsize=(4.75, 4))
    plt.errorbar(x_axis, return_rate, yerr=return_err, label="c={}".format(split), linewidth=1, alpha=0.9)


plt.figure(1)
plt.xlabel("total number of Monte-Carlo samples")
plt.ylabel("average return")
plt.xlim([x_axis[0] - 500, x_axis[-1] + 500])
plt.xticks(x_axis)

plt.legend(loc="lower left", prop={"size": 10})
plt.tight_layout()

# save_folder = "rerun/graphs"
# file_name = f"{model} - perf_vs_cfactor"
# save_path = os.path.join(save_folder, file_name)
# plt.savefig(save_path+".png", format="png", dpi=250)
# plt.savefig(save_path+".pdf", format="pdf")
plt.show()


