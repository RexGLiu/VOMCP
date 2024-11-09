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

# model_path = "../results/hard_env"


model = "POMCP"
model_path = "../results/explore_c/" + model

# model = "VOMCP"
# n_vos = 1
# model_path = os.path.join("../results/explore_c/",model,f"n_vobs={n_vos}")

model_path = "../results/hard_env/15000"


def get_nsamples(filename):
    nsamples = int(filename.split('nsamples=')[1].split(',_')[0])
    return nsamples

file_list = glob.glob(model_path + "/*.npy")
file_list.sort(key=get_nsamples)

for file in file_list:
    print(file)
    with open(file, "rb") as f:
        obs_count = np.load(f)
        visit_count = np.load(f)
        success_array = np.load(f)
        pathlen_array = np.load(f)
        tot_rew_array = np.load(f)
        R_array = np.load(f)
        obs_count_array = np.load(f)

    nruns = R_array.shape[0]
    pathlen_array = np.array(pathlen_array)
    tot_rew_array = np.array(tot_rew_array)

    print("return: ", np.mean(R_array))
    print("tot rew: ", np.mean(tot_rew_array))
    print("success: ", np.mean(success_array))
    print("obs count:", np.mean(obs_count_array))
    print("path len: ", np.mean(pathlen_array))
    # print(R_array)
    print(np.mean(np.power(0.95,pathlen_array-1)*tot_rew_array))
    print(len(tot_rew_array))