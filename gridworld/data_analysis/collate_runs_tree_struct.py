#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:41:09 2017

@author: Rex

Script for running multiple episodes and outputting summary of results.
"""
import glob

import numpy as np
import pickle
import os
import glob
import shutil as sh
import matplotlib.pyplot as plt
import argparse

def collate_files(filepath, out_folder):
    env_size = 36

    file_list = glob.glob(filepath + "/*.pkl")
    output_name = os.path.basename(file_list[0])

    # strip 'seed=X' from filename
    output_name = output_name.split(',_')
    output_name = output_name[:-1]
    output_name = ',_'.join(output_name)
    pattern = r'_(\d+)_runs' # we will use this regex to replace the number of runs in the name with the correct total number of runs

    N_obs_root = []
    N_obs_branches = []
    obs_branch_visits = []
    decision_flag_counts = []

    for filepath in file_list:
        with open(filepath, "rb") as f:
            pickle.load(f)
            pickle.load(f)
            pickle.load(f)

            N_obs_root_ = pickle.load(f)
            for x in N_obs_root_:
                N_obs_root += x

            N_obs_branches_ = pickle.load(f)
            for x in N_obs_branches_:
                N_obs_branches += x

            obs_branch_visits_ = pickle.load(f)
            for x in obs_branch_visits_:
                obs_branch_visits += x

            decision_flag_counts += pickle.load(f)

    # save collated results to file
    assert "raw_data" not in out_folder, f"Cannot save to path {out_folder} which includes raw_data"
    output_path = os.path.join(out_folder, output_name)
    # with open(output_path + ".npy", "wb") as f:
    #     pickle.dump(nruns, f)

    decision_flag_counts = np.sum(decision_flag_counts, axis=0)
    print(decision_flag_counts)
    plt.hist(N_obs_root, density=True)
    print(np.nanquantile(N_obs_root, q=[0.25,0.5,0.75]))
    plt.show()
    plt.hist(N_obs_branches, density=True)
    print(np.nanquantile(N_obs_branches, q=[0.25, 0.5, 0.75]))
    plt.show()
    plt.hist(obs_branch_visits, density=True, bins=range(0,4200,200))
    print(np.nanquantile(obs_branch_visits, q=[0.25, 0.5, 0.75]))
    plt.ylim([0, 0.0035])
    plt.show()
    plt.hist(obs_branch_visits, density=True, bins=range(0,4200,200))
    plt.ylim([0,0.00055])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="../results/raw_data/tmp/gamma=0.9/", required=False, help="Root folder of results to process")
    args = parser.parse_args()

    # walk through folder and subfolders to process files
    visited_folders = []
    for root, dirs, files in os.walk(args.path, topdown=True):
        # delete all temporary files
        tmp_files = glob.glob(root+"/*_0.*")
        for file in tmp_files:
            os.remove(file)

        # skip if no files to process
        files = glob.glob(root+"/*.npy")
        if len(files) == 0:
            continue

        # remove previous outputs if present
        path_list = root.split(os.path.sep)
        path_list = path_list[0:2] + path_list[3:-1]
        out_folder = os.path.sep.join(path_list)

        if out_folder not in visited_folders:
            visited_folders.append(out_folder)
            if os.path.exists(out_folder):
                assert 'raw_data' not in out_folder, f"Error: cannot delete subfolder {out_path} in raw_data."
                sh.rmtree(out_folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # process files
        collate_files(root, out_folder)