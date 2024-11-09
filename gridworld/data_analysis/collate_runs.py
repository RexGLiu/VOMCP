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
import re
import argparse

def collate_files(filepath, out_folder):
    env_size = 36

    file_list = glob.glob(filepath + "/*.npy")
    output_name = os.path.basename(file_list[0])

    # strip 'seed=X' from filename
    output_name = output_name.split(',_')
    output_name = output_name[:-1]
    output_name = ',_'.join(output_name)
    pattern = r'_(\d+)_runs' # we will use this regex to replace the number of runs in the name with the correct total number of runs

    obs_count = np.zeros(env_size, dtype=np.uint64)
    visit_count = np.zeros(env_size, dtype=np.uint64)
    success_array = []
    pathlen_array = []
    tot_rew_array = []
    R_array = []
    tot_obs_array = []

    for filepath in file_list:
        with open(filepath, "rb") as f:
            obs_count += np.load(f)
            visit_count += np.load(f)
            success_array.append(np.load(f))
            pathlen_array.append(np.load(f))
            tot_rew_array.append(np.load(f))
            R_array.append(np.load(f))
            tot_obs_array.append(np.load(f))

    success_array = np.concatenate(success_array)
    pathlen_array = np.concatenate(pathlen_array)
    tot_rew_array = np.concatenate(tot_rew_array)
    R_array = np.concatenate(R_array)
    tot_obs_array = np.concatenate(tot_obs_array)

    # set the correct number of runs in the filename
    nruns = R_array.shape[0]
    output_name = re.sub(pattern, f'_{nruns}_runs', output_name)
    print(output_name)

    assert np.min(pathlen_array) > 0

    # save collated results to file
    assert "raw_data" not in out_folder, f"Cannot save to path {out_folder} which includes raw_data"
    output_path = os.path.join(out_folder, output_name)
    with open(output_path + ".npy", "wb") as f:
        np.save(f, obs_count)
        np.save(f, visit_count)
        np.save(f, success_array)
        np.save(f, pathlen_array)
        np.save(f, tot_rew_array)
        np.save(f, R_array)
        np.save(f, tot_obs_array)
    print(np.mean(R_array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="../results/raw_data/gamma=0.9/POMCP/30000", required=False, help="Root folder of results to process")
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
        print(out_folder)

        if out_folder not in visited_folders:
            visited_folders.append(out_folder)
            if os.path.exists(out_folder):
                assert 'raw_data' not in out_folder, f"Error: cannot delete subfolder {out_path} in raw_data."
                sh.rmtree(out_folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # process files
        collate_files(root, out_folder)