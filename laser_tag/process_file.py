import os
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--n_lasers",
                    required=True,
                    type=int,
                    )
parser.add_argument("--method",
                    required=True,
                    type=str,
                    )
parser.add_argument("--gamma",
                    required=True,
                    type=float,
                    )
parser.add_argument("--n_vo",
                    required=False,
                    type=int,
                    )
parser.add_argument("--folder",
                    required=False,
                    type=str,
                    default="rerun_results"
                    )
args, _ = parser.parse_known_args()

file_folder = args.folder
method = args.method

if args.method != "vomcp":
    glob_pattern = os.path.join(file_folder, args.method, f"n_lasers={args.n_lasers}", f"{args.method}_LaserTagEnv_n_laser={args.n_lasers}_0_*.csv")
else:
    glob_pattern = os.path.join(file_folder, args.method, f"n_lasers={args.n_lasers}", f"{args.method}_LaserTagEnv_n_laser={args.n_lasers}_n_vo={args.n_vo}_0_*.csv")
file_list = [file for file in glob.glob(glob_pattern)]
print(glob_pattern)

n_trials = len(file_list)
reward_array = np.zeros(n_trials)
success=0

gamma = args.gamma

rew_idx = 1
obs_idx = 2
curr_rew = 0
tag_rew = 10
for ii, filename in enumerate(file_list):
    with open(filename, "r") as f:
        factor = 1
        for kk, line in enumerate(f):
            if kk == 0:
                continue
            entry = line.split(";")
            if len(entry) < 2:
                continue
            curr_rew = int(entry[rew_idx])
            reward_array[ii-1] += curr_rew*factor
            factor *= gamma
        if curr_rew == tag_rew:
            success += 1

print("")
print("{} with {} lasers:".format(args.method, args.n_lasers))
print("n trials: ", n_trials)
print("")

if args.method != "vomcp":
    savefile = "{}_LaserTagEnv_n_laser={}_gamma={}".format(args.method, args.n_lasers, gamma)
else:
    savefile = f"{args.method}_n_vo={args.n_vo}_LaserTagEnv_n_laser={args.n_lasers}_gamma={gamma}"

save_folder = "aggregate_results"
save_path = os.path.join(file_folder, save_folder, savefile)

with open(save_path + ".npy", "wb") as f:
    np.save(f, reward_array)
