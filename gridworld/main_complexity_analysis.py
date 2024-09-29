#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import numpy as np
import random
import argparse
import pickle
from PO_gridworld import GridWorldEnv

parser = argparse.ArgumentParser()
parser.add_argument("--save_name", required=False, default="", type=str, help='Save filename')
parser.add_argument("--save_folder", required=False, default=".", type=str, help='Location to save results')
parser.add_argument("--history_file", required=False, default="", type=str, help='Location of saved histories file to load')
parser.add_argument("--trial_idx", required=False, default=None, type=int, help='Which trial in history file to use. Default is to loop through all.')
parser.add_argument("--env", required=False, default="6x6", type=str, choices=["6x6", "6x6_easy", "6x6_hard", "shortcut"], help='Environment to run')
parser.add_argument("--mode", required=False, default="VOMCP", type=str, choices=["VOMCP", "POMCP", "EA", "EAF", "EAF_modified", "VOMCP_avg", "VOMCP_unequal"], help='Algorithm to run')
parser.add_argument("--seed", required=False, default=0, type=int, help='Random seed')
parser.add_argument("--gamma", required=False, default=0.95, type=float, help='Discount factor')
parser.add_argument("--p_obs", required=False, default=0.75, type=float, help="Probability of observing true state")
parser.add_argument("--p_fail", required=False, default=0.3, type=float, help="Probability of transition failure")
parser.add_argument("--obs_width", required=False, default=1, type=int, help="Observations to be drawn with (x+/-obs_width, y+/-obs_width) of agent's location (x,y)")
parser.add_argument("--survival_rew", required=False, action='store_true', help="Switches from hole penalties (default) to survival rewards to encourage avoiding holes")
parser.add_argument("--n_vobs", required=False, default=4, type=int, help="Number of virtual observations to sample. Only relevant for VOMCP.")
parser.add_argument("--n_samples", required=False, default=20000, type=int, help="By default, this sets the total rejection sampling budget (divided across trees in VOMCP) at each belief update. But if fixed_filter is set, this sets a fixed total particle budget instead.")
parser.add_argument("--fixed_filter", action='store_true', help="If set, particle filters sizes rather than number of rejection sample steps gets fixed in belief updates. See also n_samples.")
parser.add_argument("--n_rollouts", required=False, default=0, type=int, help="Total rollout budget at each planning step. Defaults to n_samples.")
parser.add_argument("--c", required=False, default=1000, type=float, help="Tree search exploration hyperparameter")
parser.add_argument("--pval", required=False, default=0.01, type=float, help="p-value threshold for t-testing")
parser.add_argument("--max_depth", required=False, default=100, type=int, help="Tree search max depth")
parser.add_argument("--max_steps", required=False, default=150, type=int, help="Cap on steps in an episode")
parser.add_argument('--split', required=False, default=None, type=float, help="Fraction of total budget to allocate to virtual search in VOMCP. Allocation will be split equally amongst the virtual trees. Default is to equally partition amongst all main and virtual trees.")
args, _ = parser.parse_known_args()

if args.n_rollouts == 0:
    args.n_rollouts = args.n_samples

'''Save run parameters for each seed to a run_parameter subfolder'''
run_params = {"env": args.env,
    "survival_reward": args.survival_rew,
    "mode": args.mode,
    "n_vobs": args.n_vobs,
    "seed": args.seed,
    "gamma": args.gamma,
    "p_obs": args.p_obs,
    "p_fail": args.p_fail,
    "obs_width": args.obs_width,
    "n_samples": args.n_samples,
    "fixed_filter": args.fixed_filter,
    "n_rollouts": args.n_rollouts,
    "c": args.c,
    "pval": args.pval,
    "max_depth": args.max_depth,
    "max_steps": args.max_steps,
    "split": args.split
}

'''Set correct planning algorithm'''
planner_module_name = (args.mode.split('_')[0]).lower()
planner_module = import_module('planners.' + planner_module_name)
if args.mode == "VOMCP_unequal":
    assert args.split is not None, "Particle split must be set for VOMCP_unequal"
    Algo = getattr(planner_module, "VOMCP")
else:
    Algo = getattr(planner_module, args.mode)

'''Set default save filename if non specified.'''
if args.save_name != "":
    save_name = args.save_name
else:
    if args.mode in ["VOMCP", "VOMCP_avg"]:  # VOMCP_avg performs EAF_style averaging
        save_name = f"complexity_analysis_{args.mode}_-_{args.nruns}_runs,_env={args.env}_{args.n_vobs}_VO,_nsamples={args.n_samples},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

    elif args.mode == "VOMCP_unequal":  # unequal split between main tree and virtual ones
        save_name = f"complexity_analysis_{args.mode}_-_{args.nruns}_runs,_env={args.env}_{args.n_vobs}_VO,_nsamples={args.n_samples},_split={args.split},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

    elif args.mode == "POMCP":
        save_name = f"complexity_analysis_{args.mode}_-_{args.nruns}_runs,_env={args.env},_nsamples={args.n_samples},_c={args.c},_seed={args.seed}"

    elif args.mode in ["EA", "EAF", "EAF_modified"]:
        save_name = f"complexity_analysis_{args.mode}_-_{args.nruns}_runs,_env={args.env},_nsamples={args.n_samples},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

'''Create save folder for results if not already existing'''
os.makedirs(args.save_folder, exist_ok=True)

save_path = os.path.join(args.save_folder, save_name)

'''Save run parameters to file'''
param_file = f"analysis_params.pkl"
param_folder = args.save_folder
os.makedirs(param_folder, exist_ok=True)
param_path = os.path.join(param_folder, param_file)

with open(param_path, 'w') as f:
    for name, value in sorted(run_params.items()):
        f.write(f"{name}: {value}\n")

'''Set random seeds'''
np.random.seed(args.seed)
seed = np.random.randint(np.iinfo(np.int32).max)

np.random.seed(seed)
random.seed(seed)

'''Create the environment'''
env_params = {"gamma": args.gamma,
    "p_obs": args.p_obs,
    "p_fail": args.p_fail,
    "obs_width": args.obs_width,
}
env = GridWorldEnv(env_params, desc=args.env, survival_rew=args.survival_rew)

'''Set up the algorithm to run'''
algorithm = Algo(env, run_params)

# load history to run
with open(args.history_file, "rb") as f:
    nruns = pickle.load(f)
    history_list = pickle.load(f)

tree_depths, rollout_depths = [], []
if args.trial_idx is None:
    for history in history_list:
        results = algorithm.run_history(history)
        tree_depths += [results['tree depths']]
        rollout_depths += [results['rollout depths']]
else:
    history = history_list[args.trial_idx]
    results = algorithm.run_history(history)
    tree_depths += [results['tree depths']]
    rollout_depths += [results['rollout depths']]

with open(save_path + ".pkl", "wb") as f:
    pickle.dump(tree_depths, f)
    pickle.dump(rollout_depths, f)
