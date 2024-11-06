#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import random
import argparse
import pickle
from PO_gridworld import GridWorldEnv
from importlib import import_module
from planners.base_planner import BeliefException

parser = argparse.ArgumentParser()
parser.add_argument("--save_name", required=False, default="", type=str, help='Save filename')
parser.add_argument("--save_folder", required=False, default=".", type=str, help='Location to save results')
parser.add_argument("--env", required=False, default="6x6", type=str, choices=["6x6", "6x6_easy", "6x6_hard", "shortcut"], help='Environment to run')
parser.add_argument("--mode", required=False, default="VOMCP", type=str, choices=["VOMCP", "POMCP", "EA", "EAF", "EAF_modified", "VOMCP_avg", "VOMCP_unequal"], help='Algorithm to run')
parser.add_argument("--nruns", required=False, default=10, type=int, help='Number of trials to run')
parser.add_argument("--seed", required=False, default=0, type=int, help='Random seed')
parser.add_argument("--gamma", required=False, default=0.95, type=float, help='Discount factor')
parser.add_argument("--p_obs", required=False, default=0.75, type=float, help="Probability of observing true state")
parser.add_argument("--p_fail", required=False, default=0.3, type=float, help="Probability of transition failure")
parser.add_argument("--obs_width", required=False, default=1, type=int, help="Observations to be drawn with (x+/-obs_width, y+/-obs_width) of agent's location (x,y)")
parser.add_argument("--survival_rew", required=False, action='store_true', help="Switches from hole penalties (default) to survival rewards to encourage avoiding holes")
parser.add_argument("--n_vobs", required=False, default=4, type=int, help="Number of virtual observations to sample. Only relevant for VOMCP.")
parser.add_argument("--n_samples", required=False, default=0, type=int, help="By default, this sets the total rejection sampling budget (divided across trees in VOMCP) at each belief update. But if fixed_filter is set, this sets a fixed total particle budget instead. Value defaults to match n_rollouts if not set.")
parser.add_argument("--fixed_filter", action='store_true', help="If set, particle filters sizes rather than number of rejection sample steps gets fixed in belief updates. See also n_samples.")
parser.add_argument("--n_rollouts", required=False, default=20000, type=int, help="Total rollout budget at each planning step.")
parser.add_argument("--c", required=False, default=1000, type=float, help="Tree search exploration hyperparameter")
parser.add_argument("--pval", required=False, default=0.20, type=float, help="p-value threshold for t-testing")
parser.add_argument("--max_depth", required=False, default=200, type=int, help="Tree search max depth")
parser.add_argument("--max_steps", required=False, default=150, type=int, help="Cap on steps in an episode")
parser.add_argument('--split', required=False, default=None, type=float, help="Fraction of total budget to allocate to virtual search in VOMCP. Allocation will be split equally amongst the virtual trees. Default is to equally partition amongst all main and virtual trees.")
parser.add_argument("--q_thresh", required=False, default=1E-15, type=float, help="When randomly choosing from amongst best actions, all actions with Q-values within q_thresh of maximum Q-value will be included.")
args, _ = parser.parse_known_args()

if args.n_samples == 0:
    args.n_samples = args.n_rollouts

'''Save run parameters for each seed to a run_parameter subfolder'''
run_params = {"nruns": args.nruns,
    "env": args.env,
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
    "split": args.split,
    "q_thresh": args.q_thresh,
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
        save_name = f"run_results_{args.mode}_-_{args.nruns}_runs,_env={args.env}_{args.n_vobs}_VO,_nsamples={args.n_samples},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

    elif args.mode == "VOMCP_unequal":  # unequal split between main tree and virtual ones
        save_name = f"run_results_{args.mode}_-_{args.nruns}_runs,_env={args.env}_{args.n_vobs}_VO,_nsamples={args.n_samples},_split={args.split},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

    elif args.mode == "POMCP":
        save_name = f"run_results_{args.mode}_-_{args.nruns}_runs,_env={args.env},_nsamples={args.n_samples},_c={args.c},_seed={args.seed}"

    elif args.mode in ["EA", "EAF", "EAF_modified"]:
        save_name = f"run_results_{args.mode}_-_{args.nruns}_runs,_env={args.env},_nsamples={args.n_samples},_pval_thresh={args.pval},_c={args.c},_seed={args.seed}"

'''Create save folder for results if not already existing'''
os.makedirs(args.save_folder, exist_ok=True)

save_path = os.path.join(args.save_folder, save_name)

'''Save run parameters to file'''
param_file = f"run_parameters_{args.seed}.pkl"
param_folder = os.path.join(args.save_folder, "run_parameters")
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

'''Arrays to store results'''
obs_count = np.zeros(env.map.size, dtype=np.uint64)      # number of extra obs taken in each state
visit_count = np.zeros(env.map.size, dtype=np.uint64)    # number of visits to each state

nruns = run_params['nruns']
success_array = np.zeros(nruns, dtype=int)
pathlen_array = np.zeros(nruns, dtype=np.uint64)
tot_rew_array = np.zeros(nruns, dtype=np.float_)
R_array = np.zeros(nruns, dtype=np.float_)
obs_count_array =  np.zeros(nruns, dtype=np.uint64)
history_list = [None]*nruns
states_traversed_list = [None]*nruns

N_obs_root = [None]*nruns
N_obs_branches = [None]*nruns
obs_branch_visits = [None]*nruns

if args.mode == "EAF_modified":
    decision_flag_counts = [None]*nruns

'''Loop through episodes and store results'''
for i in range(nruns):
    try:
        run_results = algorithm.run_episode()

    # if an invalid belief is encountered, dump belief to file for review later.
    except BeliefException as e:
        beliefs = e.belief
        distr = beliefs/float(np.sum(beliefs))
        print("Normalisation: ", float(np.sum(beliefs)))
        print("Distribution: ", distr)
        print("Beliefs: ", beliefs)

        with open(save_path + "_err.npy".format, "wb") as f:
            np.save(f, beliefs)
            np.save(f, np.sum(beliefs))
            np.save(f, distr)
        raise e

    # Print results of episode run
    for k,v in run_results.items():
        print(f"{k}: {v}")
    print('')

    success_array[i] = run_results['success']
    pathlen_array[i] = run_results['total steps']
    tot_rew_array[i] = run_results['total reward']
    R_array[i] = run_results['discounted return']
    obs_count_array[i] = run_results['observation count']
    history_list[i] = run_results['final history']
    states_visited_list = run_results['states visited']

    # track number of visits and observations for each state
    states_visited = run_results['states visited']
    history = run_results['final history']
    for idx, s in enumerate(states_visited[:-1]):
        visit_count[s] += 1
        obs = history[2*idx+1]
        assert obs[0] == 'o', f"Invalid observation format {obs}"
        obs = int(obs[1:])       # strip the 'o' prefix and convert to int
        obs_count[obs] += 1
    visit_count[states_visited[-1]] += 1

    # tree statistics
    if args.mode in ["EA", "EAF", "EAF_modified"]:
        N_obs_root[i] = run_results["N 'observe' at root"]
        N_obs_branches[i] = run_results["N branches at 'observe'"]
        obs_branch_visits[i] = run_results["'observe' branch visits"]

    if args.mode == "EAF_modified":
        decision_flag_counts[i] = run_results["decision flag counts"]

        # save result of each trial to temporary file
    with open(save_path + "_0.npy", "wb") as f:
        np.save(f, obs_count)
        np.save(f, visit_count)
        np.save(f, success_array)
        np.save(f, pathlen_array)
        np.save(f, tot_rew_array)
        np.save(f, R_array)
        np.save(f, obs_count_array)

    with open(save_path + "_0.pkl", "wb") as f:
        pickle.dump(i, f)
        pickle.dump(history_list, f)
        pickle.dump(states_traversed_list, f)

        if args.mode in ["EA", "EAF", "EAF_modified"]:
            pickle.dump(N_obs_root, f)
            pickle.dump(N_obs_branches, f)
            pickle.dump(obs_branch_visits, f)

        if args.mode == "EAF_modified":
            pickle.dump(decision_flag_counts, f)

    sys.stdout.flush()

    algorithm.reset()

# save complete results to file
with open(save_path + ".npy", "wb") as f:
    np.save(f, obs_count)
    np.save(f, visit_count)
    np.save(f, success_array)
    np.save(f, pathlen_array)
    np.save(f, tot_rew_array)
    np.save(f, R_array)
    np.save(f, obs_count_array)

with open(save_path + ".pkl", "wb") as f:
    pickle.dump(nruns, f)
    pickle.dump(history_list, f)
    pickle.dump(states_traversed_list, f)

    if args.mode in ["EA", "EAF", "EAF_modified"]:
        pickle.dump(N_obs_root, f)
        pickle.dump(N_obs_branches, f)
        pickle.dump(obs_branch_visits, f)

    if args.mode == "EAF_modified":
        pickle.dump(decision_flag_counts, f)

os.remove(save_path + "_0.npy")
os.remove(save_path + "_0.pkl")
