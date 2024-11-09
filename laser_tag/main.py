##
## Main code: template to run a single (default) experiment on AdLeap-MAS
##

import argparse
import numpy as np
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scenario_id", required=False, default=0, type=int, help='define your scenario configuration (check the available configuration in our GitHub)')
parser.add_argument("--method", required=False, default='vomcp', type=str)
parser.add_argument("--seed", required=False, default=-1, type=int)
parser.add_argument("--render", required=False, action='store_true')
parser.add_argument("--n_lasers", required=False, default=8, type=int)
parser.add_argument("--n_samples", required=False, default=10000, type=int,
                    help='If fixed_filter is false, this sets number of iterations during rejection sampling of belief update step. Otherwise, this sets filter size.')
parser.add_argument("--n_rollouts", required=False, default=0, type=int)
parser.add_argument("--fixed_filter", required=False, action='store_true', help='If true, n_samples specifies a fixed filter size.')
parser.add_argument("--n_vo", required=False, default=1, type=int, help="number of virtual observations to sample. only relevant for vomcp.")
parser.add_argument("--pval", required=False, default=0.20, type=float, help="p-value threshold for t-testing")
args, _ = parser.parse_known_args()


# 1. Initialisations
# a. setting random seed
if args.seed >= 0:
    np.random.seed(args.seed)
    random.seed(args.seed)

# b. setting default arguments where appropriate
if args.n_rollouts == 0:
    args.n_rollouts = args.n_samples

# 2. Setting the environment
method = args.method                             # choose your method
kwargs = {'n_vo': args.n_vo, 'particle_revigoration': False, 'fixed_filter': args.fixed_filter, 'pval': args.pval}       # define your additional hyperparameters to it (optional)
env_name = "LaserTagEnv"
scenario_id = args.scenario_id                   # define your scenario configuration (check the available configuration in our GitHub)

display = args.render                            # choosing to turn on or off the display

# 3. Creating the environment
# a. importing necessary modules
from importlib import import_module
from src.log import LogFile
import time

# b. creating the environment
env_module = import_module('src.envs.'+env_name)
load_default_scenario_method = getattr(env_module, 'load_default_scenario')
env, scenario_id = load_default_scenario_method(method,scenario_id,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()

exp_num = args.seed
header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']

# path = "results"
path = os.path.join(f"/users/rliu70/scratch/clean_results/lasertag/", args.method)

if args.method == 'vomcp':
    log = LogFile(env_name + f"_n_laser={args.n_lasers}_n_vo={args.n_vo}_n_rollouts={args.n_rollouts}_", scenario_id, method, exp_num, path, header)
else:
    log = LogFile(env_name + f"_n_laser={args.n_lasers}_n_rollouts={args.n_rollouts}_", scenario_id, method, exp_num, path, header)
adhoc_agent = env.get_robot()


MAX_EPISODES = 200
done = False

obs_count = 0
while not done and env.episode < MAX_EPISODES:
    # 1. Importing agent method
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    adhoc_agent.next_action = method(state, adhoc_agent, max_it=args.n_rollouts, k=args.n_samples, **kwargs)
    end = time.time()

    if adhoc_agent.next_action not in env.get_motion_list():
        obs_count += 1
        print("count: ", obs_count)

    # 3. Taking a step in the environment
    next_state,reward,done,info = env.step(adhoc_agent.next_action)

    # # 4. For learning methods
    # if 'dqn_model' in adhoc_agent.smart_parameters.keys():
    #     adhoc_agent.smart_parameters['dqn_model'].add_memory(state,adhoc_agent.next_action,next_state,reward,done)

    opp_pos = next_state.get_opponent().position
    print("Step: ", env.episode)
    print("Robot position: ", adhoc_agent.position)
    print("Action: ", adhoc_agent.next_action)
    print("Reward: ", reward)
    print("Observations: ", next_state.get_observation())
    print("Opponent position: ", opp_pos)
    print("Robot opponent position: ", next_state.get_robot().opponent_pos)
    print("")

    # 5. Logging the Data
    data = {'it':env.episode,
            'reward':reward,
            'observe': env.action_dict[adhoc_agent.next_action] == "Observe",
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations']}
    log.write(data)
    
    # 6. Updating the state
    state = next_state

env.close()
###
# THE END - That's all folks :)
###