#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from PO_gridworld import GridWorldEnv
from planners.vomcp import VOMCP, VOMCP_avg
from planners.pomcp import POMCP
from planners.ea import EA
from planners.eaf import EAF

seed=0
np.random.seed(seed)
random.seed(seed)

n_particles = 5000
max_samples = 50000

params = dict()
params['gamma'] = 0.95
params['n_vobs'] = 10
params['n_particles'] = n_particles
params['c'] = 100
params['pval'] = 0.01
params['max_depth'] = 100
params['max_samples'] = max_samples
params['max_steps'] = 5
params['reinvigorate'] = False
params['split'] = 0.1
params['nruns'] = 5

'''Create the environment'''
env_params = {"gamma": 0.95,
    "p_obs": 0.75,
    "p_fail": 0.3,
    "obs_width": 1,
}
env = GridWorldEnv(env_params, desc='6x6', survival_rew=False)


if __name__ == "__main__":
    algo = POMCP(env, params)
    run_results = algo.run_episode()
    print(run_results, '\n')

    algo = EA(env, params)
    run_results = algo.run_episode()
    print(run_results, '\n')

    algo = EAF(env, params)
    run_results = algo.run_episode()
    print(run_results, '\n')

    algo = VOMCP(env, params)
    run_results = algo.run_episode()
    print(run_results, '\n')

    algo = VOMCP_avg(env, params)
    run_results = algo.run_episode()
    print(run_results, '\n')

