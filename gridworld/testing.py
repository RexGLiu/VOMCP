#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from PO_gridworld import GridWorldEnv
from planners.vomcp import VOMCP
from planners.base_planner import int_type
from planners.node import ONode, ANode

seed=0
np.random.seed(seed)
random.seed(seed)

n_particles = 1000
max_samples = 10000

params = dict()
params['gamma'] = 0.95
params['n_vobs'] = 1
params['n_particles'] = n_particles
params['c'] = 100
params['pval'] = 0.01
params['max_depth'] = 100
params['n_samples'] = max_samples
params['n_rollouts'] = max_samples
params['fixed_filter'] = True
params['max_steps'] = 150
params['split'] = None

'''Create the environment'''
env_params = {"gamma": 0.99,
    "p_obs": 0.75,
    "p_fail": 0.3,
    "obs_width": 1,
}
env = GridWorldEnv(env_params, desc='6x6', survival_rew=False)

algo = VOMCP(env, params)
print("Uniform prior")
print(algo.root.belief)
assert np.array_equal(algo.root.belief, np.ones(env.size, int_type))


def test_belief_updating():
    belief = np.ones(env.size)
    print("Move left (into wall), observe start")
    new_belief = algo.belief_update(belief, action=0, obs=0)
    print(new_belief)
    print(np.nonzero(new_belief))

    print("Move down, observe 15")
    new_belief = algo.belief_update(belief, action=1, obs=15)
    print(new_belief)
    print(np.nonzero(new_belief))


    belief = np.zeros(env.size)
    belief[10] = 1
    print("Delta func prior")
    print(belief)

    print("Impossible history: move down, observe 7 -- belief should reset")
    new_belief = algo.belief_update(belief, action=1, obs=7)
    assert np.array_equal(new_belief, np.ones(env.size, int_type))
    print(new_belief)

    print("Delta update: move left from 10 but observe 11")
    new_belief = algo.belief_update(belief, action=0, obs=11)
    print(new_belief)
    print(np.nonzero(new_belief)[0])

    print("move down from 10 but observe 11")
    new_belief = algo.belief_update(belief, action=1, obs=11)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([10,16])
    print(states)
    assert np.array_equal(states, gt)

    print("move up from 10 but observe 11")
    new_belief = algo.belief_update(belief, action=3, obs=11)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([4, 10])
    print(states)
    assert np.array_equal(states, gt)

    print("move left from 10, observe 10")
    new_belief = algo.belief_update(belief, action=0, obs=10)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([9, 10])
    print(states)
    assert np.array_equal(states, gt)

    print("move down from 10, observe 10")
    new_belief = algo.belief_update(belief, action=1, obs=10)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([10,16])
    print(states)
    assert np.array_equal(states, gt)

    print("move right from 10, observe 10")
    new_belief = algo.belief_update(belief, action=2, obs=10)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([10, 11])
    print(states)
    assert np.array_equal(states, gt)

    print("move up from 10, observe 10")
    new_belief = algo.belief_update(belief, action=3, obs=10)
    print(new_belief)
    states, gt = np.nonzero(new_belief)[0], np.array([4, 10])
    print(states)
    assert np.array_equal(states, gt)

def test_sample_vobs():
    print('Sample from uniform prior')
    belief = np.ones(env.size)
    obs = algo.sample_vobs(belief)
    print(obs)

    print('Sample from delta on state 10')
    belief = np.zeros(env.size)
    belief[10] = 1
    obs = algo.sample_vobs(belief)
    print(obs)
    obs_list = sorted(obs.keys())
    print(obs_list)
    assert obs_list == [3,4,5,9,10,11,15,16,17]

def test_select_action():
    algo.root.children = [ANode(history=[], obs_space=algo.n_states) for a in range(algo.n_actions)]

    algo.root.children[0].V = 14.027692354776525
    algo.root.children[0].V_sqr = 12676.317818212457
    algo.root.children[0].N = 775

    algo.root.children[1].V = 51.1382078803747
    algo.root.children[1].V_sqr = 47369.480581742144
    algo.root.children[1].N = 1853

    algo.root.children[2].V = 47.61414272941671
    algo.root.children[2].V_sqr = 44959.33798348402
    algo.root.children[2].N = 1646

    algo.root.children[3].V = 10.629876243108383
    algo.root.children[3].V_sqr = 9220.307865808987
    algo.root.children[3].N = 726

    algo.vobs_dict = {0 : 1}
    algo.virtual_trees = {0 : ONode([], algo.n_actions)}
    algo.virtual_trees[0].children = [ANode(history=[], obs_space=algo.n_states) for a in range(algo.n_actions)]
    asdf = algo.virtual_trees[0]

    asdf.children[0].V = 60.96665825892856
    asdf.children[0].V_sqr = 53616.19313067179
    asdf.children[0].N = 28

    asdf.children[1].V = 573.9516239609042
    asdf.children[1].V_sqr = 560931.657782182
    asdf.children[1].N = 4725

    asdf.children[2].V = 409.66495109432503
    asdf.children[2].V_sqr = 377318.9502632353
    asdf.children[2].N = 203

    asdf.children[3].V = 170.95208749439362
    asdf.children[3].V_sqr = 145299.90585660163
    asdf.children[3].N = 44

    asdf = algo.select_action()
    print(asdf)



if __name__ == "__main__":
    # test_belief_updating()
    # test_sample_vobs()
    test_select_action()
