from .eaf import EAF
import random
import numpy as np
from scipy.stats import t as t_distr

def eafm_planning(env, agent, max_it=5000, k=5000, max_depth=20, **kwargs):
    '''planning function for eaf_modified'''
    t_test_pval = kwargs['pval']

    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. EAF_modified planning
    # - initialising/getting the planning algorithm
    eaf_modified = EAF(max_depth, max_it, k, kwargs) if 'eaf_modified' not \
     in agent.smart_parameters else agent.smart_parameters['eaf_modified']

    # - planning
    _, _, _, _, info = eaf_modified.planning(copy_env,agent)

    # 3. get best real action
    best_action, bestQ, bestQ_sqr, N = eaf_modified.root.get_best_real_action()
    main_results = (bestQ, bestQ_sqr, N)

    # 4. get stats for 'observe'
    obs_branch_stats = get_obs_branch_stats(eaf_modified.root)

    # 5. compare search results
    gamma = eaf_modified.discount_factor

    # series of steps to retrieve obs reward from env
    particles = eaf_modified.root.particle_filter
    dummy_state = random.choice(particles).copy()
    obs_action = dummy_state.get_obs_action()
    _, obs_reward, _, _ = dummy_state.step(obs_action)

    # compute t-stat for comparison
    t_stat, dof = get_t_stat(main_results, obs_branch_stats, obs_reward, gamma)

    # 6. perform t-test to determine next action
    if t_distr.cdf(t_stat, dof) > 1 - t_test_pval:
        next_action = obs_action
    else:
        next_action = best_action

    # 5. Updating the search tree
    agent.smart_parameters['eaf_modified'] = eaf_modified
    agent.smart_parameters['count'] = info
    return next_action

def get_obs_branch_stats(node):
    # get obs branch
    obs_action = node.state.get_obs_action()
    obs_branch = node.children[obs_action]

    # iterate through observations in each obs branch
    obs_branch_stats = []
    for onode in obs_branch.children:
        _, bestQ, bestQ_sqr, N = onode.get_best_real_action()
        obs_branch_stats += [(bestQ, bestQ_sqr, N)]

    return obs_branch_stats


def get_t_stat(main_results, obs_branch_stats, obs_reward, gamma):
    # Performs Welch's t-test to see if observation Q-value significantly greater than best action Q-value

    Q_main, Q_sqr_main, N_main = main_results

    # with just one datapoint, estimate of real action value unreliable, so default to conservative option of getting more data
    if N_main <= 1:
        print("Real action value unreliable, defaulting to observe")
        return 100000, 1

    std_err1 = (Q_sqr_main - Q_main**2) / (N_main-1)
    dof_denom1 = std_err1**2 / (N_main-1)

    Q_obs = 0
    std_err2 = 0
    dof_denom2 = 0
    N_obs = 0

    # loop through statistics for each history following "observe"
    for obs_stat in obs_branch_stats:
        Q_, Q_sqr_, N_ = obs_stat

        if N_ <= 1:  # only consider histories with at least two rollouts to get non-singular measure of standard errors
            print(f"Warning: observation branch visited only {N_} times. Skipping observation.")
            continue

        # multiply by discount factors for taking extra action
        Q_ *= gamma
        Q_sqr_ *= gamma**2

        N_obs += N_

        # compute statistics for Q-value of best action
        Q_obs += N_ * Q_
        std_err_ = N_**2 * (Q_sqr_ - Q_**2) / (N_-1)
        std_err2 += std_err_
        dof_denom2 += std_err_**2 / (N_-1)

    # if "observe" branch under-explored, we'll force agent to take real action
    if N_obs == 0:
        print("Observe value unreliable, defaulting to real action")
        return -100000, 1

    # normalise by total number of rollouts used in the averages
    Q_obs /= N_obs
    std_err2 /= N_obs**2
    dof_denom2 /= N_obs**4

    # apply cost of observation if any
    Q_obs += obs_reward

    # if both std_err are 0, default to picking action with largest value
    if std_err1 == std_err2 == 0:
        if Q_main > Q_obs:
            return -1000, 1
        else:
            return 1000, 1

    # compute t-statistic to compare the two values
    t_stat = (Q_obs - Q_main) / np.sqrt(std_err1 + std_err2)
    dof = (std_err1 + std_err2)**2 / (dof_denom1 + dof_denom2)

    return t_stat, dof
