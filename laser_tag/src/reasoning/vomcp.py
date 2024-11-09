from src.reasoning.node import ONode, find_new_PO_root, \
    particle_revigoration, uniform_prior
import random
from .pomcp import POMCP, obs_based_particle_update, is_terminal
import numpy as np
from scipy.stats import t as t_distr

class VOMCP(POMCP):
    def planning(self, state, agent):
        # 1. Perform a particle revigoration to guard against particle depletion
        if self.pr:
            particle_revigoration(state,agent,self.root,self.k)

        # if not reinvigorating particles, default to uniform prior should we experience collapse of posterior
        elif len(self.root.particle_filter) == 0:
            uniform_prior(state,agent,self.root,self.k)

        # 2. Searching for the best action within the tree
        best_action, bestQ, bestQ_sqr, N = self.search(self.root, agent)

        # 3. Returning the best action
        self.root.show_qtable()
        info = {'nrollouts': self.rollout_count,
                'nsimulations': self.simulation_count}
        return best_action, bestQ, bestQ_sqr, N, info

    def virtual_planning(self, state, agent, obs, particles):
        # 1. Defining the root of our search tree
        # via initialising the tree
        self.root = ONode(observation=obs, state=state, depth=0, parent=None)
        self.root.particle_filter = particles

        # 2. Perform a particle revigoration to guard against particle depletion
        if self.pr:
            particle_revigoration(state,agent,self.root,self.k)

        # if not reinvigorating particles, default to uniform prior should we experience collapse of posterior
        elif len(self.root.particle_filter) == 0:
            uniform_prior(state,agent,self.root,self.k)

        # if not reinvigorating particles, default to uniform prior should we experience collapse of posterior
        elif len(self.root.particle_filter) == 0:
            uniform_prior(state,agent,self.root,self.k)

        # 3. Searching for the best action within the tree
        best_action, bestQ, bestQ_sqr, N = self.search(self.root, agent)

        # 4. Returning the best action
        self.root.show_qtable()
        info = {'nrollouts': self.rollout_count,
                'nsimulations': self.simulation_count}
        return bestQ, bestQ_sqr, N, info

    def get_particles(self):
        return self.root.particle_filter

def sample_n_obs(state, agent, particles, n_obs):
    observations = set()
    obs_counts = {}
    obs_reward = None

    for _ in range(n_obs):
        # 1. Generate a virtual state from current state
        if particles == None or particles == []:
            virtual_state = state.sample_state(agent)
        else:
            virtual_state = random.choice(particles).copy()
        virtual_state.simulation = True

        # 2. Select observe action
        obs_action = virtual_state.get_obs_action()

        # 3. Take virtual obs
        virtual_state, obs_reward, done, info = virtual_state.step(obs_action)
        new_obs = virtual_state.get_observation()

        if new_obs not in observations:
            observations.add(new_obs)
            obs_counts[new_obs] = 1
        else:
            obs_counts[new_obs] += 1

    return observations, obs_counts, obs_reward  # assume constant deterministic observation penalty

def get_curr_vomcp(state, agent, max_depth, max_k, n_obs, **kwargs):
    '''Retrieves a planner for the current history and sets the root node of the search tree.'''

    # 0. Get previous set of particles for possible belief update. Compute number of filter iterations for main tree.
    old_particles = agent.smart_parameters['vomcp'].get_particles() if 'vomcp' in agent.smart_parameters else None
    filter_size = max_k // (n_obs + 1) if 'vomcp' in agent.smart_parameters else None
        # Note: particle budget per tree should be given by k // n_trees. But we do not know n_trees in advance,
        # as this requires sampling obs from the root belief distribution. So we use n_obs+1 in place of n_trees here.
    fixed_filter = kwargs['fixed_filter']

    # 1. Getting the current state and previous action-observation pair
    previous_action = agent.next_action
    current_observation = state.get_observation()

    # 2. Setup planner for new history.
    # First check if last action was observe. If so, check if virtual observation matches last real one. If so, retrieve virtual planner.
    if previous_action not in state.get_motion_list() + [None]:  # previous action was "observe" (at start of ep agent's prev action is None)
        matching_obs_found = False
        for vomcp_virtual in agent.smart_parameters['vomcp_virtual_dict'].values():
            virtual_obs = vomcp_virtual.root.observation
            if state.compare_obs(virtual_obs, current_observation):
                vomcp = vomcp_virtual
                matching_obs_found = True
                break
        if not matching_obs_found:
            vomcp = VOMCP(max_depth, None, filter_size, kwargs)

    else:  # previous action was not "observe"
        vomcp = VOMCP(max_depth, None, filter_size, kwargs) if 'vomcp' not \
                in agent.smart_parameters else agent.smart_parameters['vomcp']

    # 2. Set root of agent's search tree by advancing within the existing tree
    if previous_action in state.get_motion_list():
        vomcp.root = find_new_PO_root(state, previous_action, \
                                      current_observation, agent, vomcp.root, adversary=vomcp.adversary)
    # or initialising a new tree
    elif vomcp.root is None:
        vomcp.root = ONode(observation=None, state=state, depth=0, parent=None)

    # 3. Update beliefs based on last action and resulting obs
    if old_particles is not None:
        assert len(old_particles) > 0, "old_particles is empty"
        assert filter_size is not None, "number of particles to include in particle filter is invalid"
        new_particles = obs_based_particle_update(previous_action, current_observation, filter_size, fixed_filter, old_particles=old_particles)

        # Set particle_filter to new particles if no pre-existing set of particles or if particles have collapsed onto a terminal state
        if vomcp.root.particle_filter == [] or is_terminal(vomcp.root.particle_filter[0]):
            vomcp.root.particle_filter = new_particles
        else: # else just add to current set of particles
            vomcp.root.particle_filter += new_particles

    return vomcp

def multi_vomcp_planning(state, agent, observations, curr_particles, max_depth, iters, k, kwargs):
    virtual_vomcp_dict = {}
    virtual_count_dict = {}
    virtual_Q_results = {}
    obs_action = state.get_obs_action()
    fixed_filter = kwargs['fixed_filter']

    # iterate through each virtual obs and perform planning
    for ii, obs in enumerate(observations):
        vomcp = VOMCP(max_depth, iters, k, kwargs)
        if curr_particles is None or curr_particles == []:
            new_particles = obs_based_particle_update(obs_action, obs, k, fixed_filter, env=state, agent=agent)
        else:
            new_particles = obs_based_particle_update(obs_action, obs, k, fixed_filter, curr_particles)
        bestQ, bestQ_sqr, N, info = vomcp.virtual_planning(state, agent, obs, new_particles)
        virtual_Q_results[obs] = (bestQ, bestQ_sqr, N)

        virtual_vomcp_dict[obs] = vomcp
        virtual_count_dict[obs] = info

    return virtual_vomcp_dict, virtual_count_dict, virtual_Q_results

def get_t_stat(main_results, virtual_results, observations, obs_count, obs_reward, gamma):
    main_Q, main_Q_sqr, main_N = main_results
    virtual_vomcp_dict, virtual_count_dict, virtual_Q_results = virtual_results
    n_obs = sum(obs_count.values())

    # 1. Compute standard error and contribution to dof for main Q-value
    std_err1 = (main_Q_sqr - main_Q ** 2) / (main_N - 1)
    dof_denom = std_err1**2 / (main_N - 1)

    # 2. Loop through virtual observations and compute contribution to second standard error and dof
    std_err2 = 0
    virtual_Q = 0
    for obs in observations:
        virtual_Q_o, virtual_Q_o_sqr, virtual_N_o = virtual_Q_results[obs]
        weight = obs_count[obs]/n_obs
        virtual_Q += weight * (obs_reward + gamma * virtual_Q_o)
        std_err = weight**2 * gamma**2 * (virtual_Q_o_sqr - virtual_Q_o ** 2) / (virtual_N_o - 1)
        std_err2 += std_err
        dof_denom += std_err**2 / (virtual_N_o - 1)

    if std_err1 > 0 and std_err2 > 0:
        t_stat = (virtual_Q - main_Q) / np.sqrt(std_err1 + std_err2)
        dof = (std_err1 + std_err2)**2 / dof_denom
    else: # default to not observing if standard errors are too small
        t_stat = None
        dof = None

    return t_stat, dof


def vomcp_planning(env, agent, max_it=5000, max_k=5000, max_depth=20, **kwargs):
    n_obs = kwargs['n_vo']
    t_test_pval = kwargs['pval']

    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. Retrieve planner for current history
    vomcp = get_curr_vomcp(copy_env, agent, max_depth, max_k, n_obs, **kwargs)
    curr_particles = vomcp.get_particles()

    # 3. Sample n_obs virtual observations and compute computational budget per tree
    observations, obs_count, obs_reward = sample_n_obs(copy_env, agent, curr_particles, n_obs)

    # 4. Split budget equally over total number of trees to run
    n_trees = len(observations) + 1
    n_iters = max_it//n_trees
    vomcp.reset_max_it(n_iters)

    k = max_k//n_trees

    # 5. POMCP Planning
    action_proposal, bestQ, bestQ_sqr, N, info = vomcp.planning(copy_env, agent)
    main_results = (bestQ, bestQ_sqr, N)

    # 6. Copy env for virtual obs planning
    copy_env = env.copy()
    copy_env.simulation = True

    # 7. Run search for each unique observation
    virtual_results = multi_vomcp_planning(copy_env, agent, observations, curr_particles, max_depth, n_iters, k, kwargs)

    # 8. Compare search results
    gamma = vomcp.discount_factor
    t_stat, dof = get_t_stat(main_results, virtual_results, observations, obs_count, obs_reward, gamma)

    next_action = action_proposal
    if t_stat is not None and t_distr.cdf(t_stat, dof) > 1 - t_test_pval:
        next_action = copy_env.get_obs_action()

    # 8. Updating the search tree
    virtual_vomcp_dict, virtual_count_dict, _ = virtual_results
    agent.smart_parameters['vomcp'] = vomcp
    agent.smart_parameters['count'] = info
    agent.smart_parameters['vomcp_virtual_dict'] = virtual_vomcp_dict
    agent.smart_parameters['virtual_count_dict'] = virtual_count_dict
    return next_action