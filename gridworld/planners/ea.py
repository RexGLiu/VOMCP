import numpy as np
from scipy.stats import t as t_distr
from .node import ONode, int_type
from .base_planner import BasePlanner

class EA(BasePlanner):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.real_action_space = self.action_space[:-1]
        self.pval = params['pval']

    def run_episode(self):
        log = {}
        curr_step = self.env.reset()

        tot_reward = 0.0
        R = 0.0
        discount = 1
        obs_count = 0

        states_visited = [self.env.state]
        N_observe = []
        N_branch_obs = []
        obs_branch_visits = []

        n_steps = 0
        while (curr_step.done is False and n_steps < self.max_steps):
            action = self.planning()
            curr_step = self.env.step(action)

            # Update metrics being tracked
            tot_reward += curr_step.reward
            R += discount*curr_step.reward
            discount *= self.gamma

            obs_count += (action == self.obs_action)  # count number of times an extra observation was taken
            states_visited += [self.env.state]

            # update tree structure metrics
            observe_node = self.root.children[self.obs_action]
            N_observe += [observe_node.N]  # number of times 'observe' was sampled at root

            observe_branches = [onode for onode in observe_node.children if
                                onode is not None]  # list of nodes representing histories that follow 'observe'
            N_branch_obs += [len(observe_branches)]
            obs_branch_visits += [branch.N for branch in observe_branches]

            # Update root node according to action taken and observation received
            self.update_root(action, curr_step.observation)

            print(f"Step {n_steps}: agent took action {action} in state {self.env.state} and received reward {curr_step.reward} "
                  f"and observation {curr_step.observation} for a running total reward of {tot_reward}, discounted return of {R}, "
                  f"and {obs_count} extra observations.")

            assert len(self.root.history) == 2*(len(states_visited)-1), f"Mismatched lengths between history and states visited: {self.root.history} and {states_visited}"

            n_steps += 1

        log['total steps'] = n_steps
        log['success'] = self.env.goal_reached()
        log['done'] = curr_step.done
        log['total reward'] = tot_reward
        log['discounted return'] = R
        log['observation count'] = obs_count
        log['final history'] = self.root.history
        log['states visited'] = states_visited

        # search tree structure statistics
        log["N 'observe' at root"] = N_observe   # number of times 'observe' was sampled at root
        log["N branches at 'observe'"] = N_branch_obs
        log["'observe' branch visits"] = obs_branch_visits

        return log

    def planning(self):
        '''Run tree search and select the best action.'''

        # Run tree search
        self.tree_search(self.root, self.n_rollouts)

        # Select and return the best action
        return self.select_action()

    def select_action(self):
        '''Criterion for selecting actions'''
        # first find greedy real action
        q_info = self.root.get_best_action(self.real_action_space)
        action = q_info['action']
        Q_main = q_info['Q']
        Q_sqr_main = q_info['Q-squared']
        N = q_info['N']

        # get Q-value of 'observe'
        Q_obs = self.root.children[self.obs_action].V
        Q_sqr_obs = self.root.children[self.obs_action].V_sqr
        N_obs = self.root.children[self.obs_action].N

        # cannot compute t-statistic if "observe" action sampled only once. default to not observing in that case.
        if N_obs <= 1:
            return action
        elif N <= 1:  # if real actions sampled only once, t-test also not valid so default to taking an observation
            return self.obs_action

        # do t-test to compare the two values
        t_stat, dof = self.get_t_stat(Q_main, Q_obs, Q_sqr_main, Q_sqr_obs, N, N_obs)

        if t_stat == dof == None:
            if Q_main > Q_obs:
                return action
            else:
                return self.obs_action

        # if 'observe' has significantly better Q-value, return 'observe' as next action
        if t_distr.cdf(t_stat, dof) > 1 - self.pval:
            return self.obs_action
        else:
            return action

    def get_t_stat(self, Q1, Q2, Q1_sqr, Q2_sqr, N1, N2):
        std_err1 = (Q1_sqr - Q1**2) / (N1-1)
        dof_denom1 = std_err1**2 / (N1-1)

        std_err2 = (Q2_sqr - Q2**2) / (N2-1)
        dof_denom2 = std_err2**2 / (N2-1)

        assert std_err1 >= 0, f"Error: std_err1 is {std_err1}"
        assert std_err2 >= 0, f"Error: std_err2 is {std_err2}"

        # if both std_err are 0, default to picking largest value
        if std_err1 == std_err2 == 0:
            return None, None

        t_stat = (Q2 - Q1) / np.sqrt(std_err1 + std_err2)
        dof = (std_err1 + std_err2) ** 2 / (dof_denom1 + dof_denom2)

        return t_stat, dof

    def update_root(self, action, obs):
        '''Updates the search tree according to the action taken and observation received.
        First check if there is already a node in the tree with a matching history. If so, set that to be the new root.
        Otherwise, create a new node.'''

        old_belief = self.root.belief
        anode = self.root.children[action]

        # Check if there is a matching history amongst action node's children. If so, update root to that node.
        if anode.children[obs] is not None:
            self.root = anode.children[obs]
        else:
            new_history = self.root.history + [f'a{action}'] + [f'o{obs}']
            self.root = ONode(history=new_history, n_actions=self.n_actions, q_thresh=self.q_thresh)
            self.root.belief = np.zeros(self.n_states, dtype=int_type)

        # update the old belief and add this to the pre-existing particles in the new belief
        self.root.belief += self.belief_update(old_belief, action, obs, self.n_samples)

        # if particle filter empty, agent is effectively lost. default to uniform prior.
        if np.sum(self.root.belief) == 0:
            self.root.belief = self.uniform_prior()
