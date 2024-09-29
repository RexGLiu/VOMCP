import numpy as np
from scipy.stats import t as t_distr
from .node import ONode, int_type
from .base_planner import BasePlanner

class VOMCP(BasePlanner):
    def __init__(self, env, params):
        super().__init__(env, params)

        self.n_vobs = params['n_vobs']
        self.pval = params['pval']
        self.virtual_trees = None
        self.vobs_dict = None
        self.split = params['split']

        # only consider the 'real' actions
        self.action_space = env.action_space[:-1]
        self.n_actions = len(self.action_space)

        self.root = ONode(history=[], n_actions=self.n_actions)
        self.root.belief = self.uniform_prior()

    def run_episode(self):
        log = {}
        curr_step = self.env.reset()

        tot_reward = 0.0
        R = 0.0
        discount = 1
        obs_count = 0

        states_visited = [self.env.state]

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

            # Update root node according to action taken and observation received
            self.update_root(action, curr_step.observation)

            print(f"Step {n_steps}: agent took action {action} in state {self.env.state} and received reward {curr_step.reward} "
                  f"and observation {curr_step.observation} for a running total reward of {tot_reward}, discounted return of {R}, "
                  f"and {obs_count} extra observations.")

            assert len(self.root.history) == 2*(len(states_visited)-1), f"Mismatched lengths between history and states visited: {self.root.history} and {states_visited}"

            self.virtual_trees = None
            self.vobs_dict = None

            n_steps += 1

        log['total steps'] = n_steps
        log['success'] = self.env.goal_reached()
        log['done'] = curr_step.done
        log['total reward'] = tot_reward
        log['discounted return'] = R
        log['observation count'] = obs_count
        log['final history'] = self.root.history
        log['states visited'] = states_visited

        return log

    def planning(self):
        '''Split indicates fraction of budget to allocate to virtual trees. Default uses equal partition across all trees.'''

        self.vobs_dict = self.sample_vobs(self.root.belief)

        # Determine budget split between main and virtual trees
        if self.split is None:
            n_trees = len(self.vobs_dict) + 1
            virtual_samples = self.n_samples // n_trees
            main_rollouts = self.n_rollouts // n_trees
            virtual_rollouts = main_rollouts
        else:
            n_trees = len(self.vobs_dict)
            virtual_samples = int(self.split * self.n_samples // n_trees)
            virtual_rollouts = int(self.split * self.n_rollouts // n_trees)
            main_rollouts = int( (1-self.split) * self.n_rollouts)

        # Create root node for each virtual search
        self.virtual_trees = {k: ONode(history=self.root.history+[f'a{self.obs_action}']+[f'o{k}'], n_actions=self.n_actions) for k in self.vobs_dict}

        # Generate belief for each virtual observation
        for k in self.virtual_trees:
            updated_belief = self.belief_update(self.root.belief, self.obs_action, k, virtual_samples)

            # if belief update fails, fall back to uniform prior
            if np.sum(updated_belief) == 0:
                updated_belief = self.uniform_prior()

            # # otherwise do a reinvigoration step
            # elif self.reinvigorate:
            #     updated_belief = self.particle_reinvigoration(updated_belief, virtual_samples)

            self.virtual_trees[k].belief = updated_belief

        # Run tree search on main tree
        self.tree_search(self.root, main_rollouts)

        # Run tree search on each virtual tree
        for v_root in self.virtual_trees.values():
            self.tree_search(v_root, virtual_rollouts)

        # Select the best action
        best_action = self.select_action()

        return best_action

    def sample_vobs(self, belief):
        '''Samples self.n_vobs virtual observations from belief. Returns observations and their multiplicities as
        a dictionary {obs: multiplicity}.'''

        obs_dict = {}

        # sample the virtual observations
        virtual_states = self.sample_particles(belief, self.n_vobs)
        for s in virtual_states:
            v_obs = self.env.get_obs(s)
            if v_obs in obs_dict:
                obs_dict[v_obs] += 1
            else:
                obs_dict[v_obs] = 1

        return obs_dict

    def select_action(self):
        '''Criterion for selecting actions based on a t-test comparison of greedy real action against 'observe'.'''

        # first find greedy real action
        q_info = self.root.get_best_action(self.action_space)
        action = q_info['action']
        Q_main = q_info['Q']
        Q_sqr_main = q_info['Q-squared']
        N = q_info['N']

        std_err1 = (Q_sqr_main - Q_main**2) / (N-1)
        dof_denom1 = std_err1**2 / (N-1)

        # loop through virtual trees to compute Q-value of 'observe'
        assert sum(self.vobs_dict.values()) == self.n_vobs, f"Error: mismatch between number of virtual obs sampled {self.vobs_dict} and n_vobs {self.n_vobs}"
        Q_obs = 0
        std_err2 = 0
        dof_denom2 = 0
        for obs, tree in self.virtual_trees.items():
            weight = self.vobs_dict[obs] / self.n_vobs

            q_info = tree.get_best_action(self.action_space)
            Q = self.gamma * q_info['Q']                    # value of best action that follows 'observe' must be discounted by gamma
            Q_sqr = self.gamma**2 * q_info['Q-squared']
            N = q_info['N']

            Q_obs += weight * Q
            std_err_ = weight**2 * (Q_sqr - Q**2) / (N-1)
            std_err2 += std_err_
            dof_denom2 += std_err_**2 / (N-1)

        # if both std_err are 0, default to picking largest value
        if std_err1 == std_err2 == 0:
            if Q_main > Q_obs:
                return action
            else:
                return self.obs_action

        # compute t-statistic to compare the two values
        t_stat = (Q_obs - Q_main) / np.sqrt(std_err1 + std_err2)
        dof = (std_err1 + std_err2)**2 / (dof_denom1 + dof_denom2)

        # if 'observe' has significantly better Q-value, return 'observe' as next action
        if t_distr.cdf(t_stat, dof) > 1 - self.pval:
            return self.obs_action
        else:
            return action

    def update_root(self, action, obs):
        '''Updates the search tree according to the action taken and observation received.
        First check if there is already a node in the tree with a matching history. If so, set that to be the new root.
        Otherwise, create a new node.'''

        old_belief = self.root.belief
        matching_node = False

        # if action was a real action, search for next node from amongst children of the main search tree.
        if action != self.obs_action:
            anode = self.root.children[action]

            # Check if there is a matching history amongst action node's children. If so, update root to that node.
            if anode.children[obs] is not None:
                self.root = anode.children[obs]
                matching_node = True
        else: # if action was 'observe', search for a matching history amongst the virtual search trees
            for v_obs, tree in self.virtual_trees.items():
                if v_obs == obs:
                    self.root = tree
                    matching_node = True
                    break

        # Create new root if history not encountered during search
        if matching_node is False:
            new_history = self.root.history + [f'a{action}'] + [f'o{obs}']
            self.root = ONode(history=new_history, n_actions=self.n_actions)

        # Update the old belief and add this to the pre-existing particles in the new belief.
        # Note: particle budget per tree should be given by self.n_samples // n_trees. But we do not know n_trees
        # in advance, as this requires sampling obs from the root belief distribution. So we use n_vobs in place of n_trees here.
        n_samples = self.n_samples // (self.n_vobs + 1)
        if self.root.belief is None or self.retain_particles is False:
            self.root.belief = np.zeros(self.n_states, dtype=int_type)
        self.root.belief += self.belief_update(old_belief, action, obs, n_samples)

        # if particle filter empty, agent is effectively lost. default to uniform prior.
        if np.sum(self.root.belief) == 0:
            self.root.belief = self.uniform_prior()

        # # perform a reinvigoration step if requested
        # if self.reinvigorate:
        #     self.root.belief = self.particle_reinvigoration(self.root.belief, self.n_samples)

    def run_history(self, history):
        tree_depths = []
        rollout_depths = []

        n_steps = len(history) // 2

        # Note: particle budget per tree should be given by self.n_samples // n_trees. But we do not know n_trees
        # in advance, as this requires sampling obs from the root belief distribution. So we use n_vobs in place of n_trees here.
        n_samples = self.n_samples // (self.n_vobs + 1)

        for ii in range(n_steps):
            if ii > 0:
                next_history = history[:2*ii]
                next_action = next_history[-2]
                next_obs = next_history[-1]
                updated_belief = self.belief_update(self.root.belief, next_action, next_obs, n_samples)
                self.root = ONode(history=next_history, n_actions=self.n_actions + 1)
                self.root.belief = updated_belief

            self.planning()

            tree_depths += [[self.root.tree_depths] + [v_root.tree_depths for v_root in self.virtual_trees.values()]]
                # tree_depths list has dims n_histories x n_trees x n_depths
            rollout_depths += [[self.root.rollout_depths] + [v_root.rollout_depths for v_root in self.virtual_trees.values()]]
                # rollout_depths list has dims n_histories x n_trees x n_depths

        results = {'tree depths': tree_depths, 'rollout depths': rollout_depths}
        return results


class VOMCP_avg(VOMCP):
    def select_action(self):
        '''Criterion for selecting actions based on a t-test comparison of greedy real action against 'observe'.
        Here, 'observe' is valued in a similar manner to EAF -- by averaging across all histories and all actions
        that follow the virtual observation.'''

        # first find greedy real action
        q_info = self.root.get_best_action(self.action_space)
        action = q_info['action']
        Q_main = q_info['Q']
        Q_sqr_main = q_info['Q-squared']
        N = q_info['N']

        # loop through virtual trees to compute Q-value of 'observe'
        assert sum(self.vobs_dict.values()) == self.n_vobs, f"Error: mismatch between number of virtual obs sampled {self.vobs_dict} and n_vobs {self.n_vobs}"
        Q_obs = 0
        Q_sqr_obs = 0
        N_obs = 0
        for obs, tree in self.virtual_trees.items():
            multiplicity = self.vobs_dict[obs]

            for a in self.action_space:
                anode = tree.children[a]

                N_a = anode.N
                Q_a = anode.V
                Q_sqr_a = anode.V_sqr

                Q_obs += multiplicity * N_a * self.gamma * Q_a
                Q_sqr_obs += multiplicity * N_a * self.gamma**2 * Q_sqr_a
                N_obs += multiplicity * N_a

        Q_obs /= N_obs
        Q_sqr_obs /= N_obs

        # do t-test to compare the two values
        t_stat, dof = self.get_t_stat(Q_main, Q_obs, Q_sqr_main, Q_sqr_obs, N, N_obs)

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

        t_stat = (Q2 - Q1) / np.sqrt(std_err1 + std_err2)
        dof = (std_err1 + std_err2) ** 2 / (dof_denom1 + dof_denom2)

        return t_stat, dof
