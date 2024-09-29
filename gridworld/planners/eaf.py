import numpy as np
from .node import ONode, ANode, Ninit, int_type
from .ea import EA
from scipy.stats import t as t_distr

class EAF(EA):
    def __init__(self, env, params):
        super(EA, self).__init__(env, params)

        self.real_action_space = self.action_space[:-1]
        self.n_actions = len(self.real_action_space)

        self.root = ONode(history=[], n_actions=self.n_actions+1)
        self.root.belief = self.uniform_prior()

        self.pval = params['pval']

    def reset(self):
        self.root = ONode(history=[], n_actions=self.n_actions+1)
        self.root.belief = self.uniform_prior()
        self.env.reset()

    def update_root(self, action, obs):
        super().update_root(action, obs)

        # expand actions under consideration at root to include 'observe'
        if self.root.children is not None:
            self.root.children += [ANode(history=self.root.history+[f'a{self.obs_action}'], obs_space=self.n_states)]
        self.root.unvisited_actions += [self.obs_action]

    def traverse_tree(self, curr_node, state, depth):
        '''Traverses search tree according to UCT policy until reaching leaf node or maximum depth.'''
        assert depth > 0, "Error: tree depth must be at least 1"

        # Max depth reached
        if depth >= self.max_depth:
            rollout_depth = (depth-self.start_depth)//2
            return 0, depth, rollout_depth

        # Leaf node reached: initialise children nodes
        if curr_node.N == Ninit:
            # initialise children nodes
            curr_node.children = [ANode(history=curr_node.history+[f'a{a}'], obs_space=self.n_states) for a in range(self.n_actions)]

            # if root node, expand actions under consideration to include 'observe'
            if depth == self.start_depth:
                curr_node.children += [ANode(history=self.root.history + [f'a{self.obs_action}'], obs_space=self.n_states)]

        # Increment node visitation count
        curr_node.N += 1

        # If some actions haven't been tried yet, pick one at random, extend tree, and then continue with random rollout
        if len(curr_node.unvisited_actions) > 0:
            # select action at random from untried actions
            action = np.random.choice(curr_node.unvisited_actions)
            curr_node.unvisited_actions.remove(action)  # delete chosen action from list of untried actions

            # simulate action
            next_state, next_step = self.env.virtual_step(state, action)

            # extend tree
            anode = curr_node.children[action]
            next_history = curr_node.history+[f'a{action}']+[f'o{next_step.observation}']
            next_onode = ONode(history=next_history, n_actions=self.n_actions)
            if self.retain_particles:
                next_onode.belief = np.zeros(self.n_states, dtype=int_type)
                next_onode.belief[next_state] = 1
            anode.children[next_step.observation] = next_onode
            anode.N += 1

            # random rollout
            r, rollout_depth = self.random_rollout(next_state, depth+2)
                # depth increases by two bc we increment by one action and one obs node

            # track tree depth of 3 from current node
            curr_node.tree_depths += [3]
            anode.tree_depths += [2]

            # track number of rollout steps
            starting_rollout_depth = (depth-self.start_depth)//2
            curr_node.rollout_depths += [rollout_depth - starting_rollout_depth]
                # note: this accounting differs slightly from counting number of tree layers

            return next_step.reward + self.gamma * r, depth+2, rollout_depth

        # Select action using UCT tree policy
        if depth == self.start_depth:
            action = self.UCT_policy(curr_node, self.action_space)
        else:
            action = self.UCT_policy(curr_node, self.real_action_space)
        anode = curr_node.children[action]
        anode.N += 1

        # Simulate action
        next_state, next_step = self.env.virtual_step(state, action)

        next_onode = anode.children[next_step.observation]

        # If necessary, create new ONode for next history.
        if next_onode is None:
            next_history = curr_node.history + [f'a{action}'] + [f'o{next_step.observation}']
            next_onode = ONode(history=next_history, n_actions=self.n_actions)
            if self.retain_particles:
                next_onode.belief = np.zeros(self.n_states, dtype=int_type)
            anode.children[next_step.observation] = next_onode

        # Update belief for next_onode
        if self.retain_particles:
            next_onode.belief[next_state] += 1

        # Continue tree traversal
        r, max_depth, rollout_depth = self.traverse_tree(next_onode, next_state, depth+2)
            # depth increases by two bc we increment by one action and one obs node

        # Compute returns
        R = next_step.reward + self.gamma * r

        # Update node's values
        anode.V += (R - anode.V) / anode.N
        anode.V_sqr += (R*R - anode.V_sqr) / anode.N

        # track max depth reached from current node
        curr_node.tree_depths += [max_depth-depth+1]
        anode.tree_depths += [max_depth-depth]

        # track number of rollout steps
        starting_rollout_depth = (depth - self.start_depth) // 2
        curr_node.rollout_depths += [rollout_depth - starting_rollout_depth]
            # note: this accounting differs slightly from counting number of tree layers

        return R, max_depth, rollout_depth

    def random_rollout(self, state, depth):
        if depth >= self.max_depth or self.env.is_terminal(state):
            rollout_depth = (depth-self.start_depth)//2
            return 0, rollout_depth

        action = np.random.choice(self.real_action_space)  # rollout policy chooses uniformly from all possible real actions
        next_state, next_step = self.env.virtual_step(state, action)

        r, rollout_depth = self.random_rollout(next_state, depth+2)
            # depth increases by two bc we increment by one action and one obs node

        return next_step.reward + self.gamma * r, rollout_depth

    def run_history(self, history):
        tree_depths = []
        rollout_depths = []

        n_steps = len(history) // 2

        for ii in range(n_steps):
            if ii > 0:
                next_history = history[:2*ii]
                next_action = next_history[-2]
                next_obs = next_history[-1]
                updated_belief = self.belief_update(self.root.belief, next_action, next_obs, self.n_samples)
                self.root = ONode(history=next_history, n_actions=self.n_actions+1)
                self.root.belief = updated_belief

            self.planning()

            tree_depths += [[self.root.tree_depths]]         # tree_depths list has dims n_histories x n_trees x n_depths
            rollout_depths += [[self.root.rollout_depths]]   # rollout_depths list has dims n_histories x n_trees x n_depths

        results = {'tree depths': tree_depths, 'rollout depths': rollout_depths}
        return results


class EAF_modified(EAF):
    def select_action(self):
        '''Criterion for selecting actions based on a t-test comparison of greedy real action against 'observe'.'''

        # first find greedy real action
        q_info = self.root.get_best_action(self.real_action_space)
        action = q_info['action']
        Q_main = q_info['Q']
        Q_sqr_main = q_info['Q-squared']
        N_main = q_info['N']

        std_err1 = (Q_sqr_main - Q_main**2) / (N_main-1)
        dof_denom1 = std_err1**2 / (N_main-1)

        # loop through branches of each virtual obs to compute Q-value of 'observe'
        obs_branches = self.root.children[self.obs_action].children

        Q_obs = 0
        std_err2 = 0
        dof_denom2 = 0
        N_obs = 0
        for onode in obs_branches:
            if onode is None or onode.children is None:  # skip branch if history never visited or if no rollouts were made from given history
                continue

            q_info = onode.get_best_action(self.real_action_space)
            Q = self.gamma * q_info['Q']                    # value of best action that follows 'observe' must be discounted by gamma
            Q_sqr = self.gamma**2 * q_info['Q-squared']
            N = q_info['N']

            if N <= 1:
                print(f"Warning: observation branch visited only {N} times. Skipping observation.")
                continue

            N_obs += N
            Q_obs += N*Q
            std_err_ = N**2 * (Q_sqr - Q**2) / (N-1)
            std_err2 += std_err_
            dof_denom2 += std_err_**2 / (N-1)

        # if "observe" branch under-sampled, default to real action.
        if N_obs == 0:
            return action
        else: # otherwise, normalise stats for "observe" action
            Q_obs /= N_obs
            std_err2 /= N_obs**2
            dof_denom2 /= N_obs**4

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
