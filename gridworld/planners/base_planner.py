import numpy as np
from .node import ONode, ANode, Ninit

int_type = np.uint64

class BeliefException(Exception):
    def __init__(self, belief):
        self.belief = belief
        self.message = "Invalid belief distribution"
        super(BeliefException, self).__init__(self.message)

class BasePlanner():
    def __init__(self, env, params):
        self.env = env
        self.gamma = params['gamma']
        self.n_samples = params['n_samples']
        self.n_rollouts = params['n_rollouts']
        self.fixed_filter = params['fixed_filter']
        self.c = params['c']
        self.start_depth = 1
        self.max_depth = params['max_depth']
        self.max_steps = params['max_steps']
        self.q_thresh = params['q_thresh']

        # set space of actions that tree will consider
        self.action_space = env.action_space
        self.n_actions = len(self.action_space)
        self.obs_action = self.n_actions - 1    # always assume the 'observe' action is last

        self.n_states = len(self.env.observation_space)

        self.root = ONode(history=[], n_actions=self.n_actions, q_thresh=self.q_thresh)
        self.root.belief = self.uniform_prior()

    def reset(self):
        self.root = ONode(history=[], n_actions=self.n_actions, q_thresh=self.q_thresh)
        self.root.belief = self.uniform_prior()
        self.env.reset()

    def run_episode(self):
        '''Run a single episode to completion.'''
        raise NotImplementedError('Method for running an episode needs to be implemented')

    def planning(self):
        '''Run tree search and select the best action.'''
        raise NotImplementedError('Search algorithm needs to be implemented')

    def select_action(self, **kwargs):
        '''Criterion for selecting actions'''
        raise NotImplementedError('Action selection criterion needs to be implemented')

    def update_root(self, action, obs):
        '''Updates the search tree according to the action taken and observation received.'''
        raise NotImplementedError('Action selection criterion needs to be implemented')

    def tree_search(self, root, n_rollouts):
        '''Performs a POMCP search on the given root node.'''

        sampled_states = self.sample_particles(root.belief, n_rollouts)
        for s in sampled_states:
            self.traverse_tree(root, s, depth=self.start_depth)

    def traverse_tree(self, curr_node, state, depth):
        '''Traverses search tree according to UCT policy until reaching leaf node or maximum depth. Returns discounted return
         from current history in traversal and max tree depth reached.'''

        # Max depth reached
        if depth >= self.max_depth:
            rollout_depth = (depth-self.start_depth)//2
            return 0, depth, rollout_depth

        # Leaf node reached: initialise children nodes
        if curr_node.N == Ninit:
            # initialise children nodes
            curr_node.children = [ANode(history=curr_node.history+[f'a{a}'], obs_space=self.n_states) for a in range(self.n_actions)]

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
            next_onode = ONode(history=next_history, n_actions=self.n_actions, q_thresh=self.q_thresh)
            next_onode.belief = np.zeros(self.n_states, dtype=int_type)
            next_onode.belief[next_state] = 1
            anode.children[next_step.observation] = next_onode
            anode.N += 1

            # random rollout
            r, rollout_depth = self.random_rollout(next_state, depth+2)
                # depth increases by two bc we increment by one action and one obs node

            # Compute returns
            R = next_step.reward + self.gamma * r

            # Update node's values
            anode.V += (R - anode.V) / anode.N
            anode.V_sqr += (R*R - anode.V_sqr) / anode.N

            # track tree depth of 3 from current node
            curr_node.tree_depths += [3]
            anode.tree_depths += [2]

            # track number of rollout steps
            starting_rollout_depth = (depth-self.start_depth)//2
            curr_node.rollout_depths += [rollout_depth - starting_rollout_depth]
                # note: this accounting differs slightly from counting number of tree layers

            return R, depth+2, rollout_depth

        # Select action using UCT tree policy
        action = self.UCT_policy(curr_node, self.action_space)
        anode = curr_node.children[action]
        anode.N += 1

        # Simulate action
        next_state, next_step = self.env.virtual_step(state, action)

        next_onode = anode.children[next_step.observation]

        # If necessary, create new ONode for next history.
        if next_onode is None:
            next_history = curr_node.history + [f'a{action}'] + [f'o{next_step.observation}']
            next_onode = ONode(history=next_history, n_actions=self.n_actions, q_thresh=self.q_thresh)
            next_onode.belief = np.zeros(self.n_states, dtype=int_type)
            anode.children[next_step.observation] = next_onode

        # Update belief for next_onode
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

        action = np.random.choice(self.action_space)  # rollout policy chooses uniformly from all possible actions
        next_state, next_step = self.env.virtual_step(state, action)

        r, rollout_depth = self.random_rollout(next_state, depth+2)
            # depth increases by two bc we increment by one action and one obs node

        return next_step.reward + self.gamma * r, rollout_depth

    def uniform_prior(self):
        n_states = self.env.map.size
        belief = np.ones(n_states, dtype=int_type)
        belief[self.env.walls] = 0
        return belief

    def belief_update(self, old_belief, action, obs, n_samples):
        '''Update a set of particles given an action and resulting observation.'''
        belief = np.zeros(self.n_states, dtype=int_type)

        if self.fixed_filter:
            particle_count = 0  # number of particles added to belief so far
            iters = 0
            max_iters = 100*n_samples
            while particle_count < n_samples and iters < max_iters:
                # sample state from old belief, get particle for that state, and perform action
                prev_state = self.sample_particles(old_belief)
                next_state, next_step = self.env.virtual_step(prev_state, action)

                # add to belief if resulting virtual observation matches obs
                if next_step.observation == obs:
                    belief[next_state] += 1
                    particle_count += 1
                iters += 1
        else:
            for _ in range(n_samples):
                # sample state from old belief, get particle for that state, and perform action
                prev_state = self.sample_particles(old_belief)
                next_state, next_step = self.env.virtual_step(prev_state, action)

                # add to belief if resulting virtual observation matches obs
                if next_step.observation == obs:
                    belief[next_state] += 1

        return belief

    def sample_particles(self, belief, nsamples=None):
        '''
        :param belief: set of particles representing belief distribution
        :param nsamples: number of samples to draw. If None, a single value is returned.
        :return: a set of sampled states
        '''
        distr = belief / np.sum(belief)
        try:
            sampled_states = np.random.choice(belief.size, p=distr, size=nsamples)
        except ValueError:
            print("Value error")
            print("Normalisation: ", float(np.sum(belief)))
            print("Distribution: ", distr)
            print("belief: ", belief)
            raise BeliefException(belief)
        return sampled_states

    def UCT(self, onode, action):
        anode = onode.children[action]
        assert anode != None, f"Error: action node of onode not set"
        return anode.V + self.c * np.sqrt(np.log(onode.N) / anode.N)

    def UCT_policy(self, onode, action_space):
        UCT_values = [self.UCT(onode, a) for a in action_space]
        greedy_action = max(action_space,key=lambda a:UCT_values[a])
        max_UCT = UCT_values[greedy_action]
        action_list = [a for a in action_space if UCT_values[a] >= max_UCT - self.q_thresh]  # randomly break statistical ties
        return np.random.choice(action_list)

    def run_history(self, history):
        tree_depths = []
        rollout_depths = []

        n_steps = len(history) // 2

        for ii in range(n_steps+1):
            if ii > 0:
                next_history = history[:2*ii]
                next_action = next_history[-2]
                next_obs = next_history[-1]
                updated_belief = self.belief_update(self.root.belief, next_action, next_obs, self.n_samples)
                self.root = ONode(history=next_history, n_actions=self.n_actions, q_thresh=self.q_thresh)
                self.root.belief = updated_belief

            self.planning()

            tree_depths += [[self.root.tree_depths]]         # tree_depths list has dims n_histories x n_trees x n_depths
            rollout_depths += [[self.root.rollout_depths]]   # rollout_depths list has dims n_histories x n_trees x n_depths

        results = {'tree depths': tree_depths, 'rollout depths': rollout_depths}
        return results