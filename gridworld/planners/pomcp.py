import numpy as np
from .node import ONode, int_type
from .base_planner import BasePlanner

class POMCP(BasePlanner):
    def __init__(self, env, params):
        super().__init__(env, params)

        # only consider the 'real' actions
        self.action_space = env.action_space[:-1]
        self.n_actions = len(self.action_space)

        self.root = ONode(history=[], n_actions=self.n_actions, q_thresh=self.q_thresh)
        self.root.belief = self.uniform_prior()

    def run_episode(self):
        log = {}
        curr_step = self.env.reset()

        tot_reward = 0.0
        R = 0.0
        discount = 1

        states_visited = [self.env.state]

        n_steps = 0
        while (curr_step.done is False and n_steps < self.max_steps):
            action = self.planning()
            curr_step = self.env.step(action)

            # Update metrics being tracked
            tot_reward += curr_step.reward
            R += discount*curr_step.reward
            discount *= self.gamma

            states_visited += [self.env.state]

            # Update root node according to action taken and observation received
            self.update_root(action, curr_step.observation)

            print(f"Step {n_steps}: agent took action {action} in state {self.env.state} and received reward {curr_step.reward} "
                  f"and observation {curr_step.observation} for a running total reward of {tot_reward}, discounted return of {R}.")

            assert len(self.root.history) == 2*(len(states_visited)-1), f"Mismatched lengths between history and states visited: {self.root.history} and {states_visited}"

            n_steps += 1

        log['total steps'] = n_steps
        log['success'] = self.env.goal_reached()
        log['done'] = curr_step.done
        log['total reward'] = tot_reward
        log['discounted return'] = R
        log['observation count'] = 0
        log['final history'] = self.root.history
        log['states visited'] = states_visited

        return log


    def planning(self):
        '''Run tree search and select the best action.'''

        # Run tree search
        self.tree_search(self.root, self.n_rollouts)

        # Select and return the best action
        return self.select_action()

    def select_action(self):
        '''Criterion for selecting actions'''

        q_info = self.root.get_best_action(self.action_space)
        return q_info['action']

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
