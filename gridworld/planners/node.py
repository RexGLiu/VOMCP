import numpy as np

int_type = np.uint64
Ninit, Vinit, V_sqr_init = 0, 0, 0

class Node():
    def __init__(self, history):

        self.N = Ninit
        self.V = Vinit
        self.V_sqr = V_sqr_init
        self.children = None
        self.history = history
        self.tree_depths = []     # tracks depths of tree from current node
        self.rollout_depths = []  # tracks planning depth (depth of each rollout) from current node

class ONode(Node):
    '''Observation search tree node'''

    def __init__(self, history, n_actions, q_thresh):
        '''
        Parameters:
            history : list of observations and actions denoting history for current node
        '''

        super().__init__(history)
        self._belief = None
        self.unvisited_actions = list(range(n_actions))  # tracks which actions haven't been tried yet
        self.q_thresh = q_thresh

    @property
    def belief(self):
        return self._belief

    @belief.setter
    def belief(self, new_belief):
        self._belief = new_belief

    def get_best_action(self, action_space):
        assert self.children is not None, f"Children action nodes not set. Cannot select best action."

        Q_values = {a: self.children[a].V for a in action_space}
        greedy_action = max(Q_values, key=lambda a: Q_values[a])
        Q_max = self.children[greedy_action].V

        action_list = [a for a in action_space if self.children[a].V >= Q_max - self.q_thresh]
        action = np.random.choice(action_list)

        Q = self.children[greedy_action].V
        Q_sqr = self.children[action].V_sqr
        N = self.children[action].N

        q_info = {'action': action,
                  'Q': Q,
                  'Q-squared': Q_sqr,
                  'N': N}
        return q_info


class ANode(Node):
    '''Observation search tree node'''

    def __init__(self, history, obs_space):
        '''
        Parameters:
            history : list of observations and actions denoting history for current node
            obs_space : size of the observation space
        '''

        super().__init__(history)
        self.obs_space = obs_space

        # initialise a list for child action nodes
        self.children = [None] * obs_space
