import numpy as np

int_type = np.uint64
Ninit, Vinit, V_sqr_init = 0, 0, 0  # Ninit = -1 used to indicate node in need of rollout

class Node():
    def __init__(self, history, thresh=1E-15):

        self.N = Ninit
        self.V = Vinit
        self.V_sqr = V_sqr_init
        self.children = None
        self.history = history
        self.tree_depths = []     # tracks depths of tree from current node
        self.rollout_depths = []  # tracks planning depth (depth of each rollout) from current node
        self.thresh = thresh

class ONode(Node):
    '''Observation search tree node'''

    def __init__(self, history, n_actions):
        '''
        Parameters:
            history : list of observations and actions denoting history for current node
        '''

        super().__init__(history)
        self._belief = None
        self.unvisited_actions = list(range(n_actions))  # tracks which actions haven't been tried yet

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

        action_list = [a for a in action_space if self.children[a].V >= Q_max - self.thresh]
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


# class BaseONode(Node):
#     '''Observation search tree node tailored to environment specs.'''
#
#     def __init__(self, history=[], old_belief=None):
#         '''
#         Parameters:
#             history : list of observations and actions denoting history for current node
#             old_belief : list of particles from preceding history which will be updated based on last obs and action in history
#         '''
#
#         # env_size = self.__class__.env_info['size']
#         # walls_bool = self.__class__.walls_bool
#         # label2action = self.__class__.env_info['label to action']
#         action_space = self.__class__.env_info['action space']
#         n_actions = len(action_space)
#
#         super().__init__(history)
#         self._belief = None
#
#         '''Move belief handling into vomcp'''
#         # # set belief: use uniform prior if no history given or filter particles if prior belief and new history given
#         # self.belief = None
#         # if self.history == []:
#         #     # if no history, use uniform prior belief over states
#         #     self.belief = np.ones(env_size, dtype=int_type)
#         #     self.belief[walls_bool] = 0
#         # elif old_belief is not None:
#         #     last_action = label2action(self.history[-2])
#         #     last_obs = self.history[-1]
#         #     self.belief = filter_particles(old_belief, last_action, last_obs)
#
#         # initialise a list for child action nodes
#         self.children = [None]*n_actions
#
#     @property
#     def belief(self):
#         return self._belief
#
#     @belief.setter
#     def belief(self, new_belief):
#         self._belief = new_belief
#
#
# class BaseANode(Node):
#     '''Observation search tree node tailored to environment specs.'''
#
#     def __init__(self, history):
#         super().__init__(history)
#
#         state_size = self.__class__.env_info['size']
#         self.children = [None] * state_size
