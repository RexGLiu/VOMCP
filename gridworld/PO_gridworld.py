import numpy as np
import collections

MAPS = {
    "6x6": [
        "SFFFHF",
        "FFFFFF",
        "FFFHFH",
        "FFHHFF",
        "HFFFHF",
        "FFHFFG"
    ],
     "6x6_hard": [
        "SHFFFF",
        "FHHHHF",
        "FFFFHF",
        "HHHFHF",
        "FFHFFG",
        "FFFHHF"
    ],
    "6x6_easy": [
        "SFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFG"
    ],
    "shortcut": [
        "HHHWW",
        "FFFHW",
        "FHFHW",
        "SWFGF",
        "FWHWF",
        "FWWWF",
        "FWWWF",
        "FWWWF",
        "FFFFF"
    ],
}

Rewards1 = {'G': 1000., 'H': -1}
Rewards2 = {'G':1000000., 'S':1., 'F':1.}

_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)


class GridWorldEnv(object):
    """
    S : starting point
    F : free space
    W : wall
    H : hole (terminates episode)
    G : goal
    """

    _action_dict = dict(
            left=0,
            down=1,
            right=2,
            up=3,
            observe=4
        )
    _n_actions = len(_action_dict)
    _direction_dict = {v : k for k,v in _action_dict.items()}

    def __init__(self, env_params, desc='6x6', survival_rew=False):
        desc = MAPS[desc]
        self.desc = np.array(list(map(list, desc)))
        desc=self.desc
        self.n_row, self.n_col = desc.shape
        self.start_state=0
        self.state = None
        self.domain_fig = None

        self.obs_width = env_params['obs_width']
        self.p_obs = env_params['p_obs']
        self.p_fail = env_params['p_fail']

        if survival_rew:
            self.Rewards = Rewards2
        else:
            self.Rewards = Rewards1

    def reset(self):
        self.state = self.start_state
        next_obs = self.get_obs(self.state)
        return Step(observation=next_obs, reward=0, done=False)

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction.
        :return: the action index corresponding to the given direction
        """
        return GridWorldEnv._action_dict[d]

    @staticmethod
    def direction_from_action(a):
        """
        Return the direction corresponding to the given action.
        :return: the action index corresponding to the given direction
        """
        return GridWorldEnv._direction_dict[a]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        4: observe
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        
        self.state, next_step = self.virtual_step(self.state,action)
        return next_step

    def virtual_step(self, state, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        4: observe
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        possible_next_states = self.get_possible_next_states(state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]
        reward = possible_next_states[next_state_idx][2]

        next_x,next_y = self.idx2pos(next_state)

        next_state_type = self.desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
        elif next_state_type in ['F', 'S']:
            done = False
        elif next_state_type == 'G':
            done = True
        else:
            raise NotImplementedError

        next_obs = self.get_obs(next_state)
        return next_state, Step(observation=next_obs, reward=reward, done=done)

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of tuples (s', p(s'|s,a), r)
        """
        p_fail = self.p_fail

        assert state in self.observation_space
        assert action in self.action_space
        if action==self.action_from_direction('observe'):
            return [(state, 1., 0)]

        x,y = self.idx2pos(state)
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )
        next_state = next_coords[0] * self.n_col + next_coords[1]
        state_type = self.desc[x, y]
        next_state_type = self.desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state==next_state:
            return [(state, 1., self.get_reward('W'))]
        elif state_type == 'H' or state_type == 'G':
            return [(state, 1., 0)]
        else:
            return [(next_state, 1.-p_fail, self.get_reward(next_state_type)), (state, p_fail, self.get_reward(state_type))]

    def get_reward(self, state_type):
        return self.Rewards.get(state_type, 0)
    
    def get_obs(self, state):
        p_obs = self.p_obs
        obs_width = self.obs_width

        x,y = self.idx2pos(state)

        x_obs_space = np.arange(x-obs_width, x+obs_width+1)
        y_obs_space = np.arange(y-obs_width, y+obs_width+1)
        x_obs_space = x_obs_space[np.logical_and(x_obs_space > -1, x_obs_space < self.n_row)]
        y_obs_space = y_obs_space[np.logical_and(y_obs_space > -1, y_obs_space < self.n_col)]
        
        p_x = (1-p_obs)/(x_obs_space.size-1) * np.ones(x_obs_space.size)
        p_y = (1-p_obs)/(y_obs_space.size-1) * np.ones(y_obs_space.size)
        
        p_x[x_obs_space == x] = p_obs
        p_y[y_obs_space == y] = p_obs
        x_obs = np.random.choice(x_obs_space, p=p_x)
        y_obs = np.random.choice(y_obs_space, p=p_y)
        
        obs_state = x_obs * self.n_col + y_obs
        
        return obs_state

    @property
    def action_space(self):
        return list(range(GridWorldEnv._n_actions))

    @property
    def observation_space(self):
        return list(range(self.n_row*self.n_col))

    @property
    def map(self):
        return self.desc

    def idx2pos(self, idx):
        '''Converts state idx to coords'''
        x = idx // self.n_col
        y = idx % self.n_col
        return x,y

    def is_terminal(self, state):
        x, y = self.idx2pos(state)
        return self.desc[x, y] in ['H', 'G']

    @property
    def walls(self):
        return (self.desc=='W').flatten()

    def goal_reached(self):
        return self.desc.flatten()[self.state] == 'G'

    # @property
    # def real_action_space(self):
    #     return list(range(GridWorldEnv._n_actions-1))
    #
    # @property
    # def n_actions(self):
    #     return GridWorldEnv._n_actions
    #
    # @property
    # def action_dict(self):
    #     return GridWorldEnv._action_dict

    # @property
    # def state_space(self):
    #     return self.observation_space

    @property
    def size(self):
        return self.desc.size

    # @property
    # def env_info(self):
    #     _env_info = {
    #         'size': self.size,
    #         'action to label': self.direction_from_action,
    #         'label to action': self.action_from_direction,
    #         'map': self.map,
    #         'state space': self.state_space,
    #         'observation space': self.observation_space,
    #         'action space': self.action_space
    #     }
    #     return _env_info
