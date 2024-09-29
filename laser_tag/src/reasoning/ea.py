from .pomcp import POMCP

class EA(POMCP):
    def __init__(self, max_depth, max_it, k, kwargs):
        super(EA, self).__init__(max_depth, max_it, k, kwargs)
        self.obs_as_action = True

def ea_planning(env, agent, max_it=5000, k=5000, max_depth=20, **kwargs):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. EA Planning
    # - initialising/getting the planning algorithm
    ea = EA(max_depth, max_it, k, kwargs) if 'ea' not \
     in agent.smart_parameters else agent.smart_parameters['ea']

    # - planning
    next_action, bestQ, bestQ_sqr, N, info = ea.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['ea'] = ea
    agent.smart_parameters['count'] = info
    return next_action