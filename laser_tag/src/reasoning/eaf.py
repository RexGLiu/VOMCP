from src.reasoning.node import ANode, ONode, find_new_PO_root, \
    particle_revigoration, uniform_prior
from .pomcp import POMCP, obs_based_particle_update, is_terminal

class EAF(POMCP):
    def planning(self, state, agent):
        # 0. Get particles for old belief distribution
        old_particles = self.root.particle_filter if self.root is not None else None

        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = ONode(observation=None,state=state,depth=0,parent=None,obs_as_action=self.obs_as_action)
        # or advancing within the existent tree
        else:
            self.root = find_new_PO_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary, obs_as_action=self.obs_as_action)

        # Adding "observe" as action to root node only.
        self.root.make_obs_as_action()

        # 3. Update beliefs based on last action and resulting obs
        if old_particles is not None:
            assert len(old_particles) > 0, "old_particles is empty"
            new_particles = obs_based_particle_update(previous_action, current_observation, self.k, self.fixed_filter, old_particles=old_particles)

            # Set particle_filter to new particles if no pre-existing set of particles or if particles have collapsed onto a terminal state
            if self.root.particle_filter == [] or is_terminal(self.root.particle_filter[0]):
                self.root.particle_filter = new_particles
            else:  # else just add to current set of particles
                self.root.particle_filter += new_particles

        # 4. Performing particle revigoration
        if self.pr:
            particle_revigoration(state,agent,self.root,self.k)

        # if not reinvigorating particles, default to uniform prior should we experience collapse of posterior
        elif len(self.root.particle_filter) == 0:
            uniform_prior(state,agent,self.root,self.k)

        # 5. Searching for the best action within the tree
        best_action, bestQ, bestQ_sqr, N = self.search(self.root, agent)

        # 6. Returning the best action
        self.root.show_qtable()
        info = { 'nrollouts': self.rollout_count,
            'nsimulations':self.simulation_count}
        return best_action, bestQ, bestQ_sqr, N, info


def eaf_planning(env, agent, max_it=5000, k=5000, max_depth=20, **kwargs):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. EAF Planning
    # - initialising/getting the planning algorithm
    eaf = EAF(max_depth, max_it, k, kwargs) if 'eaf' not \
     in agent.smart_parameters else agent.smart_parameters['eaf']

    # - planning
    next_action, bestQ, bestQ_sqr, N, info = eaf.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['eaf'] = eaf
    agent.smart_parameters['count'] = info
    return next_action
