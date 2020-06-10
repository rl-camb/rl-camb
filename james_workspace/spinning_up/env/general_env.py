import gym, time
import numpy as np

from gym import spaces, envs

class Env(object):
    """A general class that wraps an environment.
    Environments should inherit from this class.
    """

    def __init__(self, max_episodes, selected_env='MountainCar-v0'):
        
        # The environment
        self.env = gym.make(selected_env)

        # Some meta parameters
        self.max_episodes = max_episodes
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

    def get_spaces(self, registry=True):
        """Print out the action and observation spaces for viewing."""
        if registry:
            print("Registry:\n", envs.registry.all())
        print("Action space:", self.env.action_space)
        #> Discrete(2)
        print("Obs space:", self.env.observation_space)
        #> Box(4,)

    def get_score(self, state, next_state, reward, step):
        """Returns the score for an episode"""
        raise NotImplementedException("Should be overridden")

    def reward_on_step(self, state, next_state, reward, done):
        """Returns the reward for a step within an episode"""
        raise NotImplementedException("Should be overridden")
    
    def check_solved_on_done(self, state, episodes, scores, verbose=False):
        """Checks if the task was completed by this episode"""
        raise NotImplementedException("This method should be overridden")
