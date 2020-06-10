import math, gym
import matplotlib.pyplot as plt
import numpy as np

from . import Env

class CustomCartPole(Env):

    def __init__(self, x_threshold=2.4, angle_threshold=12., max_episodes=2000, max_episode_steps=500, cp_id='CustomCartPole-v1'):

        Env.__init__(self, max_episodes=max_episodes, selected_env='CartPole-v1')

        self.angle_threshold = angle_threshold
        self.x_threshold = x_threshold
        self.max_episode_steps = max_episode_steps
        
        # Lazy singleton implementation - gym throws error if already registered
        try:
            gym.envs.register(
                id=cp_id,
                entry_point='gym.envs.classic_control:CustomCartPoleEnv',
                max_episode_steps=max_episode_steps,
                kwargs={'x_threshold' : x_threshold, 'angle_threshold' : angle_threshold}
            )
        except:
            pass

        # Override the initialised env
        self.env = gym.make('CustomCartPole-v1')

    def solve(self, plot=False, verbose=False, render=True):
        """Wrapper for solve."""
        finished, episodes, scores = self._solve(verbose=verbose, render=render)
        
        print("FINISHED:", finished)
        if plot:
            plt.figure()
            plt.plot(episodes, scores)
            plt.show()

        return episodes, scores


class CartPoleStandUp(CustomCartPole):
    
    def __init__(self, angle_threshold=12., max_episodes=2000, score_target=195., episodes_threshold=100, max_episode_steps=500):
        
        self.score_target = score_target
        self.episodes_threshold = episodes_threshold
        self.max_episodes = max_episodes

        CustomCartPole.__init__(self, angle_threshold=12., max_episodes=max_episodes, max_episode_steps=max_episode_steps)

    def get_score(self, state, next_state, reward, step_number):
        """This task's reward is simply how many steps it has survived."""
        return step_number

    def reward_on_step(self, state, next_state, reward, done):
        """The reward per step is the default reward for cartpole
        (1 for a complete step) and -1 if it failed.
        """
        return reward if not done else -reward

    def check_solved_on_done(self, state, episodes, scores, verbose=False):
        """The task is solved if the average score over the last self.episodes_threshold 
        episodes averaged to be over the score threshold.
        """
        if len(episodes) < self.episodes_threshold:
            up_to = len(episodes)
        else:
            up_to = self.episodes_threshold
        score = np.mean(scores[-up_to:])
        if verbose:
            print(" - steps ", scores[-1], " - score", int(score), "/", self.score_target)
        if (len(episodes) >= self.episodes_threshold 
            and score > self.score_target):
                return True
        return False
