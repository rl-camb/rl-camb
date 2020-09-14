import math, gym
import matplotlib.pyplot as plt
import numpy as np

from . import Env

class CustomCartPole(Env):

    def __init__(self, angle_threshold=12., x_threshold=2.4, max_episode_steps=500, cp_id='CustomCartPole-v1'):

        super().__init__(selected_env='CartPole-v1')

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


class CartPoleStandUp(CustomCartPole):
    
    def __init__(self, angle_threshold=12., score_target=195., episodes_threshold=100, max_episode_steps=500, reward_on_fail=-1.):
        
        self.score_target = score_target
        self.episodes_threshold = episodes_threshold
        self.reward_on_fail = reward_on_fail

        super().__init__(
            angle_threshold=angle_threshold,
            max_episode_steps=max_episode_steps)
        self.kwargs = locals()

    def get_score(self, state, next_state, reward_list, step_number):
        """
        This task's reward is simply how many steps it has survived.
        However some implementations provide a reward list.
        Consider summing this however NOTE: 
          this makes the finishing criterion harder, because the 
          final step has a negative reward
        """
        return step_number

    def reward_on_step(self, state, next_state, reward, done, step, **kwargs):
        """The reward per step is the default reward for cartpole
        (1 for a complete step) and -1 if it failed.
        """
        if done and step < self.max_episode_steps - 1:
            # Done before got to the end
            return self.reward_on_fail
        elif done:
            # It's okay to hit max steps
            return 0.
        else:
            # It's good to stay up
            return reward

    def check_solved_on_done(self, scores, verbose=False):
        """The task is solved if the average score over the last self.episodes_threshold 
        episodes averaged to be over the score threshold.
        """
        if len(scores) < 2:
            return False, 0
        solved = False
        if len(scores) < self.episodes_threshold:
            up_to = len(scores)
        else:
            up_to = self.episodes_threshold
        
        score = np.mean(scores[-up_to:])
        
        if (len(scores) >= self.episodes_threshold 
            and score > self.score_target):
                solved = True
        
        return solved, score
