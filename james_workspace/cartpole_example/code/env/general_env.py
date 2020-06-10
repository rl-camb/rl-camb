import gym, time
import numpy as np

from gym import spaces, envs

from dqn_solver import DQNSolver

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

        # The agent
        self.dqn_solver = DQNSolver(self.observation_space, self.action_space)

    def get_spaces(self, registry=True):
        """Print out the action and observation spaces for viewing."""
        if registry:
            print("Registry:\n", envs.registry.all())
        print("Action space:", self.env.action_space)
        #> Discrete(2)
        print("Obs space:", self.env.observation_space)
        #> Box(4,)

    def show_example(self, steps):
        """Show a quick view of the environemt, 
        without trying to solve.
        """
        self.env.reset()
        for _ in range(steps):
            self.env.render()
            self.env.step(env.action_space.sample())
        self.env.close()

    def do_random_runs(self, episodes, steps, verbose=False, wait=0.0):
        """Run some episodes with random actions, stopping on 
        actual failure / win conditions. Just for viewing.
        """
        for i_episode in range(episodes):
            observation = self.env.reset()
            print("Episode {}".format(i_episode+1))
            for t in range(steps):
                self.env.render()
                if verbose:
                    print(observation)
                # take a random action
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                time.sleep(wait)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        self.env.close()

    def _solve(self, verbose=False, wait=0.0, render=True):
        """A generic solve function (only for DQN agents?).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """
        
        # Keep track of scores for each episode
        episodes, scores = [], []
        
        for episode in range(self.max_episodes):
            
            # Initialise the environment state
            done, step = False, 0
            state = np.reshape(self.env.reset(),
                               (1, self.observation_space))
            if verbose:
                print("episode", episode, end="")
            
            # Take steps until failure / win
            while not done:

                if render:
                    self.env.render() # for viewing
                
                # Find the action the agent thinks we should take
                action = self.dqn_solver.act(state)

                # Take the action, make observations
                observation, reward, done, info = self.env.step(action)
                
                # Diagnose your reward function!
                # print("state", state[0,0], "done", done, "thresh", self.env.x_threshold, "angle %.2f" % (state[0,2] * 180 / math.pi))
                
                state_next = np.reshape(observation, (1, self.observation_space))

                # Calculate a custom reward - inherit from custom environment
                reward = self.reward_on_step(state, state_next, reward, done)
                
                # Save the action into the DQN memory
                self.dqn_solver.remember(state, action, reward, state_next, done)
                state = state_next
                step += 1

            episodes.append(episode) # a little redundant combined with scores..

            # Calculate a custom score for this episode
            score = self.get_score(state, state_next, reward, step)
            scores.append(score)

            if self.check_solved_on_done(state, episodes, scores, verbose=verbose):
                return True, episodes, scores

            self.dqn_solver.experience_replay()
            # print("reward", reward)
        
        return False, episodes, scores

    def get_score(self, state, next_state, reward, step):
        """Returns the score for an episode"""
        raise NotImplementedException("Should be overridden")

    def reward_on_step(self, state, next_state, reward, done):
        """Returns the reward for a step within an episode"""
        raise NotImplementedException("Should be overridden")
    
    def check_solved_on_done(self, state, episodes, scores, verbose=False):
        """Checks if the task was completed by this episode"""
        raise NotImplementedException("This method should be overridden")

