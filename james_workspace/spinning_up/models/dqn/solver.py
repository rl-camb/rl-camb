import os, random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from collections import deque

from models.standard_agent import StandardAgent

class DQNSolver(StandardAgent):
    """A standard dqn_solver.
    Implements a simple DNN that predicts values.
    """

    def __init__(self, experiment_name, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 1.    # discount rate was 1
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # 0.995
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01
        self.batch_size = 64

        self.model_name = "dqn"
        self.model = DQNModel()
        self.model.compile(loss='mse', 
            optimizer=Adam(lr=self.learning_rate, 
                           decay=self.learning_rate_decay))
        super(DQNSolver, self).__init__(
            self.model_name + "_" + experiment_name)

    def show_example(self, env_wrapper, steps):
        """Show a quick view of the environemt, 
        without trying to solve.
        """
        env = env_wrapper.env
        env.reset()
        for _ in range(steps):
            self.env.render()
            self.env.step(env.action_space.sample())
        env.close()

    def do_random_runs(self, env_wrapper, episodes, steps, verbose=False, wait=0.0):
        """Run some episodes with random actions, stopping on 
        actual failure / win conditions. Just for viewing.
        """
        env = env_wrapper.env
        for i_episode in range(episodes):
            observation = env.reset()
            print("Episode {}".format(i_episode+1))
            for t in range(steps):
                self.env.render()
                if verbose:
                    print(observation)
                # take a random action
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                time.sleep(wait)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        env.close()

    def solve(self, env_wrapper, max_episodes, verbose=False, render=False):
        """A generic solve function (only for DQN agents?).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """
        
        # Keep track of scores for each episode
        episodes, scores = [], []
        env = env_wrapper.env
        
        for episode in range(max_episodes):
            
            # Initialise the environment state
            done, step = False, 0
            state = np.reshape(env.reset(),
                               (1, env_wrapper.observation_space))
            if verbose:
                print("episode", episode, end="")
            
            # Take steps until failure / win
            while not done:

                if render:
                    env.render() # for viewing
                
                # Find the action the agent thinks we should take
                action = self.act(state)

                # Take the action, make observations
                observation, reward, done, info = env.step(action)
                
                # Diagnose your reward function!
                # print("state", state[0,0], "done", done, "thresh", self.env.x_threshold, "angle %.2f" % (state[0,2] * 180 / math.pi))
                
                state_next = np.reshape(observation, (1, env_wrapper.observation_space))

                # Calculate a custom reward - inherit from custom environment
                reward = env_wrapper.reward_on_step(state, state_next, reward, done)
                
                # Save the action into the DQN memory
                self.memory.append((state, action, reward, state_next, done))
                state = state_next
                step += 1

            episodes.append(episode) # a little redundant combined with scores..

            # Calculate a custom score for this episode
            score = env_wrapper.get_score(state, state_next, reward, step)
            self.scores.append(score)

            if env_wrapper.check_solved_on_done(state, episodes, self.scores, verbose=verbose):
                return True, episodes, scores

            self.learn()
            self.save_model()
        
        return False, episodes, scores
    
    def act(self, state):
        """Take a random action or the most valuable predicted
        action, based on the agent's model. 
        """

        # If in exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0]) # returns action
    
    def learn(self):
        """Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.
        """
        x_batch, y_batch = [], []

        minibatch = random.sample(self.memory, 
                                  min(len(self.memory), 
                                      self.batch_size))
        # Process the mini batch
        for state, action, reward, next_state, done in minibatch:
            
            # Get the value of the action you will not take
            y_target = self.model.predict(state)

            # Set the value (or label) for each action action as
            # the predicted value based on the discount rate and 
            # next predicted reward.
            y_target[0][action] = reward if done else reward + self.gamma * \
                         np.amax(self.model.predict(next_state))
            
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        # Batched training
        self.model.fit(np.array(x_batch), 
                       np.array(y_batch), 
                       batch_size=len(x_batch), 
                       verbose=0, 
                       epochs=1)
        
        # Reduce the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """
        self.model.save_weights(self.model_location)

        super(DQNSolver, self).save_dict()

    def load_model(self):
        """Load a model with the specified name"""
        model_dict = super(DQNSolver, self).load_dict()
        if model_dict:
            self.model.load_weights(model_dict["model_location"])


class DQNModel(tf.keras.Model):

    def __init__(self):
        super(DQNModel, self).__init__()
        """Define the network that will instruct the agent on
        predicting the action with the largest future reward, 
        based on present state.
        """
        
        # Model 1
        self.dense1 = Dense(24, input_dim=4, activation='tanh')
        self.dense2 = Dense(48, activation='tanh')
        self.output_layer = Dense(2, activation='linear')

    def call(self, inputs, training=False):

        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.output_layer(x)
