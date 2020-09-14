import pickle, os, random, pprint

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

class DQNSolver:
    """A standard dqn_solver.
    Implements a simple DNN that predicts values.
    """
    
    def __init__(self, state_size, action_size):
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
        self.model = self._build_model()

        self.models_dict_file = "models/models_dict.pickle"
    
    def _build_model(self):
        """Define the network that will instruct the agent on
        predicting the action with the largest future reward, 
        based on present state.
        """
        
        # Model 1
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', 
                      optimizer=Adam(lr=self.learning_rate, 
                                     decay=self.learning_rate_decay))
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        """
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Take a random action or the most valuable predicted
        action, based on the agent's model. 
        """

        # If in exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0]) # returns action
    
    def experience_replay(self):
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

    def save_model(self, model_name, angle_threshold, x_threshold, training_results, max_episode_steps):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

        model_no = 0
        model_loc = "models/" + model_name + ".h5"

        self.model.save_weights(model_loc)

        if os.path.exists(self.models_dict_file):
            with open(self.models_dict_file, 'rb') as md:
                models_dict = pickle.load(md)
        else:
            models_dict = {}

        models_dict[model_name] =\
            {"model_location": model_loc,
             "angle_threshold": angle_threshold,
             "x_threshold": x_threshold,
             "training_results": training_results,
             "max_episode_steps": max_episode_steps}

        with open(self.models_dict_file, 'wb') as md:
            pickle.dump(models_dict, md)

    def load_model(self, model_name):
        """Load a model with the specified name"""
        with open(self.models_dict_file, 'rb') as md:
            pickled_details = pickle.load(md)
        
        desired_model = pickled_details[model_name]
        self.model.load_weights(desired_model["model_location"])
    
    def view_models_dict(self, view=False):
        """Open the model dict to view what models we have."""
        with open(self.models_dict_file, 'rb') as md:
            model_dict = pickle.load(md)

        if view:
            pprint.pprint(model_dict)

        return model_dict
