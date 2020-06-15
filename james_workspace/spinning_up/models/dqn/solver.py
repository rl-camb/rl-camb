import os
import random
import itertools
import sys
import pprint
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from collections import deque
from models.standard_agent import StandardAgent


class DQNSolver(StandardAgent):
    """
    A standard dqn_solver, inpired by:
      https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
    Implements a simple DNN that predicts values.
    """

    def __init__(self, 
        experiment_name,
        state_size,
        action_size, 
        memory_len=100000,
        gamma=0.99,
        batch_size=64,
        epsilon=1.,
        epsilon_min=0.01,
        epsilon_decay=0.995, 
        learning_rate=0.01,
        learning_rate_decay=0.01,
        model_name="dqn",
        saving=True):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate was 1
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay # 0.995
        self.batch_size = batch_size
        self.model_name = model_name

        self.memory = deque(maxlen=memory_len)
        self.solved_on = None

        self.model = self.build_model()

        self.optimizer = Adam(
            lr=learning_rate, 
            decay=learning_rate_decay)

        super(DQNSolver, self).__init__(
            self.model_name + "_" + experiment_name, 
            saving=saving)
        self.load_state()

    def build_model(self):

        tf.keras.backend.set_floatx('float64')

        model = tf.keras.Sequential(name=self.model_name)
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))

        model.build()

        return model

    def show(self, env_wrapper, verbose=False, render=False):
        raise NotImplementedError(
            "TODO - implement a show method that runs the agent with "
            "act(epsilon=0)")

    def solve(self, env_wrapper, max_episodes, verbose=False, render=False):
        env = env_wrapper.env
        start_time = datetime.datetime.now()
        for episode in range(max_episodes):
            state = env.reset()
            # Take steps until failure / win
            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(state, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(action)
                state_next = observation
                # Custom reward if required by env wrapper
                reward = env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                self.memory.append(
                    (state, np.int32(action), reward, state_next, done))
                state = observation
                
                print(f"\rEpisode {episode + 1}/{max_episodes} - "
                      f"steps {step} ({self.total_t + 1})", 
                      end="")
                sys.stdout.flush()

                self.total_t += 1
                if done:
                    break

            self.learn()
            
            # Calculate a (optionally custom) score for this episode
            score = env_wrapper.get_score(state, state_next, reward, step)
            self.scores.append(score) 

            solved, agent_score = env_wrapper.check_solved_on_done(
                state, self.scores, verbose=verbose)

            if episode % 25 == 0 or solved:
                print(f"\rEpisode {episode + 1}/{max_episodes} "
                      f"- steps {step} - score {int(agent_score)}/"
                      f"{int(env_wrapper.score_target)}")
                if self.saving:
                    self.save_state()

            if solved:
                self.solved_on = episode
                break
        
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def act(self, state, epsilon=0.0):
        """Take a random action or the most valuable predicted
        action, based on the agent's model. 
        """
        if (state.shape != (self.state_size,) 
                and state.shape != (1, self.state_size)):
            raise NotImplementedError(
                "Not intended for use on batch state; returns integer")

        # If in exploration
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)

        if state.ndim == 1:
            state = tf.reshape(state, (1, self.state_size))
        act_values = self.model(state)

        return tf.math.argmax(act_values, axis=-1).numpy()[0]
    
    def learn(self):
        """Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.
        """
        
        minibatch_i = np.random.choice(len(self.memory),
            min(self.batch_size, len(self.memory)),
            )
        
        minibatch = [self.memory[i] for i in minibatch_i]

        loss_value = self.take_training_step(
            *tuple(map(tf.convert_to_tensor, zip(*minibatch)))
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @tf.function
    def take_training_step(self, sts, a, r, n_sts, d):

        future_q_pred = tf.math.reduce_max(self.model(n_sts), axis=-1)
        future_q_pred = tf.where(d, 
            tf.zeros((1,), dtype=tf.dtypes.float64), 
            future_q_pred)
        q_targets = tf.cast(r, tf.float64) + self.gamma * future_q_pred

        loss_value, grads = self.squared_diff_loss_at_a(sts, a, q_targets)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        
        return loss_value

    @tf.function
    def squared_diff_loss_at_a(self, sts, a, q_next):
        """
        A squared difference loss function 
        Diffs the Q model's predicted values for a state with 
        the actual reward + predicted values for the next state
        """
        with tf.GradientTape() as tape:
            q_s = self.model(sts)  # Q(st)
            # Take only predicted value of the action taken for Q(st|at)
            gather_indices = tf.range(a.shape[0]) * tf.shape(q_s)[-1] + a
            q_s_a = tf.gather(tf.reshape(q_s, [-1]), gather_indices)

            # Q(st|at) diff Q(st+1)
            losses = tf.math.squared_difference(q_s_a, q_next)
            reduced_loss = tf.math.reduce_mean(losses)

        return (reduced_loss, 
                tape.gradient(reduced_loss, self.model.trainable_variables))

    def save_state(self):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

        add_to_save = {
            "epsilon": self.epsilon,
            "memory": self.memory,
            "optimizer_config": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)

        self.model.save(self.model_location)

    def load_state(self):
        """Load a model with the specified name"""

        model_dict = self.load_state_from_dict()

        print("Loading weights from", self.model_location + "...", end="")
        
        if os.path.exists(self.model_location):
            self.model = tf.keras.models.load_model(self.model_location)

            self.optimizer = self.optimizer.from_config(self.optimizer_config)
            del model_dict["optimizer_config"], self.optimizer_config

            print(" Loaded.")
        
        else:
            print(" Model not yet saved at loaction.")

        if "memory" in model_dict:
            del model_dict["memory"]

        print("Loaded state:")
        pprint.pprint(model_dict, depth=1)
