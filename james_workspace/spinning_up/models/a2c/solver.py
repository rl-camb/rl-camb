# Inspired by:
#  https://github.com/inoryy/tensorflow2-deep-reinforcement-learning/blob/master/actor-critic-agent-with-tensorflow2.ipynb

import os
import random
import itertools
import sys
import pprint
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from collections import deque
from models.standard_agent import StandardAgent

tf.keras.backend.set_floatx('float64')


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(
            tf.random.categorical(logits, 1),
            axis=-1)


# TODO try the same model as the other tasks (or make those bigger too)
class A2CModel(tf.keras.Model):

    def __init__(self, input_shape, num_actions):

        super().__init__('mlp_policy')

        self.hidden1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')

        # Logits are unnormalized log probabilities.
        self.logits_layer = tf.keras.layers.Dense(num_actions, name='policy_logits')
        self.value_layer = tf.keras.layers.Dense(1, name='value')

        self.dist = ProbabilityDistribution()
        self.dist.build((None, input_shape))

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)

        return (
            self.logits_layer(hidden_logs), 
            self.value_layer(hidden_vals)
        )

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)

        return (
            np.squeeze(action, axis=-1), 
            np.squeeze(value, axis=-1)
        )


class A2CSolver(StandardAgent):
    """
    A standard a2c solver:
      https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
    Implements a simple DNN that predicts values.
    """

    def __init__(self, 
        experiment_name,
        state_size,
        action_size, 
        ent_coef=1e-4,
        vf_coef=0.5,
        batch_size=64,
        max_grad_norm=0.5,
        learning_rate=7e-4,
        gamma=0.99,
        lrschedule='linear',
        model_name="a2c",
        saving=True,):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # TODO reimplement
        # self.max_grad_norm = max_grad_norm
        # self.epsilon = epsilon  # exploration rate

        self.solved_on = None

        self.model = A2CModel(self.state_size, self.action_size)
        self.model.compile(
            optimizer=RMSprop(lr=learning_rate),
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])
        self.model.build((None, self.state_size))
        self.model.summary()

        super(A2CSolver, self).__init__(
            model_name + "_" + experiment_name, 
            saving=saving
        )
        
        self.load_state()

    def solve(self, env_wrapper, max_episodes, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env = env_wrapper.env

        # TODO - make it a memory deque or so
        actions = np.empty((self.batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + (self.state_size,))

        next_obs = env.reset()
        ep_rewards = [0.0]
        success_steps = 0

        for episode in range(max_episodes):
            for step in range(self.batch_size):  # itertools.count():
                if render:
                    env.render()
                
                observations[step] = next_obs.copy()
                # Calculate value model of the currebt state
                actions[step], values[step] =\
                    self.model.action_value(next_obs[None, :])
                # action = self.act(self.model, state, epsilon=self.epsilon)
                next_obs, reward, dones[step], _ = env.step(actions[step])

                # Custom reward if required by env wrapper
                rewards[step] = env_wrapper.reward_on_step(
                    observations[step], next_obs, reward, dones[step], step)

                ep_rewards[-1] += rewards[step]

                self.report_step(step, episode, max_episodes)

                if dones[step]:
                    last_ep_steps = success_steps
                    success_steps = 0
                    ep_rewards.append(0.)
                    next_obs = env.reset()
                else:
                    success_steps += 1

            _, next_value = self.model.action_value(next_obs[None, :])
            # A2C - advantage is actual return - predicted reward
            returns, advs = self._returns_advantages(
                rewards, dones, values, next_value)
            acts_and_advs = np.concatenate(
                [actions[:, None], advs[:, None]], axis=-1)
            loss_value = self.model.train_on_batch(
                observations, [acts_and_advs, returns])

            score = last_ep_steps
            # OR env_wrapper.get_score(state, state_next, reward, step)
            self.scores.append(score)

            solved = self.handle_episode_end(
                env_wrapper, observations[-1], next_obs, rewards[step], 
                step, max_episodes, verbose=verbose)

            if solved:
                break

        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def _returns_advantages(self, rewards, dones, values, next_value):

        # TODO why is axis -1 not working? next value not an array?
        returns = np.append(np.zeros_like(rewards), next_value) # , axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = (
                rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t]))
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, values):
        
        return (
            self.vf_coef 
            * tf.keras.losses.MSE(returns, values))

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(
            actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.ent_coef * entropy_loss

    def save_state(self):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            # "epsilon": self.epsilon,
            # "optimizer": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)

        self.model.save_weights(self.model_location)

    def load_state(self):
        """Load a model with the specified name"""

        model_dict = self.load_state_from_dict()

        print("Loading weights from", self.model_location + "...", end="")
        if os.path.exists(self.model_location):
            self.model.load_weights(self.model_location)
            # self.optimizer = self.optimizer.from_config(self.optimizer_config)
            # del model_dict["optimizer_config"], self.optimizer_config
            print(" Loaded.")
        else:
            print(" Model not yet saved at loaction.")

        if "memory" in model_dict:
            del model_dict["memory"]

        print("Loaded state:")
        pprint.pprint(model_dict, depth=1)
