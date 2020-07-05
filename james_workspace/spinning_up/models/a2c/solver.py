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

from utils import ProbabilityDistribution, get_batch_from_memory

tf.keras.backend.set_floatx('float64')


class A2CModel(tf.keras.Model):

    def __init__(self, input_shape, num_actions, model_name='mlp_policy'):

        super().__init__(model_name)

        self.prob_1 = tf.keras.layers.Dense(24, activation='tanh')
        self.prob_2 = tf.keras.layers.Dense(48, activation='tanh')

        self.val_1 = tf.keras.layers.Dense(24, activation='tanh')
        self.val_2 = tf.keras.layers.Dense(48, activation='tanh')

        # Logits are unnormalized log probabilities.
        self.logits_layer = tf.keras.layers.Dense(num_actions, name='policy_logits')
        self.value_layer = tf.keras.layers.Dense(1, name='value')

        self.dist = ProbabilityDistribution()
        self.dist.build((None, input_shape))

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.prob_1(x)
        hidden_logs = self.prob_2(hidden_logs)

        hidden_vals = self.val_1(x)
        hidden_vals = self.val_2(hidden_vals)

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

    def __init__(
        self, 
        experiment_name,
        env_wrapper,
        ent_coef=1e-4,
        vf_coef=0.5,
        batch_size=64,
        n_cycles=128,
        max_grad_norm=0.5,
        learning_rate=7e-4,
        gamma=0.99,
        lrschedule='linear',
        model_name="a2c",
        saving=True,):

        super(A2CSolver, self).__init__(
            env_wrapper,
            model_name,
            experiment_name, 
            saving=saving
        )

        self.batch_size = batch_size
        self.n_cycles = n_cycles

        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # TODO reimplement
        # self.max_grad_norm = max_grad_norm
        # self.epsilon = epsilon  # exploration rate

        self.solved_on = None
        self.memory = []

        self.optimizer = RMSprop(lr=learning_rate)

        self.model = A2CModel(self.state_size, self.action_size, model_name=model_name)
        self.model.compile(
            optimizer=self.optimizer,
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])
        self.model.build((None, self.state_size))
        self.model.summary()
        
        self.load_state()

    def solve(self, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env = self.env_wrapper.env

        actions = np.empty((self.n_cycles,), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.n_cycles))
        observations = np.empty((self.n_cycles,) + (self.state_size,))
        ep_rewards = [0.0]

        next_obs = env.reset()
        success_steps = 0

        for iteration in range(max_iters):
            for step in range(self.n_cycles):  # itertools.count():
                if render:
                    env.render()
                
                observations[step] = next_obs.copy()
                # Calculate value model of the currebt state
                actions[step], values[step] =\
                    self.model.action_value(next_obs[None, :])
                # action = self.act(self.model, state, epsilon=self.epsilon)
                next_obs, reward, dones[step], _ = env.step(actions[step])

                # Custom reward if required by env wrapper
                rewards[step] = self.env_wrapper.reward_on_step(
                    observations[step], next_obs, reward, dones[step], step)

                ep_rewards[-1] += rewards[step]

                self.report_step(step, iteration, max_iters)

                if dones[step]:
                    # OR env_wrapper.get_score(state, state_next, reward, step)
                    self.scores.append(success_steps)
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

            self.remember(  # Copy because np array
                observations.copy(), 
                acts_and_advs.copy(), 
                returns.copy()
            )
            loss_value = self.model.train_on_batch(
                *self.get_batch_to_train()
            )

            solved = self.handle_episode_end(
                observations[-1], next_obs, rewards[step], 
                step, max_iters, verbose=verbose)

            if solved:
                break

        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def remember(self, obs, acts_advs, rets):

        self.memory = [obs, [acts_advs, rets]]

    def get_batch_to_train(self):

        mem = self.memory
        self.memory = []
        return mem

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

    def save_state(self, add_args={}):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            # "epsilon": self.epsilon,
            "optimizer_config": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict={**add_args, **add_to_save})

        self.model.save_weights(self.model_location)

    def load_state(self):
        """Load a model with the specified name"""

        model_dict = self.load_state_from_dict()

        print("Loading weights from", self.model_location + "...", end="")
        if os.path.exists(self.model_location):
            self.model.load_weights(self.model_location)
            self.optimizer = self.optimizer.from_config(self.optimizer_config)
            del model_dict["optimizer_config"], self.optimizer_config
            print(" Loaded.")
        else:
            print(" Model not yet saved at loaction.")

        if "memory" in model_dict:
            del model_dict["memory"]

        print("Loaded state:")
        pprint.pprint(model_dict, depth=1)


# TODO make a batched learning version (create a standard wrapper that inherits whichever?)
class A2CSolverBatch(A2CSolver):

    def __init__(
        self,
        experiment_name,
        env_wrapper,
        ent_coef=1e-4,
        vf_coef=0.5,
        batch_size=64,
        n_cycles=128,
        max_grad_norm=0.5,
        learning_rate=7e-4,
        gamma=0.99,
        lrschedule='linear',
        model_name="a2c_batch",
        saving=True,
        maxlen=10000,
        rollout_steps=5000,):

        super(A2CSolverBatch, self).__init__(
            experiment_name,
            env_wrapper,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            batch_size=batch_size,
            n_cycles=n_cycles,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            gamma=gamma,
            lrschedule=lrschedule,
            model_name=model_name,
            saving=saving,
            )
        self.memory = deque(maxlen=maxlen)
        self.rollout_memory(rollout_steps - len(self.memory))

    def remember(self, obs, acts_advs, rets):

        assert len(obs) == len(acts_advs) and len(obs) == len(rets)
        for i in range(len(obs)):
            self.memory.append((obs[i], acts_advs[i], rets[i]))

    def get_batch_to_train(self):

        obs, acts_advs, rets =\
            get_batch_from_memory(self.memory, self.batch_size)

        return obs, [acts_advs, rets]

    def save_state(self):

        super().save_state(
            add_args = {
                "memory": self.memory
            }
        )

    # TODO way to add random actions to A2C? Need the value too..
    def rollout_memory(self, rollout_steps, render=False):

        if rollout_steps <= 0:
            return

        env = self.env_wrapper.env
        next_obs = env.reset()
        print("Rolling out further steps", rollout_steps)

        actions = np.empty((self.n_cycles,), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.n_cycles))
        observations = np.empty((self.n_cycles,) + (self.state_size,))
        ep_rewards = [0.0]

        iters = rollout_steps // self.n_cycles
        for iteration in range(iters):
            for step in range(self.n_cycles):
                if render:
                    env.render()

                actions[step], values[step] =\
                    self.model.action_value(next_obs[None, :])
                
                next_obs, reward, dones[step], _ = env.step(actions[step])

                # Custom reward if required by env wrapper
                rewards[step] = self.env_wrapper.reward_on_step(
                    observations[step], next_obs, reward, dones[step], step)

                ep_rewards[-1] += rewards[step]

                self.report_step(step, iteration, iters)

                if dones[step]:
                    ep_rewards.append(0.)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(
                rewards, dones, values, next_value)

            acts_and_advs = np.concatenate(
                [actions[:, None], advs[:, None]], axis=-1)

            self.remember(  # Copy because np array
                observations.copy(), 
                acts_and_advs.copy(), 
                returns.copy()
            )
    print("Complete")
