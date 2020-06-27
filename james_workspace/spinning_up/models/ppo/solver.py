# https://github.com/ajleite/basic-ppo/blob/master/ppo.py

import os
import copy
import random
import itertools
import sys
import pprint
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from collections import deque
from models.standard_agent import StandardAgent

from utils import conditional_decorator


# TODO - use it
class EnvTracker():
    """
    A class that can preserve a half-run environment
    Has an 
    """

    def __init__(self, env_wrapper):

        self.env = copy.deepcopy(env_wrapper.env)
        self.latest_state = self.env.reset()
        self.return_so_far = 0.
        self.steps_so_far = 0


# TODO change it all to PPO
class PPOSolver(StandardAgent):
    """
    PPO Solver
    Inspired by:
      https://github.com/ajleite/basic-ppo/blob/master/ppo.py
      https://github.com/anita-hu/TF2-RL/blob/master/PPO/TF2_PPO.py
    """

    can_graph = True

    # TODO make env wrapper self state for all agents
    def __init__(self, 
        experiment_name,
        state_size,
        action_size,
        epsilon=0.2,
        gamma=0.95,
        actors=1,
        cycle_length=128,
        # batch_size=64,
        minibatch_size_per_actor=64,
        cycle_epochs=4,
        # num_random_action_selection=4096,
        learning_rate=5e-4,
        model_name="ppo",
        saving=True):

        # TODO clean up and get from env wrapper
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.gamma = gamma

        self.actors = actors
        self.cycle_length = cycle_length  # Run this many per epoch
        self.batch_size = cycle_length * actors  # Sample from the memory
        self.minibatch_size = minibatch_size_per_actor * actors  # train on batch
        self.cycle_epochs = cycle_epochs  # Train for this many epochs

        # self.num_init_random_rollouts = num_init_random_rollouts
        self.model_name = model_name

        self.solved_on = None

        self.pi_model = self.build_model(
            model_name=self.model_name + "_pi")
        self.pi_model_old = self.build_model(
            model_name=self.model_name + "_old_pi")
        self.value_model = self.build_model(
            output_size=1,  # map to single value
            model_name=self.model_name + "_value"
        )

        # self._random_dataset = self._gather_rollouts(
        #     env_wrapper, num_init_random_rollouts, epsilon=1.)

        self.optimizer = Adam(lr=learning_rate)

        super(PPOSolver, self).__init__(
            self.model_name + "_" + experiment_name, 
            saving=saving)
        self.load_state()
    
    """
    def _gather_rollouts(self, env_wrapper, num_rollouts, epsilon=None, render=False):
        env = env_wrapper.env
        state = env.reset()
        memory = deque(maxlen=self.max_rollout_len)
        for _ in range(num_rollouts):
            done = False
            for t in itertools.count():
                if render:
                    env.render()
                action = self.act(self.model, state, epsilon=epsilon)
                next_state, reward, done, _ = env.step(action)
                memory.append((state, action, next_state, reward, done))
                state = next_state
                if done or len(memory) >= self.max_rollout_length:
                    break
        return memory
    """

    def show(self, env_wrapper):
        raise NotImplementedError("self.model needs to be adapted in super")

    def solve(self, env_wrapper, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env_trackers = [EnvTracker(env_wrapper) for _ in range(self.actors)]
        solved = False

        # Every episode return ever
        all_episode_returns = []
        all_episode_steps = []

        for iteration in range(max_iters):
            memory = []

            for env_tracker in env_trackers:
                state = env_tracker.latest_state

                for step in range(self.cycle_length):
                    if render:
                        env.render()

                    action = self.act(self.pi_model, state, epsilon=None)
                    observation, reward, done, _ = env_tracker.env.step(action)
                    state_next = observation

                    # Custom reward if required by env wrapper
                    reward = env_wrapper.reward_on_step(
                        state, state_next, reward, done, step)

                    env_tracker.return_so_far += reward

                    memory.append(
                        (state, np.int32(action), np.float64(reward), 
                         state_next, done)
                    )

                    self.report_step(step, iteration, max_iters)
                    if done:
                        all_episode_returns.append(
                            env_tracker.return_so_far)
                        all_episode_steps.append(env_tracker.steps_so_far)
                        state = env_tracker.env.reset()
                        env_tracker.steps_so_far = 0
                        env_tracker.return_so_far = 0.
                    else:
                        env_tracker.steps_so_far += 1
                        state = observation

                env_tracker.latest_state = state

            self.scores = all_episode_steps  # FIXME this won't handle picking up from left-off
            solved = self.handle_episode_end(
                env_wrapper, state, state_next, reward, 
                step, max_iters, verbose=verbose)
            if solved: 
                break

            self.take_training_step(
                *tuple(map(tf.convert_to_tensor, zip(*memory)))
            )
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    @conditional_decorator(tf.function, can_graph)
    def take_training_step(self, sts, a, r, n_sts, d):
        """
        Performs gradient DEscent on minibatches of minibatch_size, 
        sampled from a batch of batch_size, sampled from the memory
        """

        for _ in range(self.cycle_epochs):
            # Batch from the examples in the memory
            shuffled_indices = tf.random.shuffle(tf.range(self.batch_size))
            num_mb = self.batch_size // self.minibatch_size
            # Pick minibatch-sized samples from there
            for minibatch_i in tf.split(shuffled_indices, num_mb):
                minibatch = (
                    tf.gather(x, minibatch_i) for x in (sts, a, r, n_sts, d)
                )
                self.train_minibatch(*minibatch)

        # TODO used to be zip weights and assign
        for pi_old_w, pi_w in zip(
                self.pi_model_old.weights, self.pi_model.weights):
            pi_old_w.assign(pi_w)
    
    @conditional_decorator(tf.function, can_graph)
    def train_minibatch(self, sts, a, r, n_sts, d):

        next_value = tf.stop_gradient(
            tf.where(
                d, 
                tf.zeros(d.shape, dtype=tf.float64),
                self.value_model(n_sts)
            )
        )
        advantage = r + self.gamma * next_value - self.value_model(sts)

        # Update value model
        value_loss = tf.reduce_sum(advantage ** 2)
        value_gradient = tf.gradients(value_loss, self.value_model.weights)
        self.optimizer.apply_gradients(
            zip(value_gradient, self.value_model.weights))
        
        # Update policy model
        ratio = (
            tf.gather(self.pi_model(sts), a, axis=1) 
            / tf.gather(self.pi_model_old(sts), a, axis=1)
        )
        confident_ratio = tf.clip_by_value(
            ratio, 1 - self.epsilon, 1 + self.epsilon)

        # apply a filter (trust region?) above and below epsilon
        current_objective = ratio * advantage  # weight
        confident_objective = confident_ratio * advantage

        ppo_objective = tf.reduce_mean(
            tf.where(
                current_objective < confident_objective,
                current_objective,
                confident_objective
            )
        )

        pi_gradient = tf.gradients(-ppo_objective, self.pi_model.weights)

        self.optimizer.apply_gradients(
            zip(pi_gradient, self.pi_model.weights))

    def save_state(self):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            "epsilon": self.epsilon,
            # "memory": self.memory,
            "optimizer_config": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)

        self.pi_model.save(self.model_location)

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
