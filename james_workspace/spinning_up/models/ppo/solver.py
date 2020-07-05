# https://github.com/ajleite/basic-ppo/blob/master/ppo.py

import os
import sys
import copy
import random
import itertools
import pprint
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from collections import deque
from models.standard_agent import StandardAgent

from utils import conditional_decorator, EnvTracker


class PPOModel(tf.keras.Model):

    def __init__(self, input_shape, num_actions, model_name='mlp_policy'):

        super(PPOModel, self).__init__(model_name)
        self.num_actions = num_actions

        # VALUE
        self.L_v1 = Dense(24, activation='tanh', 
            input_shape=(None, input_shape))
        self.L_v2 = Dense(48, activation='tanh')
        self.L_vout = Dense(1, name='value')

        # PI
        self.L_p1 = Dense(24, activation='tanh',
            input_shape=(None, input_shape))
        self.L_p2 = Dense(48, activation='tanh')
        self.L_pout = Dense(self.num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)

        value = self.L_v1(x)
        value = self.L_v2(value)
        value = self.L_vout(value)

        policy_logits = self.L_p1(x)
        policy_logits = self.L_p2(policy_logits)
        policy_logits = self.L_pout(policy_logits)

        return policy_logits, value

    def act_value_logprobs(self, state, eps=None, test=False):
        if eps is not None:
            raise NotImplementedError("Need to implement epsilon-randomness")

        state = tf.cast(
            tf.expand_dims(state, axis=0), 
            dtype=tf.dtypes.float64
        )

        logits, value = self.predict_on_batch(state)
        prob_dist = tfp.distributions.Categorical(probs=tf.nn.softmax(logits))

        action = (
            tf.math.argmax(logits, axis=-1) if test
            else prob_dist.sample()  # TODO put some eps random in, too
        )

        log_probs = prob_dist.log_prob(action)

        return action, value, log_probs

    def evaluate_actions(self, state, action):

        logits, value = self.predict_on_batch(state)
        prob_dist = tfp.distributions.Categorical(probs=tf.nn.softmax(logits))

        log_probs = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()

        return log_probs, entropy, value


class PPOSolver(StandardAgent):
    """
    PPO Solver
    Inspired by:
      https://github.com/anita-hu/TF2-RL/blob/master/PPO/TF2_PPO.py
      https://github.com/ajleite/basic-ppo/blob/master/ppo.py
    """

    can_graph = True

    def __init__(self, 
        experiment_name,
        env_wrapper,
        clip_ratio=0.2,
        val_coef=1.0,
        entropy_coef=0.01,
        lam=1.0,
        gamma=0.95,
        actors=1,
        cycle_length=128,
        minibatch_size_per_actor=64,
        cycle_epochs=4,
        learning_rate=5e-4,
        model_name="ppo",
        saving=True):

        super(PPOSolver, self).__init__(
            env_wrapper,
            model_name,
            experiment_name, 
            saving=saving)

        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.val_coef = val_coef
        self.entropy_coef = entropy_coef

        self.actors = actors
        self.cycle_length = cycle_length  # Run this many per epoch
        self.batch_size = cycle_length * actors  # Sample from the memory
        self.minibatch_size = minibatch_size_per_actor * actors  # train on batch
        self.cycle_epochs = cycle_epochs  # Train for this many epochs

        # self.num_init_random_rollouts = num_init_random_rollouts
        self.model_name = model_name

        self.solved_on = None

        self.model = PPOModel(
            self.state_size, self.action_size, model_name=self.model_name)
        self.model.build(input_shape=(None, self.state_size))

        # self._random_dataset = self._gather_rollouts(
        #     env_wrapper, num_init_random_rollouts, epsilon=1.)

        self.optimizer = Adam(lr=learning_rate)

        head, _, _ = self.model_location.rpartition(".h5")
        self.model_location = head + ".weights"
        self.load_state()

    def show(self, render=False):
        raise NotImplementedError("self.model needs to be adapted in super")

    def solve(self, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env_trackers = [EnvTracker(self.env_wrapper) for _ in range(self.actors)]
        solved = False

        # Every episode return ever
        all_episode_returns = []
        all_episode_steps = []

        for iteration in range(max_iters):
            data = []

            for env_tracker in env_trackers:
                state = env_tracker.latest_state
                states, actions, log_probs, rewards, v_preds =\
                    [], [], [], [], []

                for step in range(self.cycle_length):
                    if render:
                        env_tracker.env.render()

                    action, value, log_prob = (
                        tf.squeeze(x).numpy() for x in
                        self.model.act_value_logprobs(
                            state, 
                            eps=None)
                    )
                    observation, reward, done, _ = env_tracker.env.step(action)
                    state_next = observation

                    # Custom reward if required by env wrapper
                    reward = self.env_wrapper.reward_on_step(
                        state, state_next, reward, done, step)

                    env_tracker.return_so_far += reward

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(np.float64(reward))
                    v_preds.append(value)

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

                next_v_preds = v_preds[1:] + [0.]  # TODO - both right float?
                gaes = self.get_norm_general_advantage_est(
                    rewards, v_preds, next_v_preds)

                # TODO make a handler object
                if not data:
                    data = [
                        states, actions, log_probs, next_v_preds, rewards, 
                        gaes
                    ]
                else:
                    data[0] += states; data[1] += actions; data[2] += log_probs
                    data[3] += next_v_preds; data[4] += rewards; data[5] += gaes

                env_tracker.latest_state = state

            self.scores = all_episode_steps  # FIXME this won't handle picking up from left-off
            solved = self.handle_episode_end(
                state, state_next, reward, 
                step, max_iters, verbose=verbose)
            if solved: 
                break

            self.take_training_step(
                *(tf.convert_to_tensor(lst) for lst in data)
                # *tuple(map(tf.convert_to_tensor, zip(*memory)))
            )
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def get_norm_general_advantage_est(self, rewards, v_preds, next_v_preds):
        # Sources:
        #  https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        #  https://github.com/anita-hu/TF2-RL/blob/master/PPO/TF2_PPO.py
        deltas = [
            r_t + self.gamma * v_next - v for r_t, v_next, v in 
            zip(rewards, next_v_preds, v_preds)
        ]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]

        gaes = np.array(gaes).astype(np.float64)
        norm_gaes = (gaes - gaes.mean()) / gaes.std()

        return norm_gaes

    @conditional_decorator(tf.function, can_graph)
    def take_training_step(self, sts, a, log_p, nxt_v_pred, r, adv):
        """
        Performs gradient DEscent on minibatches of minibatch_size, 
        sampled from a batch of batch_size, sampled from the memory

        Samples without replacement (to check)
        """

        assert self.batch_size == len(r)

        for _ in range(self.cycle_epochs):
            # Batch from the examples in the memory
            shuffled_indices = tf.random.shuffle(tf.range(self.batch_size))  # Every index of the cycle examples
            num_mb = self.batch_size // self.minibatch_size
            # Pick minibatch-sized samples from there
            for minibatch_i in tf.split(shuffled_indices, num_mb):
                minibatch = (
                    tf.gather(x, minibatch_i, axis=0) 
                    for x in (sts, a, log_p, nxt_v_pred, r, adv)
                )
                self.train_minibatch(*minibatch)

        # TODO used to be zip weights and assign
        # for pi_old_w, pi_w in zip(
        #         self.pi_model_old.weights, self.pi_model.weights):
        #     pi_old_w.assign(pi_w)
    
    @conditional_decorator(tf.function, can_graph)
    def train_minibatch(self, sts, a, log_p, nxt_v_pred, r, adv):
       
        # Convert from (64,) to (64, 1)
        r = tf.expand_dims(r, axis=-1)
        nxt_v_pred = tf.expand_dims(nxt_v_pred, axis=-1)

        with tf.GradientTape() as tape:
            new_log_p, entropy, sts_vals = self.model.evaluate_actions(sts, a)
            ratios = tf.exp(new_log_p - log_p)

            clipped_ratios = tf.clip_by_value(
                ratios, 
                clip_value_min=1-self.clip_ratio, 
                clip_value_max=1+self.clip_ratio
            )
            loss_clip = tf.reduce_mean(
                tf.minimum((adv  * ratios), (adv * clipped_ratios))
            )
            target_values = r + self.gamma * nxt_v_pred

            vf_loss = tf.reduce_mean(
                tf.math.square(sts_vals - target_values)
            )

            entropy = tf.reduce_mean(entropy)

            total_loss = ( 
                - loss_clip 
                + self.val_coef * vf_loss 
                - self.entropy_coef * entropy
            )
        train_variables = self.model.trainable_variables
        grads = tape.gradient(total_loss, train_variables)
        self.optimizer.apply_gradients(zip(grads, train_variables))

    def save_state(self, verbose=False):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            # "epsilon": self.epsilon,
            # "memory": self.memory,
            "optimizer_config": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)

        if verbose:
            print("Saving to", self.model_location)

        self.model.save_weights(self.model_location) # , save_format='tf')

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
