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

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Concatenate

from collections import deque
from models.standard_agent import StandardAgent

from utils import ProbabilityDistribution

tf.keras.backend.set_floatx('float64')


class DDPGSolver(StandardAgent):
    """
    A standard ddpg solver:
      https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
    Inspired by
      https://github.com/anita-hu/TF2-RL/blob/master/DDPG/TF2_DDPG_Basic.py
    """

    def __init__(self, 
        experiment_name,
        env_wrapper,
        ent_coef=1e-4,
        vf_coef=0.5,
        n_cycles=128,
        batch_size=64,
        max_grad_norm=0.5,
        learning_rate_actor=1e-5,
        learning_rate_critic=1e-3,
        memory_len=100000,
        gamma=0.99,
        epsilon=None,
        tau=0.125,
        lrschedule='linear',
        model_name="ddpg",
        saving=True,
        rollout_steps=5000,):

        super(DDPGSolver, self).__init__(
            env_wrapper,
            model_name,
            experiment_name,
            saving=saving
        )

        self.n_cycles = n_cycles
        self.batch_size = batch_size

        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # NOTE new AND need to verify deque is safe
        self.memory = deque(maxlen=memory_len)
        self.epsilon = epsilon # new but should be in A2C
        self.tau = tau

        # TODO reimplement
        # self.max_grad_norm = max_grad_norm
        # self.epsilon = epsilon  # exploration rate

        self.solved_on = None

        self.actor = self.build_model(model_name=model_name + "_actor")
        self.actor.build(input_shape=(None, self.state_size,))
        
        self.actor_dash = self.build_model(model_name=model_name + "_actor_target")
        self.actor_dash.build(input_shape=(None, self.state_size,))
        
        self.actor_dash.set_weights(self.actor.get_weights())
        
        self.actor_optimizer = Adam(learning_rate=learning_rate_actor)
        self.actor.summary()

        self.critic = self.build_critic_model(self.state_size, self.action_size, model_name=model_name + "_critic")
        # self.critic.build(input_shape=[(state_size,), (action_size,)])
        self.critic_dash = self.build_critic_model(self.state_size, self.action_size, model_name=model_name + "_critic_target")
        # self.critic_dash.build(input_shape=[(state_size,), (action_size,)])
        
        self.critic_dash.set_weights(self.critic.get_weights())

        self.critic_optimizer = Adam(learning_rate=learning_rate_critic)
        self.critic.summary()

        self.load_state()

        self.rollout_memory(rollout_steps - len(self.memory))

    def build_critic_model(self, input_size, action_size, model_name='critic'):
        """
        Returns Q(st+1 | a, s)
        """

        inputs = [Input(shape=(input_size)), Input(shape=(action_size,))]
        concat = Concatenate(axis=-1)(inputs)
        x = Dense(24, name="hidden_1", activation='tanh')(concat)
        x = Dense(48, name="hidden_2", activation='tanh')(x)
        output = Dense(1, name="Out")(x)
        model = Model(inputs=inputs, outputs=output, name=model_name)
        model.build(input_shape=[(input_size,), (action_size,)])
    
        return model

    def act_with_noise(self, state, add_noise=True):
        raise NotImplementedError(
            "Consider implementing from\nhttps://github.com/anita-hu/"
            "TF2-RL/blob/master/DDPG/TF2_DDPG_Basic.py")

    def show(self, render=False):
        raise NotImplementedError("self.model needs to be adapted in super")

    def solve(self, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env = self.env_wrapper.env
        state = env.reset()

        success_steps = 0

        for iteration in range(max_iters):
            for step in range(self.n_cycles):  # itertools.count():
                if render:
                    env.render()

                # TODO implement act and add noise
                action_dist = self.actor(tf.expand_dims(state,axis=0))
                observation, reward, done, _ = env.step(np.argmax(action_dist))
                
                # Custom reward if required by env wrapper
                reward = self.env_wrapper.reward_on_step(
                    state, observation, reward, done, step)

                self.memory.append(
                    (state, tf.squeeze(action_dist), np.float64(reward), 
                     observation, done)
                )
                state = observation

                self.report_step(step, iteration, max_iters)

                if done:
                    # OR env_wrapper.get_score(state, observation, reward, step)
                    self.scores.append(success_steps)
                    success_steps = 0
                    state = env.reset()
                else:
                    success_steps += 1

                self.take_training_step()

            solved = self.handle_episode_end(
                state, observation, reward, 
                step, max_iters, verbose=verbose)

            if solved:
                break

        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def take_training_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Note min is actually unecessary with cond above
        minibatch_i = np.random.choice(
            len(self.memory),
            min(self.batch_size, len(self.memory)),
        )
        
        minibatch = [self.memory[i] for i in minibatch_i]

        # Obs on [adv, return]
        loss_value = self.train_on_minibatch(
            *tuple(map(tf.convert_to_tensor, zip(*minibatch)))
        )

        # Update weights
        for model_name in "actor", "critic": 
            self.update_weights(model_name, self.tau)
        
        # TODO decrease epsilon if not None

    @tf.function()
    def train_on_minibatch(self, sts, a, r, n_sts, d):

        # r + gam(1-d)Q_phi_targ(s_t+1, mu_theta_targ(s_t+1))
        n_a = self.actor_dash(n_sts)
        q_future_pred = self.critic_dash([n_sts, n_a])
        target_qs = r + tf.where(
            d,
            tf.zeros(shape=q_future_pred.shape, dtype=tf.dtypes.float64),
            self.gamma * q_future_pred
        )

        # Minimise (r + target on next state) - (current critic on sts and a)
        # Makes critic better at predicting future
        with tf.GradientTape() as tape:
            updated_q_values = self.critic([sts, a])
            critic_loss = tf.reduce_mean(
                tf.math.square(updated_q_values - target_qs)
            )

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        # Use the (improving) critic to rate the actor's updated decision
        # Minimising loss means maximising actor's expectation
        with tf.GradientTape() as tape:
            # mu_phi(s)
            updated_action_dist = self.actor(sts)
            # Works due to chain rule, tracks mu gradients to improve mu prediciton
            # TODO this is quite nuanced - check this through
            actor_loss = - tf.reduce_mean(
                self.critic([sts, updated_action_dist])
            )

        actor_grad = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    def update_weights(self, model_name, tau):
        weights = getattr(getattr(self, model_name), "weights")
        target_model = getattr(self, model_name + "_dash")
        target_weights = target_model.weights
        target_model.set_weights([
            weights[i] * tau + target_weights[i] * (1. - tau) 
            for i in range(len(weights))
        ])

    def save_state(self):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            "memory": self.memory,
            "epsilon": self.epsilon,
            "actor_optimizer_config": self.actor_optimizer.get_config(),
            "critic_optimizer_config": self.critic_optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)
        
        for var in ("actor", "actor_dash", "critic", "critic_dash"):
            model = getattr(self, var)
            model.save_weights(self.model_location.replace(
                ".h5", 
                "_" + var + ".h5")
            )

    def load_state(self):
        """Load a model with the specified name"""

        model_dict = self.load_state_from_dict()

        print("Loading weights from", self.model_location + "...", end="")
        if os.path.exists(self.model_location):
            for var in ("actor", "actor_dash", "critic", "critic_dash"):
                model = getattr(self, var)
                self.model.load_weights(self.model_location.replace(
                    ".h5",
                    "_" + var + ".h5")
                )
            self.actor_optimizer = self.actor_optimizer.from_config(self.actor_optimizer_config)
            self.critic_optimizer = self.critic_optimizer.from_config(self.critic_optimizer_config)
            del model_dict["actor_optimizer_config"], self.actor_optimizer_config
            del model_dict["critic_optimizer_config"], self.critic_optimizer_config
            print(" Loaded.")
        else:
            print(" Model not yet saved at loaction.")

        if "memory" in model_dict:
            del model_dict["memory"]

        print("Loaded state:")
        pprint.pprint(model_dict, depth=1)

    def rollout_memory(self, rollout_steps, render=False):
        if rollout_steps <= 0:
            return
        print("Rolling out steps", rollout_steps)
        env = self.env_wrapper.env
        state = env.reset()

        max_iters = rollout_steps // self.n_cycles

        for iteration in range(max_iters):
            for step in range(self.n_cycles):
                if render:
                    env.render()

                # TODO implement act and add noise
                action_dist = self.actor(tf.expand_dims(state, axis=0))
                observation, reward, done, _ = env.step(np.argmax(action_dist))
                
                # Custom reward if required by env wrapper
                reward = self.env_wrapper.reward_on_step(
                    state, observation, reward, done, step)

                self.memory.append(
                    (state, tf.squeeze(action_dist), np.float64(reward), 
                     observation, done)
                )
                state = observation

                self.report_step(step, iteration, max_iters)

                if done:
                    state = env.reset()

        print("\nCompleted.")
