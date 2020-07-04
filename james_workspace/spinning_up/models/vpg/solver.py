import os
import random
import itertools
import sys
import pprint
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from collections import deque
from models.standard_agent import StandardAgent

from utils import conditional_decorator


class VPGSolver(StandardAgent):
    """
    A standard vpg_solver, inpired by:
      https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py
    NOTE: 
        will need to examine steps (total_t), not episodes, as VPG doesn't
        implement episodes per-training-step
    """
    can_graph = True  # batch size is variable, cannot use tf graphing

    def __init__(self, 
        experiment_name, 
        env_wrapper,
        gamma=0.99, 
        epsilon=None,
        epsilon_decay_rate=0.995,
        epsilon_min=0.1,
        batch_size=64,
        n_cycles=128,
        learning_rate=0.01,
        model_name="vpg", 
        saving=True):

        super(VPGSolver, self).__init__(
            env_wrapper,
            model_name,
            experiment_name,
            saving=saving)

        self.label = "Batch"  # not by episode, by arbitrary batch
        self.action_size_tensor = tf.constant(self.action_size)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min

        # TODO could go to standard..
        self.batch_size = batch_size
        self.n_cycles = n_cycles

        self.memory = []  # state
        self.solved_on = None

        self.model = self.build_model()
        self.optimizer = Adam(lr=learning_rate)  # decay=learning_rate_decay)

        self.load_state()

        # TODO rollout steps

    @staticmethod
    def discount_future_cumsum(episode_rewards, gamma):
        """
        Takes: 
            A list of rewards per step for an episode
        Returns: 
            The future reward at each step, with the future discounting 
            rate applied from that step onwards.
        """
        ep_rwds = np.array(episode_rewards)
        n = len(ep_rwds)
        discounts = gamma ** np.arange(n)
        discounted_futures = np.zeros_like(ep_rwds, dtype=np.float64)
        for j in range(n):
            discounted_futures[j] = sum(ep_rwds[j:] * discounts[:(n-j)])

        assert len(discounted_futures) == len(episode_rewards)
        return discounted_futures

    def solve(self, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env = self.env_wrapper.env
        state, done, episode_rewards = env.reset(), False, []
        success_steps = 0

        for batch_num in range(max_iters):
            
            state_batch, act_batch, batch_future_rewards = [], [], []

            for step in range(self.n_cycles):
                if render:
                    env.render()

                action = self.act(self.model, state, epsilon=self.epsilon)
                state_next, reward, done, _ = env.step(action)

                # Custom reward if required by env wrapper
                reward = self.env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                
                state_batch.append(state.copy())
                act_batch.append(np.int32(action))
                episode_rewards.append(reward)

                # NOTE: Removed copy
                state = state_next

                self.report_step(step, batch_num, max_iters)

                if done:

                    # At the end of each episode:
                    # Create a list of future rewards, 
                    #  discounting by how far in the future
                    batch_future_rewards += list(
                        self.discount_future_cumsum(
                            episode_rewards, self.gamma))
                    state, done, episode_rewards = env.reset(), False, []
                    self.scores.append(success_steps)
                    success_steps = 0
                else:
                    success_steps +=1
            
            # Add any trailing rewards to done
            batch_future_rewards += list(
                self.discount_future_cumsum(
                episode_rewards, self.gamma)
            )
            episode_rewards = []

            # HANDLE END OF EPISODE
            batch_advs = np.array(batch_future_rewards)

            # This is R(tau), normalised
            normalised_batch_advs = ( 
                (batch_advs - np.mean(batch_advs))
                / (np.std(batch_advs) + 1e-8)
            )

            self.remember(state_batch, act_batch, normalised_batch_advs)
            self.learn(*self.get_batch_to_train())

            solved = self.handle_episode_end(
                state, state_next, reward, 
                step, max_iters, verbose=verbose)

            if solved:
                break
        
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def remember(self, state_batch, act_batch, batch_advs):

        self.memory = (state_batch, act_batch, batch_advs)

    def get_batch_to_train(self):

        assert len(self.memory[0]) == len(self.memory[1]), f"{len(self.memory[0])}, {len(self.memory[1])}"
        assert len(self.memory[1]) == len(self.memory[2]), f"{len(self.memory[1])}, {len(self.memory[2])}"

        minibatch_i = np.random.choice(len(self.memory[0]),
            min(self.batch_size, len(self.memory[0])),
            )
        
        sampled_memory = []
        for i in range(len(self.memory)):
            sampled_memory.append(tf.convert_to_tensor([self.memory[i][j] for j in minibatch_i]))

        self.memory = []  # Only learning from last set of trajectories

        return sampled_memory
    
    def learn(self, sts, acts, advs):
        """Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.
        """

        loss_value = self.take_training_step(sts, acts, advs)

        if self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate

        return loss_value

    @conditional_decorator(tf.function, can_graph)
    def take_training_step(self, sts, acts, advs):
        tf.debugging.assert_equal(tf.shape(sts)[0], tf.size(acts), summarize=1) 
        tf.debugging.assert_equal(tf.size(acts), tf.size(advs), summarize=1)

        with tf.GradientTape() as tape:
            
            # One step away from Pi_theta(at|st)
            pi_action_logits = self.model(sts)
            
            action_one_hots = tf.one_hot(
                acts, self.action_size_tensor, dtype=tf.float64)
            
            # This IS pi_theta(at|st), only at the actual action taken
            pi_action_log_probs = tf.math.reduce_sum(
                action_one_hots * tf.nn.log_softmax(pi_action_logits), 
                axis=1)

            tf.debugging.assert_equal(tf.size(advs), tf.size(pi_action_log_probs))

            loss_value = - tf.math.reduce_mean(
                advs * pi_action_log_probs
            )

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss_value

    def save_state(self, add_to_save={}):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

        self.save_state_to_dict(append_dict={
            "optimizer_config": self.optimizer.get_config(),
            "epislon": self.epsilon,
        })

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
