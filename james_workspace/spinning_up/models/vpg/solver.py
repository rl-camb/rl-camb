# TODO - continue with https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py

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


# TODO: Consider going back to a memory and batch learning..
class VPGSolver(StandardAgent):
    """
    A standard dqn_solver, inpired by:
      https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
    Implements a simple DNN that predicts values.
    """

    def __init__(self, 
        experiment_name, 
        state_size, 
        action_size, 
        gamma=0.99, 
        epsilon=None,
        epsilon_decay_rate=0.995,
        epsilon_min=0.1,
        batch_size=64,
        learning_rate=0.01,
        model_name="vpg", 
        saving=True):

        self.state_size = state_size
        self.action_size = action_size
        self.action_size_tensor = tf.constant(action_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_name = model_name

        self.solved_on = None

        self.model = self.build_model()

        self.optimizer = Adam(lr=learning_rate)  # decay=learning_rate_decay)

        super(VPGSolver, self).__init__(
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

    def show(self, env_wrapper, verbose=False, render=False):
        raise NotImplementedError(
            "TODO - implement a show method that runs the agent with "
            "act(epsilon=None)")

    def solve(self, env_wrapper, max_iters, verbose=False, render=False):
        env = env_wrapper.env
        start_time = datetime.datetime.now()
        
        for batch_num in range(max_iters):
            
            done, state, episode_rewards = False, env.reset(), []
            state_batch, act_batch, batch_future_rewards = [], [], []

            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(state, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(action)

                # Custom reward if required by env wrapper
                state_next = observation
                reward = env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                
                # TODO figue out if need copy on both (first step -corner case)
                state_batch.append(state.copy())
                act_batch.append(np.int32(action))
                episode_rewards.append(reward)

                state = observation.copy()
                print(f"\rBatch {batch_num + 1}/{max_iters} - "
                      f"steps {step} ({self.total_t + 1})", 
                      end="")
                sys.stdout.flush()

                self.total_t += 1
                if done:
                    batch_future_rewards += list(
                        self.discount_future_cumsum(
                            episode_rewards, self.gamma))
                    # Run episodes until we have at least batch size example
                    if len(state_batch) > self.batch_size:
                        break
                    else:
                        # Get some more experience, clean episode
                        done, state, episode_rewards = False, env.reset(), []

            # Normalise advantages
            batch_advs = np.array(batch_future_rewards)
            batch_advs = (
                (batch_advs - np.mean(batch_advs))
                 / (np.std(batch_advs) + 1e-8)
            )

            # TODO implement a get_batch function so that solves are identical
            self.learn(
                *map(tf.convert_to_tensor, 
                    (state_batch, act_batch, batch_advs)))

            score = len(episode_rewards)  # E.g. steps in last episode
            self.scores.append(score) 

            solved, agent_score = env_wrapper.check_solved_on_done(
                state, self.scores, verbose=verbose)

            if batch_num % 25 == 0 or solved:
                print(f"\rBatch {batch_num + 1}/{max_iters} "
                      f"- steps {step} - score {int(agent_score)}/"
                      f"{int(env_wrapper.score_target)}")
                if self.saving:
                    self.save_state(add_to_save={
                        "optimizer_config": self.optimizer.get_config(),
                    })

            if solved:
                self.solved_on = batch_num
                break
        
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def act(self, state, epsilon=None):
        """Take a random action or the most valuable predicted
        action, based on the agent's model. 
        """
        if (state.shape != (self.state_size,) 
                and state.shape != (1, self.state_size)):
            raise NotImplementedError(
                "Not intended for use on batch state; returns integer")

        # If in exploration
        if epsilon and np.random.rand() <= epsilon:
            return random.randrange(self.action_size)

        if state.ndim == 1:
            state = tf.reshape(state, (1, self.state_size))

        return tf.math.argmax(self.model(state), axis=-1).numpy()[0]
    
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

    # TODO have an issue where the batch is not consistent size so "tracing"
    #  Resolve: Experiment with how good it is with / without memory and fixed BS
    # @tf.function
    def take_training_step(self, sts, acts, advs):
        tf.debugging.assert_equal(tf.shape(sts)[0], tf.size(acts), summarize=1) 
        tf.debugging.assert_equal(tf.size(acts), tf.size(advs), summarize=1)

        with tf.GradientTape() as tape:
            logits = self.model(sts)
            action_one_hots = tf.one_hot(
                acts, self.action_size_tensor, dtype=tf.float64)
            log_probs = tf.math.reduce_sum(
                action_one_hots * tf.nn.log_softmax(logits), 
                axis=1)

            tf.debugging.assert_equal(tf.size(advs), tf.size(log_probs))

            loss_value = - tf.math.reduce_mean(advs * log_probs)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss_value

    def save_state(self, add_to_save={}):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

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



class VPGSolverWithMemory(VPGSolver):

    # The VPG cannot be optimised as batch size constantly changes
    # Instead implement a memory and memory replay, investigate stability
    # NOTE: will need to examine steps (total_t), not episodes
    # as VPG doesn't implement episodes

    def __init__(self, 
        experiment_name, 
        state_size, 
        action_size, 
        gamma=0.99,
        memory_len=100000,
        epsilon=None,
        epsilon_decay_rate=0.995,
        epsilon_min=0.1,
        batch_size=64,
        learning_rate=0.01,
        model_name="vpg_batch", 
        saving=True):
        
        self.memory = deque(maxlen=memory_len)

        super().__init__(
            experiment_name, 
            state_size=state_size, 
            action_size=action_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name, 
            saving=saving)

    
    # TODO switch batch implementation to memory step saving
    def solve(self, env_wrapper, max_episodes, verbose=False, render=False):
        # TODO implement episodes and deque instead of batch
        env = env_wrapper.env
        start_time = datetime.datetime.now()
        
        for episode in range(max_episodes):
            
            done, state, episode_rewards = False, env.reset(), []
            state_batch, act_batch, batch_future_rewards = [], [], []

            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(state, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(action)

                # Custom reward if required by env wrapper
                state_next = observation
                reward = env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                
                # TODO figue out if need copy on both (first step -corner case)
                state_batch.append(state.copy())
                act_batch.append(np.int32(action))
                episode_rewards.append(reward)

                state = observation.copy()
                print(f"\rEpisode {episode + 1}/{max_episodes} - "
                      f"steps {step} ({self.total_t + 1})", 
                      end="")
                sys.stdout.flush()

                self.total_t += 1
                if done:
                    batch_future_rewards = list(
                        self.discount_future_cumsum(
                            episode_rewards, self.gamma))
                    break

            # Normalise advantages
            batch_advs = np.array(batch_future_rewards)
            batch_advs = (
                (batch_advs - np.mean(batch_advs))
                 / (np.std(batch_advs) + 1e-8)
            )

            # TODO optimise
            for i in range(len(state_batch)):
                self.memory.append((state_batch[i], act_batch[i], batch_advs[i]))
            # self.memory.append(tuple(zip(state_batch, act_batch, batch_advs)))
            assert len(state_batch) == len(act_batch) and len(act_batch) == len(batch_advs)

            self.learn(*self.get_batch_to_train())

            score = len(episode_rewards)  # E.g. steps in last episode
            self.scores.append(score) 

            solved, agent_score = env_wrapper.check_solved_on_done(
                state, self.scores, verbose=verbose)

            if episode % 25 == 0 or solved:
                print(f"\rEpisode {episode + 1}/{max_episodes} "
                      f"- steps {step} - score {int(agent_score)}/"
                      f"{int(env_wrapper.score_target)}")
                if self.saving:
                    self.save_state(add_to_save={
                        "optimizer_config": self.optimizer.get_config(),
                    })

            if solved:
                self.solved_on = episode
                break
        
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved


    def get_batch_to_train(self):
        minibatch_i = np.random.choice(len(self.memory),
            min(self.batch_size, len(self.memory)),
            )
        
        minibatch = [self.memory[i] for i in minibatch_i]

        return tuple(map(tf.convert_to_tensor, zip(*minibatch)))

    @tf.function
    def take_training_step(self, sts, acts, advs):
        super().take_training_step(sts, acts, advs)