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
    can_graph = False  # batch size is variable, cannot use tf graphing

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
        
        self.label = "Batch"  # not by episode, by arbitrary batch
        self.batch_size = batch_size
        self.min_batch_size = batch_size
        

        self.memory = []  # state
        self.solved_on = None

        self.model_name = model_name
        self.model = self.build_model()
        self.optimizer = Adam(lr=learning_rate)  # decay=learning_rate_decay)

        super(VPGSolver, self).__init__(
            self.model_name + "_" + experiment_name, 
            saving=saving)
        self.load_state()

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

    def solve(self, env_wrapper, max_iters, verbose=False, render=False):
        env = env_wrapper.env
        start_time = datetime.datetime.now()
        
        for batch_num in range(max_iters):
            
            done, state, episode_rewards = False, env.reset(), []
            state_batch, act_batch, batch_future_rewards = [], [], []

            # STEP THROUGH EPISODE
            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(self.model, state, epsilon=self.epsilon)
                state_next, reward, done, _ = env.step(action)

                # Custom reward if required by env wrapper
                reward = env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                
                state_batch.append(state.copy())
                act_batch.append(np.int32(action))
                episode_rewards.append(reward)

                # TODO figue out if need copy
                state = state_next.copy()
                print(f"\r{self.label} {batch_num + 1}/{max_iters} - "
                      f"steps {step} ({self.total_t + 1})", 
                      end="")
                sys.stdout.flush()

                self.total_t += 1

                if done:
                    batch_future_rewards += list(
                        self.discount_future_cumsum(
                            episode_rewards, self.gamma))
                    # Run episodes until we have at least batch size example
                    if len(state_batch) > self.min_batch_size:
                        break
                    else:
                        # Get some more experience, clean episode
                        done, state, episode_rewards = False, env.reset(), []

            # HANDLE END OF EPISODE
            # Normalise advantages
            batch_advs = np.array(batch_future_rewards)
            batch_advs = (
                (batch_advs - np.mean(batch_advs))
                 / (np.std(batch_advs) + 1e-8)
            )

            self.remember(state_batch, act_batch, batch_advs)
            self.learn(*self.get_batch_to_train())

            score = len(episode_rewards)  # E.g. steps in last episode
            self.scores.append(score)

            solved, agent_score = env_wrapper.check_solved_on_done(
                state, self.scores, verbose=verbose)

            if batch_num % 25 == 0 or solved:
                print(f"\r{self.label} {batch_num + 1}/{max_iters} "
                      f"- steps {step} - score {int(agent_score)}/"
                      f"{int(env_wrapper.score_target)}")

                if self.saving:
                    self.save_state()

            if solved:
                self.solved_on = (batch_num, self.total_t)
                break
        
        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def remember(self, state_batch, act_batch, batch_advs):

        self.memory = (state_batch, act_batch, batch_advs)

    def get_batch_to_train(self):

        return map(tf.convert_to_tensor, self.memory)
    
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


class VPGSolverWithMemory(VPGSolver):
    """
    A VPG solver that implements batch learning from a memory, for more 
    stable training (pending test).
    """
    can_graph = True  # batch size is fixed, can use tf graphing
    
    def __init__(self, 
        experiment_name, 
        state_size, 
        action_size, 
        memory_len=100000,
        model_name="vpg_batch",
        **kwargs):
        

        # Rest of state defaults remain the same
        super().__init__(
            experiment_name, 
            state_size=state_size, 
            action_size=action_size,
            model_name=model_name,
            **kwargs)
    
        # Overwrite parent state
        self.memory = deque(maxlen=memory_len)
        self.label = "Episode"  # Iterates by episode
        self.min_batch_size = 0  # stop episodes on done no matter what length

    def remember(self, state_batch, act_batch, batch_advs):
        """
        Save each step as a tuple of values.
        """
        self.memory.extend(tuple(zip(state_batch, act_batch, batch_advs)))

    def get_batch_to_train(self):
        """
        Sample a random self.batch_size-sized minibatch from self.memory
        """
        minibatch_i = np.random.choice(
            len(self.memory),
            min(self.batch_size, len(self.memory)),
            )
        minibatch = [self.memory[i] for i in minibatch_i]
        return tuple(map(tf.convert_to_tensor, zip(*minibatch)))

    def save_state(self, add_to_save={}):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

        self.save_state_to_dict(append_dict={
            "optimizer_config": self.optimizer.get_config(),
            "memory": self.memory,
            "epislon": self.epsilon
        })

        self.model.save(self.model_location)
