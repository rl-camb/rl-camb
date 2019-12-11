
import numpy as np
import tqdm
import gym
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor

state_bins = [100,100]
epis = 10000

def run_learning(params):

    learning_rate = params[0][0]
    discount_factor = params[0][1]
    proc_num = params[1]
    env = gym.make('MountainCar-v0')

    # will need to convert continuous observation space (Box) into discrete
    def discrete_state(state, n_bins):
        return ((n_bins * (state - env.observation_space.low)/(env.observation_space.high -
                                                               env.observation_space.low))
                .astype(int)).tolist()

    # Q-learning alg: store reward(state,action) matrix
    # or should be randomized? Q = np.random.uniform(low = -1, high = 1, ...
    Q = np.zeros(discrete_state(env.observation_space.high,state_bins) + [env.action_space.n])

    from collections import defaultdict
    results = defaultdict(list)
    for i_episode in tqdm.tqdm(range(epis),unit='episodes',position=proc_num):
        s = tuple(discrete_state(env.reset(),state_bins)) # returns initial observation (state)
        action = np.argmax(Q[s]) if np.random.rand() <= 1./(i_episode+1) else \
            env.action_space.sample()
        done_episode = False
        steps = 0
        total_reward = 0
        success = False
        while not done_episode:
            #if (i_episode % 1000 == 0): env.render() # visualize state

            # choose action - is randomised to start with but each episode the randomness
            # will diminish - policy becomes less stochastic
            # RANDOM policy: action = env.action_space.sample()

            #env.action_space is Discrete(3) - i.e. action is 0 1 or 2
            #action = np.argmax(Q[s] + np.random.randn(1,env.action_space.n)/(i_episode+1))

            # alternative could be 'greedy epsilon' - throw random number and if its less than
            # 1 - epsilon (epsilon decreases over episodes) then use deterministic action
            # otherwise use completely random action

            next_s, reward, done_episode,_ = env.step(action)
            success = (next_s[0] >= 0.5) # climbed the hill!
            next_s = tuple(discrete_state(next_s,state_bins)) # convert to tuple
            #Update Q-Table with new knowledge

            #next_action = np.argmax(Q[next_s] + np.random.randn(1,env.action_space.n)/(i_episode+1))
            # using argmax of next Q is the 'greedy in Q' policy
            if np.random.rand() <= 1/(i_episode+1):
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_s])

            #Q_next = np.max(Q[next_s])
            Q_next = Q[next_s][next_action] # SARSA

            Q[s][action] += learning_rate * (reward + discount_factor*Q_next - Q[s][action])
            #Q[s][action] = (1.-learning_rate) * Q[s][action] + \
            #               learning_rate*(reward + discount_factor* np.max(Q[next_s]))

            action = next_action
            steps += 1
            total_reward += reward
            s = next_s

        results["episode"].append(i_episode)
        results["discount_factor"].append(str(discount_factor))
        results["learning_rate"].append(str(learning_rate))
        results["steps"].append(steps)
        results["success"].append(success)
    env.close()
    return pd.DataFrame(results)


# 2. Parameters of Q-learning - try all combinations
learning_rates = [0.5,.9,1.0]
discount_factors = [0.9,1.0]


with ProcessPoolExecutor(3) as pool:
    num_procs = len(learning_rates)*len(discount_factors)
    results = pd.concat(pool.map( run_learning,
                                  zip( itertools.product(learning_rates,discount_factors),
                                       range(0,num_procs) )
                                  )
                        )






results.set_index(['episode','learning_rate','discount_factor'],inplace=True)
results['success_over_steps'] = results['success'] / results['steps']
results = results.unstack(level=[1,2])
ax = results['success_over_steps'].rolling(100).mean().plot(title="success divide by steps")