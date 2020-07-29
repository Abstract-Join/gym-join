

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import yaml
from datetime import datetime
from collections import defaultdict
import random 
import math

from nested_loop import start_experiment

def n_explore_val(page_size, r_size, s_size, k):
    blocks = r_size // page_size

    n = k ** (2/3) * (math.log(k)) ** (1/3)

    return n

def n_explore_policy(n, iteration, Q, observation):
    if n < iteration:
        action  = random.randint(0,1)
    else:
        best_action = np.argmax(Q[observation])
        action  = best_action
    return action

def k_explore_policy(t, k, Q, observation):
    epsilon = t ** (-1/3) * (k * math.log(t)) ** (1/3)
    # print(epsilon)

    if random.random() > epsilon:
        best_action = np.argmax(Q[observation])
        action  = best_action
    else:
        action  = random.randint(0,1)
    return action


def ep_policy(observation, epsilon, Q):
    if random.random() > epsilon:
        best_action = np.argmax(Q[observation])
        action  = best_action
    else:
        action  = random.randint(0,1)
    return action


def q_learning_new(env, discount_factor=0.2, alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # print(Q)
    # print(type(Q))
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_iteration = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    state = env.reset()
    state = tuple(state)
    # One step in the environment
    # total_reward = 0.0
    r_size, s_size = env.get_db_size()
    # n_val = n_explore_val(config["env"]["page_size"], r_size, s_size, env.k)
    # print(n_val)
    done = False
    iterations = 1
    while not done:
        # print(epsilon)
        epsilon = epsilon_by_iteration(iterations)
        
        action = ep_policy(state, epsilon, Q)

        # action = n_explore_policy(n_val, iterations, Q, state)
        # action = k_explore_policy(iterations, env.k, Q, state)

        next_state, reward, done = env.step(action)
        # print(state, next_state, reward, action)
        # print(next_state)
        next_state = tuple(next_state)
        # print(next_state)

        # Update statistics
        # stats.episode_rewards[i_episode] += reward
        # stats.episode_lengths[i_episode] = t
        
        # TD Update
        best_next_action = np.argmax(Q[next_state])
        # print(next_state, Q[next_state])
        td_target = reward + discount_factor * Q[next_state][best_next_action]
        # print(reward)
        print(Q)
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta
            
        state = next_state
        iterations += 1

    return Q

def start_ql_experiment(config, iters):
    total_time = 0

    env_id = 'gym_join:join-v0'
    # print(config)
    results = 0
    for _ in range(iters):
        env = gym.make(env_id)
        env.set_config(config)
        start = datetime.now().timestamp()
        
        _ = q_learning_new(env)
        # print(Q)
        total_time += (datetime.now().timestamp() - start)
        results += len(env.results)
        # print(datetime.now() - start)
        # print(len(env.results))
    # print("Avg time taken after " + str(iters) + " iterations : " + str(total_time / iters))
    return total_time / iters, results / iters

if __name__ == "__main__":


    with open('config.yml') as f:
        config = yaml.safe_load(f)
    env_id = 'gym_join:join-v0'
    print("QLearning Gen Bandit Join")
    
    # page_sizes = [32, 64, 128, 256, 512]
    # k_size = [50, 100, 500, 1000, 5000, 10000, 50000]

    page_sizes = [128]
    k_size = [5000]

    # n_array = 2
    ql_result= []
    nl_result = []
    for k in k_size:
        config["env"]["k"] = k
        for page_size in page_sizes:
            config["env"]["page_size"] = page_size
            print("K : "  + str(k) + " Page Size: " + str(page_size))
            time, avg_results = start_ql_experiment(config, 1)
            print("    QL time: " + str(time) + " Results : " + str(avg_results))
            ql_result.append((k, page_size, time, avg_results))
            time, avg_results = start_experiment(config, 1)
            nl_result.append((k, page_size, time, avg_results))
            print("    NL time: " + str(time) + " Results : " + str(avg_results))

    print("QL Result")
    print(ql_result)

    print("\n\nNL Result")
    print(nl_result)




