

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


with open('config.yml') as f:
    config = yaml.safe_load(f)
env_id = 'gym_join:join-v0'
print("DQN Gen Bandit Join")
env = gym.make(env_id)
env.set_config(config)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
    
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



# def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
#     """
#     Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
#     while following an epsilon-greedy policy
    
#     Args:
#         env: OpenAI environment.
#         num_episodes: Number of episodes to run for.
#         discount_factor: Gamma discount factor.
#         alpha: TD learning rate.
#         epsilon: Chance to sample a random action. Float between 0 and 1.
    
#     Returns:
#         A tuple (Q, episode_lengths).
#         Q is the optimal action-value function, a dictionary mapping state -> action values.
#         stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
#     """
    
#     # The final action-value function.
#     # A nested dictionary that maps state -> (action -> action-value).
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))

#     # Keeps track of useful statistics
#     # stats = plotting.EpisodeStats(
#     #     episode_lengths=np.zeros(num_episodes),
#     #     episode_rewards=np.zeros(num_episodes))    
    
#     # The policy we're following
#     policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
#     for i_episode in range(num_episodes):
#         # Print out which episode we're on, useful for debugging.
#         if (i_episode + 1) % 100 == 0:
#             print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
#             sys.stdout.flush()
        
#         # Reset the environment and pick the first action
#         state = env.reset()
#         state = tuple(state)
#         # One step in the environment
#         # total_reward = 0.0
#         for t in itertools.count():
            
#             # Take a step
#             action_probs = policy(state)
#             action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
#             next_state, reward, done = env.step(action)
#             print(state)
#             next_state = tuple(next_state)
#             # Update statistics
#             # stats.episode_rewards[i_episode] += reward
#             # stats.episode_lengths[i_episode] = t
            
#             # TD Update
#             # print(Q[next_state])
#             best_next_action = np.argmax(Q[next_state])    
#             td_target = reward + discount_factor * Q[next_state][best_next_action]
#             td_delta = td_target - Q[state][action]
#             Q[state][action] += alpha * td_delta
                
#             if done:
#                 break
                
#             state = next_state
    
#     return Q


def ep_policy(observation, epsilon, Q):
    if random.random() > epsilon:
        best_action = np.argmax(Q[observation])
        action  = best_action
    else:
        action  = random.randint(0,1)
        # if random.random() < 0.8:
        #     action = 0
        # else:
        #     action = 1
        # print(action)
    return action


def q_learning_new(env, discount_factor=1.0, alpha=0.6, epsilon=0.1):
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
    done = False
    iterations = 0
    while not done:
        # print(epsilon)
        epsilon = epsilon_by_iteration(iterations)

        action = ep_policy(state, epsilon, Q)
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
        # print(Q[next_state])
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta
            
        state = next_state
        iterations += 1

    return Q

if __name__ == "__main__":

    iters = 5
    total_time = 0
    for _ in range(iters):
            
        start = datetime.now().timestamp()
        
        Q = q_learning_new(env)
        total_time += (datetime.now().timestamp() - start)
        # print(datetime.now() - start)
        print(len(env.results))
    print("Avg time taken after " + str(iters) + " iterations : " + str(total_time / iters))





