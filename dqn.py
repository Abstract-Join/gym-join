import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import yaml
from datetime import datetime
from collections import deque
from IPython.display import clear_output
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    # def act(self, state, epsilon):
    #     if random.random() > epsilon:
    #         state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
    #         q_value = self.forward(state)
    #         action  = q_value.max(1)[1].data[0]
    #     else:
    #         action = random.randrange(env.action_space.n)
    #     return action

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            # q_values
            # print(q_value)
            action  = q_value.max(1)[1].data[0]
        else:
            if random.random() < 0.8:
                action = 0
            else:
                action = 1


        return action


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":

    with open('config.yml') as f:
        config = yaml.safe_load(f)
    USE_CUDA = torch.cuda.is_available()
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

    env_id = 'gym_join:join-v0'
    print("DQN Gen Bandit Join")
    start = datetime.now()
    env = gym.make(env_id)
    env.set_config(config)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    model = DQN(env.observation_space.shape[0], env.action_space.n)

    if USE_CUDA:
        model = model.cuda()
        
    optimizer = optim.Adam(model.parameters())

    replay_buffer = ReplayBuffer(1000)


    batch_size = 16
    gamma      = 0.5

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    frame_idx = 0
    done = False

    start = datetime.now()
    while not done:
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)
        
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        all_rewards.append(reward)
            
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.data.item())
            
    #     if frame_idx % 200 == 0:
    #         plot(frame_idx, all_rewards, losses)

        frame_idx += 1
    print("Time taken : " + str(datetime.now() - start))
    print(len(env.results))