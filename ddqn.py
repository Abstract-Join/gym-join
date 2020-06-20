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
from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

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
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def plot(frame_idx, rewards, losses):
    print("plot")
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    # plt.subplot(132)
    # plt.title('loss')
    # plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    writer = SummaryWriter(logdir='scalar/training')

    with open('config.yml') as f:
        config = yaml.safe_load(f)

    env_id = 'gym_join:join-v0'
    start = datetime.now()

    env = gym.make(env_id)
    env.set_config(config)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 100

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model  = DQN(env.observation_space.shape[0], env.action_space.n)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
        
    optimizer = optim.Adam(current_model.parameters())

    replay_buffer = ReplayBuffer(1000)


    update_target(current_model, target_model)


    batch_size = 15
    gamma      = 0.5

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    frame_idx = 0
    done = False

    print("Gen Bandit Join")
    start = datetime.now()
    total_reward = 0
    episode_reward = 0
    forward = 0
    stay = 0
    epi_reward_list = []
    while not done:
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)
        # print("State : " + str(state))       
        # print(epsilon) 

        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        # print("State : " + str(state) + " Actions : " + str(int(action)) + " Reward : " + str(reward), " NextState : " + str(next_state))
        state = next_state
        episode_reward += reward
        all_rewards.append(reward)
        if action == 1:
            total_reward += episode_reward
            # print(episode_reward)
            epi_reward_list.append(episode_reward)
            forward += 1
            writer.add_scalar('reward', episode_reward, frame_idx)
            episode_reward = 0
        else:
            stay += 1
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.data.item())
            
        # if frame_idx % 20 == 0:
        #     plot(frame_idx, forward, stay)
            
        if frame_idx % 100 == 0:
            update_target(current_model, target_model)
        frame_idx += 1
    
    plot(len(epi_reward_list), epi_reward_list, None)
    print("Forward : " + str(forward))
    print("Stay : " + str(stay))

    print("Time taken : " + str(datetime.now() - start))
    print(len(env.results))
    writer.close()
