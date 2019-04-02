import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    def __init__(self, n_past_action_to_remember=10):
        super(DeepQNetwork, self).__init__()
        per_trainned_cnn_output_size = 4096
        n_action = 9
        input_size = per_trainned_cnn_output_size + n_action * n_past_action_to_remember

        self.linear_1 = nn.Linear(input_size, 1024)
        self.relu_1 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(1024, 1024)
        self.relu_2 = nn.ReLU()
        self.linear_out = nn.Linear(1024, n_action)

    def forward(self, x):
        out_l1 = self.dropout(self.relu_1(self.linear_1(x)))
        out_l2 = self.relu_2(self.linear_2(out_l1))
        out = self.linear_out(out_l2)
        return out


class Agent:
    def __init__(self, env, target_update = 10, discout_rate=0.99, eps_start=0.9, eps_end=0.05, eps_decay=200, batch_size = 64, memory_size=1000, n_past_action_to_remember=10):
        self.target_update = target_update
        self.discount_rate = discout_rate
        self.n_action = env.action_space.n
        self.batch_size = batch_size
        self.n_past_action_to_remember = n_past_action_to_remember

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.pretrained_cnn = self.get_pretrained_cnn()

        self.memory_size = memory_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_q_net = DeepQNetwork(n_past_action_to_remember).to(self.device)
        self.target_q_net = DeepQNetwork(n_past_action_to_remember).to(self.device)
        self.target_q_net.load_state_dict(self.policy_q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.RMSprop(self.policy_q_net.parameters())
        self.memory = ReplayMemory(memory_size)

        self.t = 0
        self.history = self.clear_history()

    def get_pretrained_cnn(self):
        pretrained_cnn = models.vgg16(pretrained=True)
        # remove the last 3 layers to have 4096 ouput
        pretrained_cnn.classifier = nn.Sequential(*list(self.pretrained_cnn.classifier.children())[:-3])

        # make this model not trainable, so that is simply does features extraction
        for param in pretrained_cnn.parameters():
            param.requires_grad = False
        return pretrained_cnn

    def get_greedy_action(self, state):
        return self.policy_q_net(state).max(1)[1]  # .view(x,y)

    def get_action(self, state):
        self.t += 1
        threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.t / self.eps_decay)
        if random.random() > threshold:
            with torch.no_grad():
                return self.get_greedy_action(state)

        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # batch of transistions to transitions of batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Q(s_t, a)
        state_action_values = self.policy_q_net(batch.state).gather(1, batch.action)

        #V(s_{t+1}) by older q_net
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_q_net(non_final_next_states).max(1)[0].detach()

        # E[Q(s_t, a)]
        expected_state_action_values = (next_state_values * self.discount_rate) + batch.reward

        # Loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        #optimization
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_q_net.parameters():
        #     param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def get_state_from_observation(self, obs):
        return torch.cat((self.pretrained_cnn(obs), self.history))

    def update_history(self, action):
        self.history[self.n_action:] = self.history[:len(self.history)-self.n_action]
        self.history[:self.n_action] = torch.zeros(self.n_action)
        self.history[action] = torch.tensor([1], device=self.device)

    def clear_history(self):
        return torch.zeros(self.n_action * self.n_past_action_to_remember)


    def train_episode(self, env):
        t = 0
        self.history = self.clear_history()
        observation = env.reset()
        state = self.get_state_from_observation(observation)
        past_state = state
        done = False
        while not done:
            action = self.get_action(state)
            self.update_history(action)
            observation, reward, done, info = env.step(action)
            reward = torch.tensor([reward], device=self.device)
            state = self.get_state_from_observation(observation)
            self.memory.push(past_state, action, state, reward)
            past_state = state

            self.optimize_model()
            t+=1
        return t

    def train(self, env, nb_episode=100):
        episode_lenghts = []
        for episode in range(nb_episode):
            episode_lenght = self.train_episode(env)
            episode_lenghts.append(episode_lenght)
            if episode % self.target_update == 0 :
                self.target_q_net.load_state_dict(self.policy_q_net.state_dict())
        return episode_lenghts





