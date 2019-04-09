import random
from collections import namedtuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

from gym_project.envs.object_localization import  intersection_over_union

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
        # self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(1024, 1024)
        self.relu_2 = nn.ReLU()
        self.linear_out = nn.Linear(1024, n_action)

    def forward(self, x):
        out_l1 = self.dropout(self.relu_1(self.linear_1(x)))
        out_l2 = self.relu_2(self.linear_2(out_l1))
        out = self.linear_out(out_l2)
        return out


class Agent:
    def __init__(self, env, target_update=10, discout_rate=0.99, eps_start=0.9, eps_end=0.05, eps_decay=5,
                 batch_size=64, memory_size=1000, n_past_action_to_remember=10, device = None, save_path=""):
        self.target_update = target_update
        self.discount_rate = discout_rate
        self.n_action = env.action_space.n
        self.batch_size = batch_size
        self.n_past_action_to_remember = n_past_action_to_remember

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == None else device
        self.pretrained_cnn = self.get_pretrained_cnn()


        self.memory_size = memory_size
        self.policy_q_net = DeepQNetwork(n_past_action_to_remember).to(self.device)
        self.target_q_net = DeepQNetwork(n_past_action_to_remember).to(self.device)
        self.target_q_net.load_state_dict(self.policy_q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.policy_q_net.parameters())
        #self.optimizer = optim.SGD(self.policy_q_net.parameters())
        self.memory = ReplayMemory(memory_size)


        self.timestep_until_last_trigger_treshold = 40

        self.current_epoch = 0
        self.t = 0
        self.history = self.clear_history()
        self.save_path = save_path
        print("Agent initialization done")

    def save_model(self, path):
        torch.save(self.policy_q_net.state_dict(), os.path.join(path, 'best_policy_q_net.pt'))

    def load_model(self, path):
        self.policy_q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(torch.load(path, map_location=self.device))

    def get_pretrained_cnn(self):
        pretrained_cnn = models.vgg16(pretrained=True)
        pretrained_cnn.classifier = nn.Sequential(*list(pretrained_cnn.classifier.children())[:-2])
        for param in pretrained_cnn.parameters():
            param.requires_grad = False
        return pretrained_cnn.to(self.device)

    def get_greedy_action(self, state):
        return self.policy_q_net(state).max(1)[1].view(1, 1)

    def get_action(self, state, env):
        # self.t += 1
        #threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.t / self.eps_decay)
        threshold = max(self.eps_start - (self.eps_start - self.eps_end) / self.eps_decay * self.current_epoch, self.eps_end)

        if random.random() > threshold:
            with torch.no_grad():
                return self.get_greedy_action(state)
        else:
            positive_rewards=[]
            # positive_rewards = env.get_positive_reward_actions()
            # print("positive_rewards", positive_rewards)
            if len(positive_rewards)!= 0:
                action = random.choice(positive_rewards)
            else:
                action = random.randrange(self.n_action)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return
        # print("OPTIMIZING MODEL")

        # batch of transistions to transitions of batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).float()

        # Mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Q(s_t, a)
        state_action_values = self.policy_q_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}) by older q_net
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_q_net(non_final_next_states).max(1)[0].detach()

        # E[Q(s_t, a)]
        expected_state_action_values = (next_state_values * self.discount_rate) + reward_batch

        # Loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_q_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def get_state_from_observation(self, obs):
        obs_reshaped_for_cnn = torch.from_numpy(obs).permute(2,0,1).unsqueeze(0).float().to(self.device)/255
        features = self.pretrained_cnn(obs_reshaped_for_cnn)
        return torch.cat((features, self.history.unsqueeze(0)), 1)

    def update_history(self, action):
        self.history[self.n_action:] = self.history[:len(self.history) - self.n_action]
        self.history[:self.n_action] = torch.zeros(self.n_action)
        self.history[action] = torch.tensor([1], device=self.device)

    def clear_history(self):
        return torch.zeros(self.n_action * self.n_past_action_to_remember).to(self.device)

    def train_episode(self, env):
        t = 0
        self.history = self.clear_history()
        observation = env.reset()
        state = self.get_state_from_observation(observation)
        past_state = state
        done = False
        while not done:
            action = self.get_action(state, env)
            self.update_history(action)
            observation, reward, done, info = env.step(action)
            reward = torch.tensor([reward], device=self.device)
            state = self.get_state_from_observation(observation)
            self.memory.push(past_state, action, state, reward)
            past_state = state

            self.optimize_model()
            t += 1
        self.save_model(self.save_path)

        return t

    def train(self, env, nb_epoch=15):
        episode_lenghts = []
        for epoch in range(nb_epoch):
            print(f"Epoch:{epoch}")
            for episode in tqdm(range(env.epoch_size)):
                episode_lenght = self.train_episode(env)
                episode_lenghts.append(episode_lenght)
                if episode % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.policy_q_net.state_dict())
                # print(f"Episode : {episode}, len : {episode_lenght}")
            self.save_model(self.save_path)
            self.t+=1
        return episode_lenghts

    def test_episode(self, env):
        t = 0
        timestep_until_last_trigger = 0
        self.history = self.clear_history()
        observation = env.reset()
        state = self.get_state_from_observation(observation)
        done = False

        imgs = [observation]
        actions = ["reset"]
        rewards = [0]
        ious = [env.past_iou]

        while not done:
            action = self.get_greedy_action(state)
            self.update_history(action)
            observation, reward, done, info = env.step(action)
            state = self.get_state_from_observation(observation)

            imgs.append(observation)
            actions.append(env.action_index_to_names[action.item()])
            rewards.append(reward)
            ious.append(env.past_iou)


            if action == env.action_space.n - 1:
                timestep_until_last_trigger += 1
            if timestep_until_last_trigger == self.timestep_until_last_trigger_treshold:
                env.current_bb = env.restart_box()
                timestep_until_last_trigger = 0
                env.past_iou = intersection_over_union(env.current_bb, self.get_ground_truth_bb())
                state = self.get_state_from_observation(env.get_obs())
            t += 1
        return imgs, actions, rewards, ious, t

