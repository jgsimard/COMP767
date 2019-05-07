import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import gym
from hw2 import function_approximation as FA

'''
exploration policy : epsilon greedy, different for each thread, epsilon smapled from a given distribution averey n step
'''


class ActorCriticModel1(nn.Module):
    def __init__(self, feature_vector_dim, nb_action = 3):
        super(ActorCriticModel1, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_vector_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 34),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.actor_linear(64, nb_action)
        self.citirc_linear(64, 1)

    def forward(self, x):
        shared_representation = self.model(x)
        return self.actor_linear(shared_representation), self.citirc_linear(shared_representation)

def train(actor_critic_shared, rank, optimizer=None, counter, lock, seed=0, n_bins=32, n_tilings=1, discount_factor = 0.9):
    manual_seed = rank + seed
    torch.manual_seed(manual_seed)
    env = gym.make('MountainCar-v0')
    env.seed(manual_seed)
    fa = FA.TileCoding(n_bins=n_bins,
                       n_tilings=n_tilings,
                       observation_space=env.observation_space,
                       action_space=env.action_space)
    feature_vector_dim = env.observation_space.shape[0] + env.action_space.n
    actor_critic = ActorCriticModel1(feature_vector_dim = feature_vector_dim, nb_action=env.action_space.n)

    if optimizer == None:
        optimizer = optim.Adam(actor_critic_shared.parameters())



    max_episode_lenght = 200
    while True:
        actor_critic.load_state_dict(actor_critic_shared.state_dict())
        is_terminal_state = False
        done = False
        episode_lenght = 0
        rewards =[]
        states = []
        entopies = []
        state_values = []

        obs = env.reset()
        state = torch.from_numpy(obs)

        states.append(state)
        while True:
        # for step in range(max_step):
            action_values, state_value = actor_critic(state)
            actions_prob = F.softmax(action_values)
            actions_log_prob = F.log_softmax(action_values)
            entropy = -(actions_log_prob * actions_prob).sum()

            action = actions_prob.multinomial(num_samples=1).detach()
            action_log_prob = actions_log_prob.gather(1, action)

            obs, reward, done, _ = env.step(action.numpy())
            state = torch.from_numpy(obs)
            is_terminal_state = done

            states.append(state)
            rewards.append(reward)
            entopies.append(entropy)
            state_values.append(state_value)

            with lock:
                counter.value += 1

            if is_terminal_state or episode_lenght == max_episode_lenght:
                break

        if is_terminal_state:
            R = 0
        else:
            _, state_value = actor_critic(state)
            R = state_value

        actor_loss = 0
        critic_loss = 0
        for i in reversed(len(rewards)):
            R = discount_factor * R + rewards[i]
            advantage = R - state_value[i]

            critic_loss += advantage.pow(2)
            #use Generalized advantage estimation if I have the time
            actor_loss += actions_log_prob[i] * advantage
        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()

        for param, shared_param in zip(actor_critic.parameters(),
                                       actor_critic_shared.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
        optimizer.step()