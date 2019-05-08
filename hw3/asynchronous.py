import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing as mp

import gym
from hw2 import function_approximation as FA

'''
exploration policy : epsilon greedy, different for each thread, epsilon smapled from a given distribution averey n step
'''


class ActorCriticModel(nn.Module):
    def __init__(self, nb_input, nb_action = 3):
        super(ActorCriticModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(nb_input, 16),
            nn.ReLU(),
            nn.Linear(16, 34),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(64, nb_action)
        self.citirc_linear = nn.Linear(64, 1)

    def forward(self, x):
        shared_representation = self.model(x)
        return self.actor_linear(shared_representation), self.citirc_linear(shared_representation)

def train(actor_critic_shared,
          rank,
          counter,
          lock,
          optimizer=None,
          seed=0,
          n_bins=32,
          n_tilings=1,
          discount_factor = 0.9,
          env_name = 'MountainCar-v0',
          max_episode_lenght = 200):

    #make the environment
    env = gym.make(env_name)

    #seed the environment
    manual_seed = rank + seed
    torch.manual_seed(manual_seed)
    env.seed(manual_seed)

    # fa = FA.TileCoding(n_bins=n_bins,
    #                    n_tilings=n_tilings,
    #                    observation_space=env.observation_space,
    #                    action_space=env.action_space)
    # feature_vector_dim = env.observation_space.shape[0] + env.action_space.n
    feature_vector_dim = env.observation_space.shape[0]
    actor_critic = ActorCriticModel(nb_input= feature_vector_dim, nb_action=env.action_space.n)

    if optimizer == None:
        optimizer = optim.Adam(actor_critic_shared.parameters())

    while True:
        actor_critic.load_state_dict(actor_critic_shared.state_dict())
        episode_lenght = 0 # t in the paper

        obs = env.reset()
        state = torch.from_numpy(obs)

        rewards = []
        states = [state]
        entopies = []
        state_values = []

        while True:
        # for step in range(max_step):
            action_values, state_value = actor_critic(state)
            actions_prob = F.softmax(action_values)
            actions_log_prob = F.log_softmax(action_values)
            entropy = -(actions_log_prob * actions_prob).sum()

            action = actions_prob.multinomial(num_samples=1).detach()
            action_log_prob = actions_log_prob.gather(1, action)

            obs, reward, done, _ = env.step(action.numpy()) #done says if it is a terminal state
            state = torch.from_numpy(obs)

            states.append(state)
            rewards.append(reward)
            entopies.append(entropy)
            state_values.append(state_value)

            with lock:
                counter.value += 1

            if done or episode_lenght == max_episode_lenght:
                break

        if done:
            R = 0
        else:
            _, state_value = actor_critic(state)
            R = state_value

        actor_loss = 0
        critic_loss = 0
        for i in reversed(len(rewards)):
            R = discount_factor * R + rewards[i]
            advantage = R - state_value[i]

            actor_loss += actions_log_prob[i] * advantage #use Generalized advantage estimation if I have the time
            critic_loss += advantage.pow(2)

        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()

        for param, shared_param in zip(actor_critic.parameters(),
                                       actor_critic_shared.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
        optimizer.step()

if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    actor_critic_shared = ActorCriticModel(nb_input=env.observation_space.shape[0],
                                           nb_action=env.action_space.n)
    actor_critic_shared.share_memory()

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    nb_process = 1
    processes = []
    for rank in range(0, nb_process):
        p = mp.Process(target=train,
                       args=(actor_critic_shared, rank, counter, lock))
        p.start()
        processes.append(p)
    [p.join() for p in processes]


