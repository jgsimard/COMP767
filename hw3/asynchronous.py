import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import math

import torch.multiprocessing as mp

import gym
from hw2 import function_approximation as FA


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class ActorCriticModel(nn.Module):
    def __init__(self, nb_input, nb_action = 3):
        super(ActorCriticModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(nb_input, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(128, nb_action)
        self.citirc_linear = nn.Linear(128, 1)

    def forward(self, x):
        shared_representation = self.shared(x)
        return self.actor_linear(shared_representation), self.citirc_linear(shared_representation)

def train(actor_critic_shared,
          rank,
          counter,
          lock,
          optimizer=None,
          seed=0,
          max_total_step = 10000,
          n_bins=32,
          n_tilings=1,
          discount_factor = 0.99,
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
        episode_length = 0 # t in the paper

        obs = env.reset()
        state = torch.from_numpy(obs).float().unsqueeze(0)

        action_log_probs =[]
        rewards = []
        states = [state]
        entopies = []
        state_values = []

        while True:
            episode_length += 1
            action_values, state_value = actor_critic(state)
            actions_prob = F.softmax(action_values, dim=-1)
            actions_log_prob = F.log_softmax(action_values, dim=-1)
            # entropy = -(actions_log_prob * actions_prob).sum(1, keepdim=True)

            action = actions_prob.multinomial(num_samples=1).detach()
            action_log_prob = actions_log_prob.gather(1, action)
            obs, reward, done, _ = env.step(action.numpy()[0,0]) #done says if it is a terminal state
            state = torch.from_numpy(obs).float().unsqueeze(0)

            action_log_probs.append(action_log_prob)
            states.append(state)
            rewards.append(reward)
            # entopies.append(entropy)
            state_values.append(state_value)

            with lock:
                counter.value += 1

            if done or episode_length == max_episode_lenght:
                break

        if done:
            R = 0
        else:
            _, state_value = actor_critic(state)
            R = state_value

        actor_loss = 0
        critic_loss = 0
        for i in reversed(range(len(rewards))):
            R = discount_factor * R + rewards[i]
            advantage = R - state_values[i]

            actor_loss += action_log_probs[i] * advantage #use Generalized advantage estimation if I have the time
            critic_loss += advantage.pow(2)

        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()

        for param, shared_param in zip(actor_critic.parameters(),
                                       actor_critic_shared.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad
        optimizer.step()

        print(f"Rank {rank}, episode length = {episode_length}")

        if counter.value > max_total_step:
            return


if __name__ == "__main__":
    seed = 1234
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    actor_critic_shared = ActorCriticModel(nb_input=env.observation_space.shape[0],
                                           nb_action=env.action_space.n)
    actor_critic_shared.share_memory()

    # shared_optim = False
    shared_optim = True

    if shared_optim:
        optimizer = SharedAdam(actor_critic_shared.parameters())
        optimizer.share_memory()
    else:
        optimizer = None


    counter = mp.Value("i", 0)
    lock = mp.Lock()

    nb_process = 8
    max_total_step = 100000
    processes = []
    for rank in range(0, nb_process):
        p = mp.Process(target=train, args=(actor_critic_shared, rank, counter, lock, optimizer, seed, max_total_step))
        p.start()
        processes.append(p)
    [p.join() for p in processes]


