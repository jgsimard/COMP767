import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import math

import torch.multiprocessing as mp

import gym
from hw2 import function_approximation as FA

import numpy as np


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

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



class ActorCriticModel(nn.Module):
    def __init__(self, nb_input, nb_action = 3):
        super(ActorCriticModel, self).__init__()
        self.actor_linear = nn.Linear(nb_input, nb_action)
        self.citirc_linear = nn.Linear(nb_input, 1)

    def forward(self, x):
        return self.actor_linear(x), self.citirc_linear(x)

# class ActorCriticModel(nn.Module):
#     def __init__(self, nb_input, nb_action = 3):
#         super(ActorCriticModel, self).__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(nb_input, 128),
#             nn.ReLU(),
#         )
#         self.actor_linear = nn.Linear(128, nb_action)
#         self.citirc_linear = nn.Linear(128, 1)
#
#         self.apply(weights_init)
#
#     def forward(self, x):
#         shared_representation = self.shared(x)
#         return self.actor_linear(shared_representation), self.citirc_linear(shared_representation)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            shared_param._grad += param.grad
        shared_param._grad = param.grad


def state_from_obs(obs, fa):
    return torch.from_numpy(fa.get_state_feature_vector(obs)).float().unsqueeze(0)

def train(actor_critic_shared,
          rank,
          counter,
          lock,
          episode_counter,
          episode_lock,
          fa,
          optimizer=None,
          seed=0,
          max_total_step = 10000,
          learning_rate=1e-3,
          entropy_coef = 0.01,
          discount_factor = 0.999,
          env_name = 'MountainCar-v0',
          max_episode_lenght = 200):

    #make the environment
    env = gym.make(env_name)

    #seed the environment
    manual_seed = rank + seed
    torch.manual_seed(manual_seed)
    env.seed(manual_seed)

    actor_critic = ActorCriticModel(nb_input= fa.size, nb_action=env.action_space.n)
    actor_critic.train()

    if optimizer == None:
        optimizer = optim.Adam(actor_critic_shared.parameters(), lr=learning_rate)

    while True:
        actor_critic.load_state_dict(actor_critic_shared.state_dict())
        # print("not shared \n", actor_critic.actor_linear.weight)
        episode_length = 0 # t in the paper

        obs = env.reset()
        state = state_from_obs(obs, fa)

        action_log_probs =[]
        rewards = []
        states = [state]
        entropies = []
        state_values = []

        while True:
            episode_length += 1
            # print(state)
            actor_values, critic_value = actor_critic(state)
            actions_prob = F.softmax(actor_values, dim=-1)
            actions_log_prob = F.log_softmax(actor_values, dim=-1)
            entropy = -(actions_log_prob * actions_prob).sum(1, keepdim=True)

            action = actions_prob.multinomial(num_samples=1).detach()
            obs, reward, done, _ = env.step(action.numpy()[0,0]) #done says if it is a terminal state
            state = state_from_obs(obs, fa)

            action_log_probs.append(actions_log_prob.gather(1, action))
            states.append(state)
            rewards.append(reward)
            entropies.append(entropy)
            state_values.append(critic_value)

            with lock:
                counter.value += 1

            if done or episode_length == max_episode_lenght:
                break

        if done:
            R = torch.zeros(1,1)
        else:
            _, critic_value = actor_critic(state)
            R = critic_value.detatch()

        state_values.append(R)
        actor_loss = 0
        critic_loss = 0
        for i in reversed(range(len(rewards))):
            R = discount_factor * R + rewards[i]
            advantage = R - state_values[i]

            actor_loss = actor_loss - action_log_probs[i] * advantage - entropy_coef * entropies[i]#use Generalized advantage estimation if I have the time
            critic_loss += advantage.pow(2)

        # print(f"actor_loss={actor_loss}, critic_loss={critic_loss}")
        # print(R)
        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        # print(f"total_loss={total_loss}")
        # print(actor_critic.actor_linear.weight.grad)

        ensure_shared_grads(actor_critic, actor_critic_shared)

        # print(actor_critic.actor_linear.weight)
        # print(actor_critic_shared.actor_linear.weight)
        optimizer.step()
        # print(actor_critic.actor_linear.weight)
        # print(actor_critic_shared.actor_linear.weight)

        with episode_lock:
            episode_counter.value += 1
            print(f"Rank {rank}, episode_counter={episode_counter.value}, episode length = {episode_length}")

        if counter.value > max_total_step:
            print("not shared, end \n", actor_critic.actor_linear.weight)
            return


if __name__ == "__main__":
    seed = 123
    env_name = 'MountainCar-v0'
    # env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    n_bins = 8
    n_tilings = 5
    learning_rate = 1e-1
    entropy_coef = 0.1
    discount_factor = 0.99

    fa = FA.TileCoding(n_bins=n_bins,
                       n_tilings=n_tilings,
                       observation_space=env.observation_space)
    actor_critic_shared = ActorCriticModel(nb_input=fa.size,
                                           nb_action=env.action_space.n)
    actor_critic_shared.share_memory()
    print("shared\n", actor_critic_shared.actor_linear.weight)


    # shared_optim = False
    shared_optim = True

    if shared_optim:
        optimizer = SharedAdam(actor_critic_shared.parameters(), lr=learning_rate)
        optimizer.share_memory()
    else:
        optimizer = None


    counter = mp.Value("i", 0)
    lock = mp.Lock()

    episode_counter = mp.Value("i", 0)
    episode_lock = mp.Lock()

    nb_process = 8
    max_total_step = 100000000
    processes = []
    for rank in range(0, nb_process):
        p = mp.Process(target=train,
                       args=(actor_critic_shared,
                             rank,
                             counter,
                             lock,
                             episode_counter,
                             episode_lock,
                             fa,
                             optimizer,
                             seed,
                             max_total_step,
                             learning_rate,
                             entropy_coef,
                             discount_factor,
                             env_name))
        p.start()
        processes.append(p)
    [p.join() for p in processes]

    # print(actor_critic_shared.actor_linear.weight)


