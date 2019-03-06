import numpy as np

#################
# Policy
#################
class Policy:
    def __init__(self, n):
        self.n = n

    def get_action(self, q):
        pass

    def get_action_probability_distribution(self, q, s):
        pass


class EpsilonGreedy(Policy):
    def __init__(self, epsilon=0.1, n=6):
        super().__init__(n)
        self.epsilon = epsilon

    def get_action(self, q, s):
        p = np.random.rand(1)
        if p < self.epsilon:
            a = np.random.randint(self.n)
        else:
            a = np.argmax(q[s])
        return a

    def get_action_probability_distribution(self, q, s):
        out = np.ones(self.n) * self.epsilon / self.n
        out[np.argmax(q[s])] += 1 - self.epsilon
        return out


class Greedy(EpsilonGreedy):
    def __init__(self, n=6):
        super().__init__(epsilon=0, n=n)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


class SoftmaxExploration(Policy):
    def __init__(self, temperature_factor=1, n=6):
        super().__init__(n)
        self.temperature_factor = temperature_factor

    def get_action(self, q, s):
        return np.random.choice(self.n, 1, p=self.get_action_probability_distribution(q, s))[0]

    def get_action_probability_distribution(self, q, s):
        return softmax(q[s] / self.temperature_factor)


#################
# Control and Prediction Algorithms
#################

class History:
    def __init__(self, s=None, a=None, discount_rate=0.9):
        self.states = []
        self.actions = []
        self.rewards = []
        self.t = 0
        self.g = 0
        self.discount_rate = discount_rate

        self.register(s, a)

    def register(self, s=None, a=None, r=None):
        if s != None:
            self.states.append(s)
        if a != None:
            self.actions.append(a)
        if r != None:
            self.rewards.append(r)
            self.t += 1
            self.g += r * (self.discount_rate ** self.t)

    def undiscounted_return(self):
        return np.sum(self.rewards)

    def discounted_return(self):
        return self.g


def sarsa_update(env, policy, q, learning_rate, discount_rate):
    s = env.reset()
    a = policy.get_action(q, s)
    history = History(s, a)
    done = False
    while not done:
        observation, reward, done, info = env.step(a)
        s_prime = observation
        a_prime = policy.get_action(q, s_prime)
        q[s][a] = q[s][a] + learning_rate * (reward + discount_rate * q[s_prime][a_prime] - q[s][a])
        s = s_prime
        a = a_prime
        history.register(s, a, reward)
    return history


def qlearning_update(env, policy, q, learning_rate, discount_rate):
    s = env.reset()
    history = History(s)
    done = False
    while not done:
        a = policy.get_action(q, s)
        observation, reward, done, info = env.step(a)
        s_prime = observation
        q[s][a] = q[s][a] + learning_rate * (reward + discount_rate * q[s_prime][np.argmax(q[s_prime])] - q[s][a])
        s = s_prime
        history.register(s, a, reward)
    return history


def expected_sarsa_update(env, policy, q, learning_rate, discount_rate):
    s = env.reset()
    history = History(s)
    done = False
    while not done:
        a = policy.get_action(q, s)
        observation, reward, done, info = env.step(a)
        s_prime = observation
        expectation = np.sum(policy.get_action_probability_distribution(q, s_prime) * q[s_prime])
        q[s][a] = q[s][a] + learning_rate * (reward + discount_rate * expectation - q[s][a])
        s = s_prime
        history.register(s, a, reward)
    return history
