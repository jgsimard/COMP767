import numpy as np
from utils import softmax

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


class SoftmaxExploration(Policy):
    def __init__(self, temperature_factor=1, n=6):
        super().__init__(n)
        self.temperature_factor = temperature_factor

    def get_action(self, q, s):
        return np.random.choice(self.n, 1, p=self.get_action_probability_distribution(q, s))[0]

    def get_action_probability_distribution(self, q, s):
        return softmax(q[s] / self.temperature_factor)
