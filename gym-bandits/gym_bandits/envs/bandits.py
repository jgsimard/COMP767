import numpy as np
import gym
from gym import spaces

class Bandits(gym.Env):
    def __init__(self, n=4, range=(-1,1), means=-1, covs=-1):
        self.action_space = spaces.Discrete(n)
        inf = np.array([np.inf])
        self.observation_space = spaces.Box(low=-inf, high=inf, dtype=np.float32)

        if means == -1:
            r = range[1] - range[0]
            self.means = np.random.rand(n) * r - r/2
        else:
            self.means = means

        if covs != -1:
            self.covs = covs
        else:
            self.covs = np.ones(n)

    def step(self, action):
        return 0, np.random.normal(self.means[action], self.covs[action]), False, {}

