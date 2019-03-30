import numpy as np
import gym
from gym import spaces

class GridWorld(gym.Env):
    def __init__(self, n=4, p_action_works=1, terminal_states = None):

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(n**2)
        self.S = self.encode(n-1, 0)

        if terminal_states is not None:
            self.terminal_states = terminal_states
        else:
            self.terminal_states = {self.encode(0,0):1, self.encode(0, n-1):10}

        def get_action():
            return np.ones(self.action_space.n)*(1-p_action_works)/self.action_space.n
        self.P = {state : {action: get_action() for action in range(self.action_space.n)} for state in range(self.observation_space.n)}
        self.next_S = {state :  [] for state in range(self.observation_space.n)}

        self.R = np.zeros(self.observation_space.n)
        for terminal_state in self.terminal_states:
            self.R[terminal_state] = self.terminal_states[terminal_state]


        def get_next_states(s):
            actions = np.arange(self.action_space.n)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]#up, right, down, left
            next_states =[]
            row, col = self.decode(s)
            for a, d in zip(actions, directions):
                next_row = max(0, min(row + d[0], n - 1))
                next_col = max(0, min(col + d[1], n - 1))
                next_states.append(self.encode(next_row, next_col))
            return next_states

        for s in range(self.observation_space.n):
            next_states = get_next_states(s)

            for a in range(self.action_space.n):
                if s in self.terminal_states:
                    self.P[s][a] = [0] * self.action_space.n
                    self.next_S[s] = [s] * self.action_space.n
                else:
                    self.P[s][a][a] += p_action_works
                    self.next_S[s] = next_states

    def encode(self, row, col):
        n = np.sqrt(self.observation_space.n).astype(int)
        assert 0 <= col < n
        assert 0 <= row < n
        return row * n + col

    def decode(self, s):
        n = np.sqrt(self.observation_space.n).astype(int)
        row = s // n
        col = s % n
        return row, col

    def set_state(self, row, col=None):
        if col != None:
            self.S = self.encode(row, col)
        else:
            self.S = row

    def step(self, action):
        direction = np.random.choice(self.action_space.n, p = self.P[self.S][action])
        self.S = self.next_S[self.S][direction]
        done = True if self.S in self.terminal_states else False
        reward = 0 if not done else self.terminal_states[self.S]
        return self.S, reward, done, {}

    def reset(self):
        n = np.sqrt(self.observation_space.n).astype(int)
        self.S = self.encode(n-1, 0)
