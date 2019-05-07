import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

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
