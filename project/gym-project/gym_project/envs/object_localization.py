import numpy as np
import gluoncv
import gym
from gym import spaces

class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        return (self.x2-self.x1) * (self.y2-self.y1)


def intersection_over_union(bb1, bb2):
    intersection_bb = BoundingBox(x1 = max(bb1.x1, bb2.x1),
                                  y1 = max(bb1.y1, bb2.y1),
                                  x2 = min(bb1.x2, bb2.x2),
                                  y2 = min(bb2.y2, bb2.y2))
    iou = intersection_bb.area()/(bb1.area() + bb2.area() - intersection_bb.area())
    return iou


class ProjectEnv(gym.Env):
    def __init__(self, root="/home/jg/MILA/COMP767-Reinforcement_Learning/COMP767/project/data/VOCtrainval_06-Nov-2007/VOCdevkit"):

        voc_dataset = gluoncv.data.VOCDetection(root=root, splits=[(2007, 'trainval')])
        action_low = 0
        action_high = 1
        self.action_space = spaces.Tuple([spaces.Box(low=action_low, high=action_high), spaces.Discrete(2)])

        self.output_image_size = 224
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.output_image_size, self.output_image_size, 3))


    def step(self, action):
        direction = np.random.choice(self.action_space.n, p = self.P[self.S][action])
        self.S = self.next_S[self.S][direction]
        done = True if self.S in self.terminal_states else False
        reward = 0 if not done else self.terminal_states[self.S]
        return self.S, reward, done, {}

    def reset(self):
        n = np.sqrt(self.observation_space.n).astype(int)
        self.S = self.encode(n-1, 0)