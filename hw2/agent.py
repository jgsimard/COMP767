import numpy as np
from utils import  History


class Agent:
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function = lambda x:x):
        self.action_space = env.action_space
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.policy = policy
        self.state_from_observation = state_from_observation_function

class DiscreteAgent(Agent):
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function = lambda x:x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)
        self.qtable = {state: np.zeros(env.action_space.n) for state in range(env.observation_space.n)}

class ApproximateAgent(Agent):
    def __init__(self, env, discount_rate, learning_rate, approximation_function, policy, state_from_observation_function = lambda x:x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)
        self.FA = approximation_function
        self.weights = np.random.rand(approximation_function.size)
