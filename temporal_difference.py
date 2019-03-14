import numpy as np
from agent import ApproximateAgent, DiscreteAgent
from utils import  History


#################
# PREDICTION
#################
class SemiGradientTD(ApproximateAgent):
    def __init__(self, env, discount_rate, learning_rate, lambda_rate, approximation_function, policy, state_from_observation_function = lambda x:x, reset_function = None):
        super().__init__(env, discount_rate, learning_rate, approximation_function, policy, state_from_observation_function)
        self.lambda_rate = lambda_rate
        self.eligibity_trace = np.zeros_like(self.weights)
        self.reset_function = reset_function

    def episode(self, env):
        self.eligibity_trace = np.zeros_like(self.weights)
        observation = env.reset() if self.reset_function == None else self.reset_function(env)
        state = self.state_from_observation_function(observation)
        done = False
        while not done:
            action = self.policy(env, state)
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation_function(observation)
            self.eligibity_trace = self.discount_rate * self.lambda_rate * self.eligibity_trace + self.approximation_function.grad(state, self.weights)
            td_error = reward + self.discount_rate * self.approximation_function(state_prime, self.weights) - self.approximation_function(state, self.weights)
            self.weights += self.learning_rate * td_error * self.eligibity_trace
            state = state_prime

    def get_value(self, state):
        return self.approximation_function(state, self.weights)


class TrueOnlineTD(ApproximateAgent):
    def __init__(self, env, discount_rate, learning_rate, lambda_rate, approximation_function, policy, state_from_observation_function = lambda x:x, reset_function = None):
        super().__init__(env, discount_rate, learning_rate, approximation_function, policy, state_from_observation_function)
        self.lambda_rate = lambda_rate
        self.eligibity_trace = np.zeros_like(self.weights)
        self.reset_function = reset_function

    def episode(self, env):
        observation = env.reset() if self.reset_function == None else self.reset_function(env)
        state = self.state_from_observation_function(observation)
        self.eligibity_trace = np.zeros_like(self.weights)
        x = self.approximation_function.get_feature_vector(state)
        v_old = 0
        done = False
        while not done:
            action = self.policy(env, state)
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation_function(observation)
            x_prime = self.approximation_function.get_feature_vector(state_prime)

            v = self.weights.T @ x
            v_prime = self.weights.T @ x_prime

            td_error = reward + self.discount_rate * v_prime - v
            self.eligibity_trace = self.discount_rate * self.lambda_rate * self.eligibity_trace + (
                        1 - self.learning_rate * self.discount_rate * self.lambda_rate * self.eligibity_trace.T @ x) * x

            self.weights += self.learning_rate * (td_error + v - v_old) * self.eligibity_trace - self.learning_rate * (
                        v - v_old) * x
            v_old = v_prime
            x = x_prime

    def get_value(self, state):
        return self.approximation_function(state, self.weights)


#################
# CONTROL
#################

class SARSA(DiscreteAgent):
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function = lambda x:x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)

    def learn(self, s, a, r, s_prime, a_prime):
        self.qtable[s][a] += self.learning_rate * (r + self.discount_rate * self.qtable[s_prime][a_prime] - self.qtable[s][a])

    def update(self, env, s, a):
        observation, reward, done, info = env.step(a)
        s_prime = self.state_from_observation_function(observation)
        a_prime = self.policy.get_action(self.qtable, s_prime)
        self.learn(s, a, reward, s_prime, a_prime)
        s = s_prime
        a = a_prime
        return s, a, reward, done

    def episode(self, env):
        s = env.reset()
        a = self.policy.get_action(self.qtable, s)
        history = History(s, a)
        done = False
        while not done:
            s, a, reward, done = self.update(env, s, a)
            history.register(s, a, reward)
        return history


class QLearning(DiscreteAgent):
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function = lambda x:x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)

    def learn(self, s, a, r, s_prime):
        self.qtable[s][a] += self.learning_rate * (r + self.discount_rate * self.qtable[s_prime][np.argmax(self.qtable[s_prime])] - self.qtable[s][a])

    def update(self, env, s):
        a = self.policy.get_action(self.qtable, s)
        observation, reward, done, info = env.step(a)
        s_prime = self.state_from_observation_function(observation)
        self.learn(s, a, reward, s_prime)
        s = s_prime
        return s, a, reward, done

    def episode(self, env):
        s = env.reset()
        history = History(s)
        done = False
        while not done:
            s, a, reward, done = self.update(env, s)
            history.register(s, a, reward)
        return history


class Expected_SARSA(DiscreteAgent):
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function=lambda x: x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)

    def learn(self, s, a, r, s_prime):
        expectation = np.sum(self.policy.get_action_probability_distribution(self.qtable, s_prime) * self.qtable[s_prime])
        self.qtable[s][a] += self.learning_rate * (r + self.discount_rate * expectation - self.qtable[s][a])

    def update(self, env, s):
        a = self.policy.get_action(self.qtable, s)
        observation, reward, done, info = env.step(a)
        s_prime = self.state_from_observation_function(observation)
        self.learn(s, a, reward, s_prime)
        s = s_prime
        return s, a, reward, done

    def episode(self, env):
        s = env.reset()
        history = History(s)
        done = False
        while not done:
            s, a, reward, done = self.update(env, s)
            history.register(s, a, reward)
        return history

