import numpy as np
from agent import ApproximateAgent, DiscreteAgent
from utils import  History, softmax


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
        state = self.state_from_observation(observation)
        done = False
        while not done:
            action = self.policy(env, state)
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation(observation)
            self.eligibity_trace = self.discount_rate * self.lambda_rate * self.eligibity_trace + self.FA.grad(state, self.weights)
            td_error = reward + self.discount_rate * self.FA(state_prime, self.weights) - self.FA(state, self.weights)
            self.weights += self.learning_rate * td_error * self.eligibity_trace
            state = state_prime

    def get_value(self, state):
        return self.FA(state, self.weights)


class TrueOnlineTD(ApproximateAgent):
    def __init__(self, env, discount_rate, learning_rate, lambda_rate, approximation_function, policy, state_from_observation_function = lambda x:x, reset_function = None):
        super().__init__(env, discount_rate, learning_rate, approximation_function, policy, state_from_observation_function)
        self.lambda_rate = lambda_rate
        self.eligibity_trace = np.zeros_like(self.weights)
        self.reset_function = reset_function

    def episode(self, env):
        observation = env.reset() if self.reset_function == None else self.reset_function(env)
        state = self.state_from_observation(observation)
        self.eligibity_trace = np.zeros_like(self.weights)
        x = self.FA.get_feature_vector(state)
        v_old = 0
        done = False
        while not done:
            action = self.policy(env, state)
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation(observation)
            x_prime = self.FA.get_feature_vector(state_prime)

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
        return self.FA(state, self.weights)


#################
# CONTROL
#################

class TrueOnlineSARSA(ApproximateAgent):
    def __init__(self, env, discount, learning_rate, lambda_rate, approximation_function, policy = None, state_from_observation_function = lambda x:x, reset_function = None):
        super().__init__(env, discount, learning_rate, approximation_function, policy, state_from_observation_function)
        self.lambda_rate = lambda_rate
        self.reset_function = reset_function


    def policy_test(self, env, state):
        epsilon = 0.9
        if np.random.rand(1) < epsilon:
            return self.policy_greedy(env, state)
        else:
            return np.random.randint(self.FA.action_space.n)

    def policy_greedy(self, env, state):
        q = [self.weights.T @ self.FA.get_feature_vector(state, action) for action in range(self.FA.action_space.n)]
        return np.argmax(q)

    def policy_softmax(self, env, state):
        a = np.random.choice(self.FA.action_space.n, 1, p=softmax([self.weights.T @ self.FA.get_feature_vector(state, action) for action in range(self.FA.action_space.n)]))[0]
        return a

    def episode(self, env):
        if self.policy == None:
            self.policy = self.policy_softmax
        observation = env.reset() if self.reset_function == None else self.reset_function(env)
        state = self.state_from_observation(observation)
        action = self.policy(env, state)
        x = self.FA.get_feature_vector(state, action)
        z = np.zeros_like(self.weights)
        q_old = 0
        done = False
        t = 0
        while not done:
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation(observation)
            action_prime = self.policy(env, state_prime)
            x_prime = self.FA.get_feature_vector(state_prime, action_prime)
            q = self.weights.T @ x
            q_prime = self.weights.T @ x_prime
            td_error = reward + self.discount_rate * q_prime - q
            z = self.discount_rate * self.lambda_rate * z + (1 - self.learning_rate * self.discount_rate * self.lambda_rate * z.T @ x) * x
            self.weights += self.learning_rate * (td_error + q - q_old) * z - self.learning_rate * (q - q_old) * x
            q_old = q_prime
            x = x_prime
            action = action_prime
            t += 1
            # if state[0] >= 0.5:
            #     break
        return t

class SemiGradientSARSA(ApproximateAgent):
    def __init__(self, env, discount_rate, learning_rate, lambda_rate, approximation_function, policy = None, state_from_observation_function = lambda x:x, reset_function = None):
        super().__init__(env, discount_rate, learning_rate, approximation_function, policy, state_from_observation_function)
        self.lambda_rate = lambda_rate
        self.eligibity_trace = np.zeros_like(self.weights)
        self.reset_function = reset_function


    def policy_epsilon_greedy(self, env, state):
        epsilon = 0.9
        p = np.random.rand(1)
        if p < epsilon:
            return self.policy_greedy(env, state)
        else:
            return np.random.randint(self.FA.action_space.n)

    def policy_greedy(self, env, state):
        q = [self.weights.T @ self.FA.get_feature_vector(state, action) for action in
             range(self.FA.action_space.n)]
        return np.argmax(q)


    def episode(self, env):
        if self.policy == None:
            self.policy = self.policy_epsilon_greedy
        observation = env.reset() if self.reset_function == None else self.reset_function(env)
        state = self.state_from_observation(observation)
        action = self.policy(env, state)
        t = 0
        while not done:
            observation, reward, done, info = env.step(action)
            state_prime = self.state_from_observation(observation)
            q = self.weights.T @ self.FA.get_feature_vector()
            grad = self.FA.grad(state, self.weights)
            if done:
                self.weights += self.learning_rate * (r - q)*grad
                break

            action_prime = self.policy(env, state_prime)
            q_prime = self.FA(state, self.weights)
            x_prime = self.FA.get_feature_vector(state_prime, action_prime)
            q = self.weights.T @ x
            q_prime = self.weights.T @ x_prime
            td_error = reward + self.discount_rate * q_prime - q
            self.eligibity_trace = self.discount_rate * self.lambda_rate * self.eligibity_trace + (
                    1 - self.learning_rate * self.discount_rate * self.lambda_rate * self.eligibity_trace.T @ x) * x
            self.weights += self.learning_rate * (td_error + q - q_old) * self.eligibity_trace - self.learning_rate * (q - q_old) * x
            q_old = q_prime
            x = x_prime
            action = action_prime
            t += 1
        return t

class SARSA(DiscreteAgent):
    def __init__(self, env, discount_rate, learning_rate, policy, state_from_observation_function = lambda x:x):
        super().__init__(env, discount_rate, learning_rate, policy, state_from_observation_function)

    def learn(self, s, a, r, s_prime, a_prime):
        self.qtable[s][a] += self.learning_rate * (r + self.discount_rate * self.qtable[s_prime][a_prime] - self.qtable[s][a])

    def update(self, env, s, a):
        observation, reward, done, info = env.step(a)
        s_prime = self.state_from_observation(observation)
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
        s_prime = self.state_from_observation(observation)
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
        s_prime = self.state_from_observation(observation)
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

