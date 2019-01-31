import numpy as np

class Bandits(object):
    def __init__(self, k):
        self.k = k
        self.q_star = np.random.normal(0,1,k)
        self.q_star_variance = 1.0

    def reward(self, arm):
        assert 0 <= arm < self.k
        return np.random.normal(self.q_star[arm], self.q_star_variance)

    def is_optimal_arm(self, arm):
        return arm == np.argmax(self.q_star)

    def new_q_star(self):
        self.q_star = np.random.normal(0, 1, self.k)


class AlgorithmBandit:
    def __init__(self, k, initial_value=0):
        self.k = k
        self.N = np.zeros(k)
        self.Q = np.zeros(k) + initial_value
        self.t = 0
        self.initial_value = initial_value

    def reset(self):
        self.N = np.zeros(self.k)
        self.Q = np.zeros(self.k) + self.initial_value
        self.t = 0

    def policy(self):
        pass

    def take_action(self, bandits):
        self.t += 1
        A = self.policy()
        R = bandits.reward(A)
        self.N[A] += 1
        self.Q[A] += (R - self.Q[A]) / self.N[A]
        return bandits.is_optimal_arm(A), R

def U(t, d, epsilon=0.5):
    m = (1+epsilon) * t
    return (1+ np.sqrt(epsilon)) * np.sqrt(m*np.log(np.log(m)/d) /(2*t))

class ActionEliminate(AlgorithmBandit):
    def __init__(self, k, initial_value=0):
        super().__init__(k, initial_value)

    def get_set(self):
        a = np.argmax(self.Q)

    def take_action(self, bandits):
        self.t += 1


class EpsilonGreedy(AlgorithmBandit):
    def __init__(self, k, epsilon=0, initial_value=0):
        super().__init__(k, initial_value)
        self.epsilon = epsilon

    def policy(self):
        if np.random.rand() < 1 - self.epsilon:
            return np.argmax(self.Q)
        else:
            return np.random.randint(0, self.k)


class OptimisticGreedy(EpsilonGreedy):
    def __init__(self, k, initial_value=0):
        super().__init__(k, epsilon=0, initial_value=initial_value)


class UpperConfidenceBound(AlgorithmBandit):
    def __init__(self, k, c=1, initial_value=0):
        super().__init__(k, initial_value)
        self.c = c  # degree of exploration

    def policy(self):
        never_tried = np.where(self.N == 0)[0]
        if never_tried.any():
            return never_tried[0]
        return np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))

class LowerUpperConfidenceBound(AlgorithmBandit):
    def __init__(self, k, c=0.1, initial_value=0):
        super().__init__(k, initial_value)
        self.c = c  # degree of exploration

    def policy(self):
        never_tried = np.where(self.N == 0)[0]
        if never_tried.any():
            return never_tried[0], never_tried[1]

        h = np.argmax(self.Q)
        temp = self.Q[h]
        self.Q[h] = np.NINF
        l = np.argmax(self.Q)
        self.Q[h] = temp
        return h, l

    def take_action(self, bandits):
        self.t += 1
        h, l = self.policy()
        Rh = bandits.reward(h)
        Rl = bandits.reward(l)
        self.N[h] += 1
        self.N[l] += 1
        self.Q[h] += (Rh - self.Q[h]) / self.N[h]
        self.Q[l] += (Rl - self.Q[l]) / self.N[l]
        return h, l
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class GradientAscent(AlgorithmBandit):
    def __init__(self, k, step_size=0.125, initial_value=0):
        super().__init__(k, initial_value)
        self.H = np.zeros(k)  # preference of each action
        self.step_size = step_size  # learning rate
        self.base_line = 0  # choose baseline = average of all reward

    def reset(self):
        super().reset()
        self.base_line = 0
        self.H = np.zeros(self.k)

    def take_action(self, bandits):
        self.t += 1
        policy = softmax(self.H)
        A = np.random.choice(self.k, p=policy)
        R = bandits.reward(A)

        for i in range(self.k):
            if i == A:
                self.H[i] += self.step_size * (R - self.base_line) * (1 - policy[i])
            else:
                self.H[i] -= self.step_size * (R - self.base_line) * policy[i]

        self.base_line += (R - self.base_line) / self.t
        return bandits.is_optimal_arm(A), R


def learning_curve(bandits, algorithm, nb_test, nb_step):
    rewards = np.zeros((nb_test, nb_step))
    is_optimal = np.zeros((nb_test, nb_step), bool)

    for test in range(nb_test):
        bandits.new_q_star()
        algorithm.reset()
        for step in range(nb_step):
            is_optimal[test, step], rewards[test, step] = algorithm.take_action(bandits)

    rewards_average = np.average(rewards, 0)
    is_optimal_average = np.count_nonzero(is_optimal, 0) / nb_test

    return is_optimal_average, rewards_average


def parameter_study(param, type_algo, k=10, nb_test=100, nb_step = 100):
    rewards = np.zeros((len(param), nb_test, nb_step))
    for i in range(len(param)):
        for j in range(nb_test):
            algo = type_algo(k, param[i])
            bandit = Bandits(k)
            for n in range(nb_step):
                _, rewards[i,j, n] = algo.take_action(bandit)

    return np.mean(np.mean(rewards, axis=1), axis=1)

if __name__ == '__main__':
    print("hello")