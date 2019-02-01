import numpy as np
import matplotlib.pyplot as plt

###############################################################
# Q1 b
###############################################################

class NormalBandit:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def reward(self):
        return np.random.normal(self.mu, self.sigma, 1)[0]


def softmax_2d(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis)[:, None])
    return e_x / np.sum(e_x, axis=axis)[:, None]


class BanditTrials:

    def __init__(self, bandits, n_trials=10, n_time_steps=100):
        self.n_trials = n_trials
        self.n_time_steps = n_time_steps
        self.total_trial_results = []
        self.bandits = bandits

    def run_trials(self, strategy):
        self.total_trial_results = []
        h1 = self.H1([b.mu for b in self.bandits])
        for trial_num in range(self.n_trials):
            trial = strategy(self.bandits)
            trial.run_trial(time_steps=self.n_time_steps)
            self.total_trial_results.append(trial.pull_count_per_timestep / h1)
            print("Trial {} of {} complete".format(trial_num + 1, self.n_trials), end='\r')

    def H1(self, true_means):
        #Hardness
        delta = np.max(true_means) - true_means
        return np.sum(np.power(np.delete(delta, 0), -2))

    def results_as_probability(self):
        return softmax_2d(np.mean(self.total_trial_results, axis=0), axis=1)


def plot_bandits(results):
    fig, ax = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    time_steps = len(np.array(results).T[0])

    plt.plot(np.arange(0, time_steps), [ts[0] for ts in results[:time_steps]], label="$\mu_1 = 1$")
    plt.plot(np.arange(0, time_steps), [ts[1] for ts in results[:time_steps]], label="$\mu_2 = 0.8$")
    plt.plot(np.arange(0, time_steps), [ts[2] for ts in results[:time_steps]], label="$\mu_3 = 0.6$")
    plt.plot(np.arange(0, time_steps), [ts[3] for ts in results[:time_steps]], label="$\mu_4 = 0.4$")
    plt.plot(np.arange(0, time_steps), [ts[4] for ts in results[:time_steps]], label="$\mu_5 = 0.2$")
    plt.plot(np.arange(0, time_steps), [ts[5] for ts in results[:time_steps]], label="$\mu_6 = 0$")
    plt.legend()
    plt.show()


class Arm:
    def __init__(self, index, mean):
        self.index = index
        self.mean = mean


class AlgorithmBanditTrial:
    def __init__(self, bandits):
        """
        bandit_means: a list of means that will be used to build the bandits
        r_k = 1     : Number of samples per epoch for each arm.
        """
        self.bandits = bandits

        self.bandit_count = len(self.bandits)
        self.k = self.bandit_count

        bandit_means = [b.mu for b in bandits]
        self.optimal_bandit = np.argmax(bandit_means)
        self.rewards_per_arm = [[] for _x in np.arange(0, self.bandit_count)]
        self.delta = 0.1
        self.active_bandits = np.ones(self.bandit_count)
        self.pull_count_per_timestep = []

    def empirical_mean(self, bandit_index):
        if len(self.rewards_per_arm[bandit_index]) == 0:
            return -np.Inf
        return np.mean(self.rewards_per_arm[bandit_index])

    def active_bandit_indexes(self):
        return np.nonzero(self.active_bandits)[0]

    def estimated_best_bandit_mean(self):
        """ returns a tuple with the best bandit index and the empirical mean"""
        all_empirical_means = [self.empirical_mean(idx) for idx, rewards in enumerate(self.bandits)]
        best_arm_index = np.nanargmax(all_empirical_means)
        return (best_arm_index, all_empirical_means[best_arm_index])

    def arm(self, idx):
        return self.bandits[idx]

    def pull_arm(self, idx):
        return self.arm(idx).reward()

    def drop_arm(self, idx):
        self.active_bandits[idx] = 0

    def C_ik(self, bandit_index):
        k = len(self.rewards_per_arm[bandit_index])
        n = self.bandit_count
        if k == 0:
            return 0

        A = np.pi ** 2 / 3
        B = n * (k ** 2) / self.delta

        return np.sqrt(np.log(A * B) / k)

    def print_stopping_condition(self, step):
        mean = self.estimated_best_bandit_mean()
        print("Stopping. Best Arm: {}. Found in {} time steps".format(mean[0], step))
        print("Estimated mean: {}. ".format(mean[1]))
        print("Empirical mean: {}. ".format(self.arm(self.optimal_bandit).mu))

    def best_filtered_bandit_index(self, bandit_indexes):
        results = [mean for idx, mean in enumerate(self.all_empirical_means()) if idx in bandit_indexes]
        return bandit_indexes[np.argmax(results)], results

    def all_empirical_means(self):
        return [self.empirical_mean(idx) for idx, rewards in enumerate(self.bandits)]

    def best_filtered_bandit_index_with_C_ik(self, bandit_indexes):
        results = [mean + self.C_ik(idx) for idx, mean in enumerate(self.all_empirical_means()) if
                   idx in bandit_indexes]
        return bandit_indexes[np.argmax(results)], results

    def get_h_and_l(self):
        h_index, results = self.best_filtered_bandit_index(np.arange(0, self.bandit_count))
        h_mean = results[h_index]

        filtered_indexes = np.delete(np.arange(self.bandit_count), h_index)

        l_index, _ = self.best_filtered_bandit_index_with_C_ik(filtered_indexes)
        l_mean = results[l_index]

        return Arm(h_index, h_mean), Arm(l_index, l_mean)


class ActionEliminationBanditTrial(AlgorithmBanditTrial):
    def __init__(self, bandits):
        super().__init__(bandits)

    def run_trial(self, time_steps=500):
        current_epoch = 0
        active_bandits_for_epoch = self.active_bandit_indexes()
        for step in np.arange(0, time_steps):

            for bandit_index in active_bandits_for_epoch:

                self.rewards_per_arm[bandit_index].append(self.pull_arm(bandit_index))

                reference_arm = self.estimated_best_bandit_mean()
                reference_C_t = self.C_ik(reference_arm[0])

                for bandit_idx in self.active_bandit_indexes():
                    candidate_arm_mean = self.empirical_mean(bandit_idx)
                    candidate_C_t = self.C_ik(bandit_idx)
                    lhs = reference_arm[1] - reference_C_t
                    rhs = candidate_arm_mean + candidate_C_t
                    if lhs >= rhs and rhs > (-np.inf):
                        self.drop_arm(bandit_idx)

            if current_epoch > 0:
                self.pull_count_per_timestep.append(
                    [len(self.rewards_per_arm[idx]) for idx, _b in enumerate(self.bandits)])

            if step > 0 and step % (self.k - 1) == 0:
                active_bandits_for_epoch = self.active_bandit_indexes()
                current_epoch += 1


class UCBBanditTrial(AlgorithmBanditTrial):
    def __init__(self, bandits):
        super().__init__(bandits)

    def run_trial(self, time_steps=500):
        for step in np.arange(0, time_steps):

            # check to see if we haven't sampled a bandit yet:
            unexplored = np.where(np.isinf(self.all_empirical_means()))[0]

            if len(unexplored) != 0:
                best_bandit_index = unexplored[0]

            else:
                best_bandit_index, results = self.best_filtered_bandit_index_with_C_ik(np.arange(0, self.bandit_count))

                h, l = self.get_h_and_l()

                lhs = h.mean - self.C_ik(h.index)
                rhs = l.mean + self.C_ik(l.index)

                if lhs > rhs:
                    self.print_stopping_condition(step)
                    break

            self.rewards_per_arm[best_bandit_index].append(self.pull_arm(best_bandit_index))
            self.pull_count_per_timestep.append([len(self.rewards_per_arm[idx]) for idx, _b in enumerate(self.bandits)])


class LUCBBanditTrial(AlgorithmBanditTrial):
    def __init__(self, bandits):
        super().__init__(bandits)

    def run_trial(self, time_steps=500):
        for step in np.arange(0, time_steps):

            # check to see if we haven't sampled a bandit yet:
            unexplored = np.where(np.isinf(self.all_empirical_means()))[0]

            if len(unexplored) != 0:
                # grab the next one:
                arm = unexplored[0]
                self.rewards_per_arm[arm].append(self.pull_arm(arm))

            else:
                h, l = self.get_h_and_l()
                lhs = h.mean - self.C_ik(h.index)
                rhs = l.mean + self.C_ik(l.index)
                if lhs > rhs:
                    self.print_stopping_condition(step)
                    break

                self.rewards_per_arm[h.index].append(self.pull_arm(h.index))
                self.rewards_per_arm[l.index].append(self.pull_arm(l.index))

            self.pull_count_per_timestep.append([len(self.rewards_per_arm[idx]) for idx, _b in enumerate(self.bandits)])


def trial_bandit(algorithm, n_trials=2, n_time_steps=2500):
    BANDIT_MEANS = [ 1, 4/5, 3/5, 2/5, 1/5, 0]
    SIGMA = 1/4
    bandits = [NormalBandit(mean, SIGMA) for mean in BANDIT_MEANS]
    trials = BanditTrials(bandits, n_trials, n_time_steps)
    trials.run_trials(algorithm)

    results = trials.results_as_probability()
    plot_bandits(results)



###############################################################
# Q1 C
###############################################################


class BanditMachine(object):
    def __init__(self, k):
        self.k = k
        self.q_star = np.random.normal(0, 1, k)
        self.q_star_variance = 1.0

    def pull_arm(self, arm):
        assert 0 <= arm < self.k
        return np.random.normal(self.q_star[arm], self.q_star_variance)

    def optimal_action(self):
        return np.argmax(self.q_star)


class AlgorithmBandit:
    def __init__(self, k, initial_value=0):
        self.k = k
        self.N = np.zeros(k)
        self.Q = np.zeros(k) + initial_value
        self.t = 0
        self.rewards = []
        self.actions = []
        self.regrets = []

    def policy(self):
        pass

    def update_N_Q(self, A, R):
        self.N[A] += 1
        self.Q[A] += (R - self.Q[A]) / self.N[A]

    def record(self, bandit_machine, A, R):
        self.actions.append(A)
        self.rewards.append(R)
        self.regrets.append(bandit_machine.q_star[bandit_machine.optimal_action()] - R)

    def take_action(self, bandit_machine):
        set_A = self.policy()
        for A in set_A:
            self.t += 1
            R = bandit_machine.pull_arm(A)
            self.update_N_Q(A, R)

            self.record(bandit_machine, A, R)


class EpsilonGreedy(AlgorithmBandit):
    def __init__(self, k, epsilon=0, initial_value=0):
        super().__init__(k, initial_value)
        self.epsilon = epsilon

    def policy(self):
        if np.random.rand() < 1 - self.epsilon:
            return [np.argmax(self.Q)]
        else:
            return [np.random.randint(0, self.k)]


class OptimisticGreedy(EpsilonGreedy):
    def __init__(self, k, initial_value=0):
        super().__init__(k, epsilon=0, initial_value=initial_value)


class UpperConfidenceBound1(AlgorithmBandit):
    def __init__(self, k, c=1, initial_value=0):
        super().__init__(k, initial_value)
        self.c = c  # degree of exploration

    def policy(self):
        never_tried = np.where(self.N == 0)[0]
        if never_tried.any():
            return [never_tried[0]]
        return [np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))]


def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis))
    return e_x / np.sum(e_x, axis=axis)


class GradientAscent(AlgorithmBandit):
    def __init__(self, k, step_size=0.125, initial_value=0):
        super().__init__(k, initial_value)
        self.H = np.zeros(k)  # preference of each action
        self.step_size = step_size  # learning rate
        self.base_line = 0  # choose baseline = average of all reward

    def take_action(self, bandit_machine):
        self.t += 1
        policy = softmax(self.H)
        A = np.random.choice(self.k, p=policy)
        R = bandit_machine.pull_arm(A)

        for i in range(self.k):
            if i == A:
                self.H[i] += self.step_size * (R - self.base_line) * (1 - policy[i])
            else:
                self.H[i] -= self.step_size * (R - self.base_line) * policy[i]

        self.base_line += (R - self.base_line) / self.t

        self.record(bandit_machine, A, R)


def parameter_study(param, type_algo, k=10, nb_test=100, nb_step = 100):
    rewards = np.zeros((len(param), nb_test, nb_step))
    for i in range(len(param)):
        for j in range(nb_test):
            algo = type_algo(k, param[i])
            bandit = BanditMachine(k)
            for n in range(nb_step):
                algo.take_action(bandit)
            rewards[i,j,:] = algo.rewards

    return np.mean(np.mean(rewards, axis=1), axis=1)


def plot_parameter_study(nb_test = 100, nb_step = 100, k=10):
    param = [2**i for i in range(-7, 5, 1)]
    algos = [EpsilonGreedy, GradientAscent, UpperConfidenceBound1, OptimisticGreedy]

    parameter_study_values={}
    for algo in algos:
        parameter_study_values[algo.__name__] = parameter_study(param, algo, k, nb_test, nb_step)

    ####### PLOT

    for algo_name in parameter_study_values:
        plt.plot(param, parameter_study_values[algo_name], label=algo_name)
    plt.xscale("log", basex=2)
    plt.xlabel("epsilon/learning rate/c/Q_0")
    plt.ylabel("Average reward over 1000 steps")
    plt.legend()
    plt.show()