import numpy as np


def get_q(env, V, state, action, discount_factor=0.9):
    q=0
    for action_taken in range(env.action_space.n):
        s_prime = env.next_S[state][action_taken]
        q += env.P[state][action][action_taken] * (env.R[s_prime] + discount_factor * V[s_prime])
    return q


def get_V_PI_Q(env):
    V = np.zeros(env.observation_space.n)
    PI = np.random.randint(env.action_space.n, size=env.observation_space.n)
    Q = {state : np.zeros(env.action_space.n) for state in range(env.observation_space.n)}
    return V, PI, Q


def policy_evaluation(env, PI, V, discount_factor=0.9, epsilon=1e-5, modified_max_k=np.Inf):
    k = 0
    delta = epsilon + 1

    while delta > epsilon and (k < modified_max_k):

        delta = 0
        for s in range(env.observation_space.n):
            old_value = V[s]
            V[s] = get_q(env, V, s, PI[s], discount_factor)
            delta = np.max([delta, np.abs(old_value - V[s])])
        k += 1
    return k


def policy_improvement(env, PI, V, Q, discount_factor=0.9, epsilon=1e-5):
    policy_stable = True

    for s in range(env.observation_space.n):
        Q[s] = [get_q(env, V, s, action, discount_factor) for action in range(env.action_space.n)]
        old_value = PI[s]
        PI[s] = np.argmax(Q[s])

        if old_value != PI[s]:
            policy_stable = False
    return policy_stable


def policy_iteration(env, discount_factor=0.9, epsilon=1e-5, modified_max_k=np.Inf):
    V, PI, Q = get_V_PI_Q(env)
    policy_stable = False
    k = []

    while not policy_stable:
        last_k = policy_evaluation(env, PI, V, discount_factor, epsilon, modified_max_k)
        k.append(last_k)

        policy_stable = policy_improvement(env, PI, V, Q, discount_factor, epsilon)

    return V, PI, k


def value_iteration(env, discount_factor=0.9, epsilon=1e-4):
    V, PI, Q = get_V_PI_Q(env)

    k = 0
    delta = epsilon + 1
    while delta > epsilon:
        delta = 0
        for s in range(env.observation_space.n):
            Q[s] = [get_q(env, V, s, action, discount_factor) for action in range(env.action_space.n)]
            old_value = V[s]
            V[s] = np.max(Q[s])

            delta = np.max([delta, np.abs(old_value - V[s])])
        k += 1

    for s in range(env.observation_space.n):
        PI[s] = np.argmax(Q[s])

    return V, PI, k
