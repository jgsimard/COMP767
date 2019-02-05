import numpy as np
import copy


#######################
# GRIDWORLD
#######################
class GridWorld:
    def __init__(self, world_size, terminal_states, rewards, p_desired_action):
        self.actions, self.states, self.nextState = self.create_grid_world(world_size, terminal_states)
        self.world_size = world_size
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.p_desired_action = p_desired_action
        self.index_to_actions = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}

    def create_grid_world(self, world_size, terminal_states):
        """
            world_size: height and width of the squared-shape gridworld
            return
                actions: list of str, possible actions
                states: list of coordinate tuples representing all non-terminal states
                nextState: list of list of dict, index 3 times to return the next state coordinate tuple
            """

        # left, up, right, down
        actions = ['L', 'U', 'R', 'D']

        # Next
        nextState = []
        for i in range(0, world_size):
            nextState.append([])
            for j in range(0, world_size):
                # Creates a dictionnary that
                next = dict()
                if i == 0:
                    next['U'] = (i, j)
                else:
                    next['U'] = (i - 1, j)

                if i == world_size - 1:
                    next['D'] = (i, j)
                else:
                    next['D'] = (i + 1, j)

                if j == 0:
                    next['L'] = (i, j)
                else:
                    next['L'] = (i, j - 1)

                if j == world_size - 1:
                    next['R'] = (i, j)
                else:
                    next['R'] = (i, j + 1)

                nextState[i].append(next)

        states = []
        for i in range(0, world_size):
            for j in range(0, world_size):
                if (i, j) in terminal_states:
                    continue
                else:
                    states.append((i, j))

        return actions, states, nextState

    def step(self, s, a):
        out=[]
        for action_index in range(len(self.actions)):
            action = self.index_to_actions[action_index]
            # print(s, s[0], s[1])
            newState = self.nextState[s[0]][s[1]][action]
            reward = self.rewards[newState]
            p = (1-self.p_desired_action)/len(self.actions)
            if(action_index == a):
                p += self.p_desired_action
            out.append((newState, reward,p))
        return out

    def get_next_state_reward(self, state, PI):
        p = np.random.rand(1)
        if p < self.p_desired_action:
            action = self.index_to_actions[PI[state]]
        else:
            action = np.random.choice(self.actions)
        new_state = self.nextState[state[0]][state[1]][action]
        reward = self.rewards[new_state]
        return new_state, reward


#######################
# ALGORITHMS
#######################
def get_q(grid_world, V, state, action, gamma=0.9):
    return np.sum([p * (r + gamma * V[newState]) for newState, r, p in grid_world.step(s=state, a=action)])


def get_V_Vnext_PI_Q(grid_world):
    V = np.zeros((grid_world.world_size, grid_world.world_size))
    V_next = np.zeros_like(V)
    PI = np.random.randint(len(grid_world.actions), size=(grid_world.world_size, grid_world.world_size))
    Q = np.zeros((grid_world.world_size, grid_world.world_size, 4), dtype=np.float)
    return V, V_next, PI, Q


def policy_iteration(grid_world, gamma=0.9, epsilon=1e-5, modified_max_k=np.Inf):
    V_k, V_kplus1, PI, Q = get_V_Vnext_PI_Q(grid_world)
    policy_stable = False
    all_k = []

    while not policy_stable:
        # POLICY EVALUATION (iterates until V_k converges)
        k = 0
        V_kplus1 = copy.deepcopy(V_k)
        delta = epsilon + 1

        while delta > epsilon and (k < modified_max_k):

            delta = 0
            for state in grid_world.states:
                V_kplus1[state] = get_q(grid_world, V_k, state, PI[state], gamma)
                delta = np.max([delta, np.abs(V_kplus1[state] - V_k[state])])

            # Updates our current estimate
            V_k = copy.deepcopy(V_kplus1)
            k += 1
        all_k.append(k)

        # POLICY IMPROVEMENT (greedy action selection with respect to V_k)
        policy_stable = True
        old_PI = copy.deepcopy(PI)

        for state in grid_world.states:
            Q[state[0], state[1], :] = [get_q(grid_world, V_k, state, action, gamma) for action in range(4)]
            PI[state] = np.argmax(Q[state[0], state[1], :])

            if old_PI[state] != PI[state]:
                policy_stable = False

    return {"V": V_k, "PI": PI, "k": all_k}


def value_iteration(grid_world, gamma=0.9, epsilon=1e-4):
    V_k, V_kplus1, PI, Q = get_V_Vnext_PI_Q(grid_world)

    # POLICY EVALUATION
    k = 0
    delta = epsilon + 1
    while delta > epsilon:

        delta = 0
        for state in grid_world.states:
            Q[state[0], state[1], :] = [get_q(grid_world, V_k, state, action, gamma) for action in range(4)]
            V_kplus1[state] = np.max(Q[state[0], state[1], :])

            delta = np.max([delta, np.abs(V_kplus1[state] - V_k[state])])

        V_k = copy.deepcopy(V_kplus1)
        k += 1

    for state in grid_world.states:
        PI[state] = np.argmax(Q[state[0], state[1], :])

    return {"V": V_k, "PI": PI, "k": k}


#######################
# ALGORITHMS TESTING
#######################
def get_true_value(PI, grid_world, start_state, gamma, iterations=100):
    g = np.zeros(iterations)

    for i in range(iterations):
        discounted_return = 0
        t = 0
        state = start_state
        while state not in grid_world.terminal_states:
            state, reward = grid_world.get_next_state_reward(state, PI)
            discounted_return += gamma ** t * reward
            t += 1
        g[i] = discounted_return
    return g.mean()


def policy_and_value_testing(results, grid_world, gamma):
    position_bl = (grid_world.world_size - 1, 0)
    position_br = (grid_world.world_size - 1, grid_world.world_size - 1)

    bl = get_true_value(results["PI"], grid_world, position_bl, gamma)
    br = get_true_value(results["PI"], grid_world, position_br, gamma)

    print("\n\nTesting")
    print("Bottom left. V :", results["V"][position_bl], ", True : ", bl)
    print("Bottom right. V :", results["V"][position_br], ", True : ", br)


def compute_algorithm(algorithm, algorithm_name, grid_world, gamma, modified_max_k=np.Inf):
    if modified_max_k != np.Inf:
        results = algorithm(grid_world, gamma, modified_max_k = modified_max_k)
    else:
        results = algorithm(grid_world, gamma)
    print("\n\n"+algorithm_name)
    clean_print(results, grid_world.terminal_states)
    policy_and_value_testing(results, grid_world, gamma)


def hyper_parameter_testing(world_size=5, p_desired_direction=0.7, gamma=0.9, epsilon=1e-5):
    print("HYPER PARAMETERS")
    print("world_size : ", world_size)
    print("p_desired_direction : ", p_desired_direction)
    print("gamma : ", gamma)
    print("epsilon : ", epsilon)

    terminal_states = [(0, 0), (0, world_size - 1)]
    rewards = np.zeros((world_size, world_size))
    rewards[0, 0] = 1
    rewards[0, world_size - 1] = 10

    grid_world = GridWorld(world_size, terminal_states, rewards, p_desired_direction)

    compute_algorithm(policy_iteration, "POLICY ITERATION", grid_world, gamma)
    compute_algorithm(policy_iteration, "MODIFIED POLICY ITERATION", grid_world, gamma, modified_max_k=2)
    compute_algorithm(value_iteration, "VALUE ITERATION", grid_world, gamma)


#######################
# PRINTING
#######################
def print_policy(policy, terminal_states):
    arrows = {0: '\u2190', 1: '\u2191', 2: '\u2192', 3: '\u2193'}
    terminal_state = '\u25A0'
    for i in range(policy.shape[0]):
        string = ""
        for j in range(policy.shape[1]):
            if (i, j) in terminal_states:
                string += terminal_state
            else:
                string += arrows[policy[i, j]]
        print(string)


def clean_print(results, terminal_states):
    if int == type(results["k"]):
        print("Policy found in {} iterations".format(results["k"]))
    else:
        print("Policy found in {} iterations, where each policy evaluation lasted for k = {}".format(len(results["k"]),
                                                                                                     results["k"]))

    print("\nV\n", np.round(results["V"], 1))
    print("\nPI")
    print_policy(results["PI"], terminal_states)

