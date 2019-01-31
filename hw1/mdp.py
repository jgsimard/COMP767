import numpy as np
import copy

def create_gridworld(world_size, terminal_states):
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


class GridWorld:
    def __init__(self, world_size, terminal_states, rewards, p_desired_action):
        self.actions, self.states, self.nextState = create_gridworld(world_size, terminal_states)
        self.world_size = world_size
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.p_desired_action = p_desired_action
        self.index_to_actions = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}

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


def init(grid_world):
    V_k = np.zeros((grid_world.world_size, grid_world.world_size))
    PI = np.random.randint(4, size=(grid_world.world_size, grid_world.world_size))

    idx_to_a = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}


def policy_iteration(grid_world, p_desired_direction, gamma=0.9, epsilon=1e-5, modified_max_k=np.Inf):
    V_k = np.zeros((grid_world.world_size, grid_world.world_size))
    PI = np.random.randint(4, size=(grid_world.world_size, grid_world.world_size))

    policy_stable = False
    all_k = []
    idx_to_a = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}

    while not policy_stable:
        # POLICY EVALUATION (iterates until V_k converges)
        k = 0
        V_kplus1 = copy.deepcopy(V_k)
        delta = epsilon + 1

        while delta > epsilon and (k < modified_max_k):

            delta = 0
            for state in grid_world.states:
                V_kplus1[state] = np.sum([p*(r + gamma * V_k[newState]) for newState, r, p in grid_world.step(s=state, a=PI[state])])
                delta = np.max([delta, np.abs(V_kplus1[state] - V_k[state])])

            # Updates our current estimate
            V_k = copy.deepcopy(V_kplus1)
            k += 1
        all_k.append(k)

        # POLICY IMPROVEMENT (greedy action selection with respect to V_k)
        Q = np.zeros((grid_world.world_size, grid_world.world_size, 4), dtype=np.float)

        policy_stable = True
        old_PI = copy.deepcopy(PI)

        for i, j in grid_world.states:
            for action_index in range(4):
                a = idx_to_a[action_index]
                newPosition = grid_world.nextState[i][j][a]

                # Policy Improvement rule
                Q[i, j, action_index] = (grid_world.rewards[newPosition] + gamma * V_k[newPosition])

            PI[i, j] = np.argmax(Q[i, j, :])

            if old_PI[i, j] != PI[i, j]:
                policy_stable = False

    return {"V": V_k, "PI": PI, "k": all_k}


def value_iteration(grid_world, p_desired_direction, gamma=0.9, epsilon=1e-4):
    V_k = np.zeros((grid_world.world_size, grid_world.world_size))
    PI = np.random.randint(4, size=V_k.shape)

    idx_to_a = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}

    # POLICY EVALUATION
    k = 0
    V_kplus1 = np.zeros_like(V_k)
    delta = epsilon + 1
    Q = np.zeros((grid_world.world_size, grid_world.world_size, 4), dtype=np.float)
    while delta > epsilon:

        delta = 0
        # for state in grid_world.states:
        #     q = [p * (r + gamma * V_k[newState]) for newState, r, p in grid_world.step(s=state, a=PI[state])]
        #     V_kplus1[state] = np.max(q)
        #
        #     # Keeps biggest difference seen so far
        #     delta = np.max([delta, np.abs(V_kplus1[state] - V_k[state])])

        # WORKING VERSION
        for i, j in grid_world.states:
            policy_action = idx_to_a[PI[i, j]]

            def update_Q(a_i, i, j, p, target):
                a = idx_to_a[a_i]
                nP = grid_world.nextState[i][j][a]
                Q[i, j, target] += p * (grid_world.rewards[nP] + gamma * V_k[nP])

            for a_idx in range(4):
                Q[i, j, a_idx] = 0
                p = p_desired_direction
                update_Q(a_idx, i, j, p, a_idx)

                for random_action_i in range(4):
                    update_Q(random_action_i, i, j, (1 - p_desired_direction) / 4, a_idx)

            my_bob = [p * (r + gamma * V_k[newState]) for newState, r, p in grid_world.step(s=(i,j), a=PI[(i,j)])]
            print(np.round(my_bob,1), np.round(Q[i, j, :],1))
                    # This step replaces the poilicy improvement step
            V_kplus1[i, j] = np.max(Q[i, j, :])

            # Keeps biggest difference seen so far
            delta = np.max([delta, np.abs(V_kplus1[i, j] - V_k[i, j])])

        # Updates our current estimate
        V_k = copy.deepcopy(V_kplus1)
        k += 1

    # Updates the policy to be greedy with respect to the value function
    for i, j in grid_world.states:
        PI[i, j] = np.argmax(Q[i, j, :])

    return {"V": V_k, "PI": PI, "k": k}


def get_true_value(PI, grid_world, start_position, gamma, terminal_states, p_desired_direction, iterations=100):
    g = np.zeros(iterations)

    for i in range(iterations):
        discounted_return = 0
        idx_to_a = {0: 'L', 1: 'U', 2: 'R', 3: 'D'}
        t = 0
        position = start_position
        while position not in terminal_states:
            p = np.random.rand(1)
            if p < p_desired_direction:
                action = idx_to_a[PI[position]]
            else:
                action = idx_to_a[np.random.randint(4)]
            position = grid_world.nextState[position[0]][position[1]][action]
            discounted_return += gamma ** t * grid_world.rewards[position]
            t += 1
        g[i] = discounted_return
    return g.mean()


def policy_and_value_testing(results, grid_world, gamma, terminal_states, p_desired_direction):
    position_bl = (grid_world.world_size - 1, 0)
    position_br = (grid_world.world_size - 1, grid_world.world_size - 1)

    bl = get_true_value(results["PI"], grid_world, position_bl, gamma, terminal_states, p_desired_direction)
    br = get_true_value(results["PI"], grid_world, position_br, gamma, terminal_states, p_desired_direction)

    print("\n\nTesting")
    print("Bottom left. V :", results["V"][position_bl], "True : ", bl)
    print("Bottom right. V :", results["V"][position_br], "True : ", br)


def hyperparameter_testing(world_size=5, p_desired_direction=0.7, gamma=0.9, epsilon=1e-5):
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

    ############################################
    ## POLICY ITERATION
    ############################################
    policy_iteration_results = policy_iteration(grid_world, p_desired_direction)
    print("\n\nPOLICY ITERATION ")
    clean_print(policy_iteration_results, terminal_states)
    policy_and_value_testing(policy_iteration_results, grid_world, gamma, terminal_states, p_desired_direction)
    #
    # ############################################
    # ## MODIFIED POLICY ITERATION
    # ############################################
    # modified_policy_iteration_results = policy_iteration(grid_world, p_desired_direction, modified_max_k=2)
    # print("\n\nMODIFIED POLICY ITERATION ")
    # clean_print(modified_policy_iteration_results, terminal_states)
    # policy_and_value_testing(modified_policy_iteration_results, grid_world, gamma, terminal_states, p_desired_direction)

    ############################################
    ## VALUE ITERATION
    ############################################
    value_iteration_results = value_iteration(grid_world, p_desired_direction)
    print("\n\nVALUE ITERATION")
    clean_print(value_iteration_results, terminal_states)
    policy_and_value_testing(value_iteration_results, grid_world, gamma, terminal_states, p_desired_direction)

