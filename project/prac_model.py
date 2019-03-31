# Importer les librairies
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.models as models


# Commencer le modèle.Je fait peur quand je commence les choses et
# J'essayer de construire un class dans python
class DeepQNetwork(nn.Module):
    def __init__(self, alpha):
        super(DeepQNetwork, self).__init__()
        self.linear_1 = nn.Linear(4096 + 90, 1024)
        self.relu_1 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(1024, 1024)
        self.relu_2 = nn.ReLU()
        self.linear_out = nn.Linear(1024, 9)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out_l1 = self.dropout(self.relu_1(self.linear_1(x)))
        out_l2 = self.relu_2(self.linear_2(out_l1))
        out = self.linear_out(out_l2)
        return out


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05,
                 replace=10000, actionSpace=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.action_history = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

        self.pretrained_cnn = models.vgg16(pretrained=True)
        # remove the last 3 layers to have 4096 ouput
        self.pretrained_cnn.classifier = nn.Sequential(*list(self.pretrained_cnn.classifier.children())[:-3])


    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:  # store the action history
            self.action_history.append([state, action, reward, state_])
        else:
            self.action_history[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        rand = np.random.random()  # Epsilon greedy
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.epsilon:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q)  # Initializing

        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize - batch_size - 1)))
        miniBatch = self.action_history[memStart:memStart + batch_size]  # initialize the memory
        memory = np.array(miniBatch)

    # convert to list because memory is an array of numpy objects
    Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)  # state vector
    Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)  # next state

    maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
    rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
    Qtarget = Qpred
    Qtarget[:, maxA] = rewards + self.GAMMA * T.max(Qnext[1])  # learning rule

    if self.steps > 500:
        if self.EPSILON - 1e-4 > self.EPS_END:
            self.EPSILON -= 1e-4
        else:
            self.EPSILON = self.EPS_END

    # Qpred.requires_grad_()
    loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)  # MSE Loss
    loss.backward()  # Function approximation
    self.Q_eval.optimizer.step()
    self.learn_step_counter += 1
