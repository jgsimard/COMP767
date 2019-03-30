#Importer les librairies
import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 

#Commencer le modèle.Je fait peur quand je commence les choses et 
#J'essayer de construire un class dans python
class DeepQNetwork(nn.Module):
	def __init__(self, alpha):
		super(DeepQNetwork,self).__init__()

		#self.fc1 = nn.Linear(4096,1024,4)
		self.fc2 = nn.Linear(4096,1024,4)
		self.fc3 = nn.Linear(1024,9)

		self.optimizer = optim.RMSprop(self.parameters(), lr = alpha)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:loda' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, observation):
		observation = T.Tensor(observation).to(self.device)
		observation = observation.view(-1,1,240,240)
		observation = F.relu(self.conv1(observation))
		observation = F.relu(self.conv2(observation))
		observation = F.relu(self.conv3(observation))
		observation = observation.view(-1,128*19*8)
		#observation = F.relu(self.fc1(observation))
		observation = F.relu(self.fc2(observation))

		actions =  self.fc2(observation)

		return actions

class Agent(object):
	def __init__(self, gamma, epsilon, alpha,maxMemorySize, epsEnd = 0.05,
		replace = 10000, actionSpace = [0,1,2,3,4,5,6,7,8]):
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

	def storeTransition(self, state, action, reward, state_):
		if self.memCntr < self.memSize:
			self.action_history.append([state, action, reward, state_])
		else:
			self.action_history[self.memCntr%self.memSize] = [state, action, reward, state_]
		self.memCntr += 1

	def chooseAction(self, observation):
		rand = np.random.random()
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
			self.Q_next.load_state_dict(self.Q)

		if self.memCntr+batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCntr)))
		else:
			memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
		miniBatch =  self.action_history[memStart:memStart + batch_size]
		memory = np.array(miniBatch)

		# convert to list because memory is an array of numpy objects
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:,3][:])).to(self.Q_eval.device)       
        
        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device) 
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)        
        Qtarget = Qpred        
        Qtarget[:,maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        #Qpred.requires_grad_()        
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
