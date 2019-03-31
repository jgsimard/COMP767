from prac_model import DeepQNetwork, Agent
import numpy as np 


if __name__ == '__main__':
	#Environment call command

	#defining the action
    brain = Agent(gamma=0.95, epsilon=1.0, 
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)   

   	scores = []
    epsHistory = []
    numGames = 50
    batch_size=32

    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)        
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        lastAction = 0   
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2)) #Redefine this
            if done:
                reward = -100 # Will redefine
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward, 
                                  np.mean(observation_[15:200,30:125], axis=2)) #Redefine this
            observation = observation_            
            brain.learn(batch_size)
            lastAction = action
            #env.render(
        scores.append(score)
        print('score:',score)