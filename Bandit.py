# RL: an Intro
# Ch 2: reproduce Fig 2.2 

import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    def __init__(self, arms, epsilon):
        self.arms = arms                # # of handles
        self.epsilon = epsilon          # fraction of time spent exploring
        self.N = np.zeros(arms)         # # of times each handle is pulled
        self.Q = np.zeros(arms)         # action-value fn for each handle
        self.R = np.random.randn(arms)  # true reward of each handle
        
    def pull(self): # epsilon-greedy policy
        rand = np.random.random()
        if rand < self.epsilon: # explore
            A = np.random.choice(self.arms)
        else: # exploit
            A = np.argmax(self.Q)
        return A, self.R[A] + np.random.randn()
    
    def updateQ(self, A): # use sample average
        self.N[A] += 1
        self.Q[A] += (self.R[A] - self.Q[A]) / self.N[A]
    
    def optimalA(self):
        return np.argmax(self.R)

if __name__ == '__main__':
    arms  = 10    # 10 handles
    runs  = 2000  # play 2000 games at each epsilon
    steps = 1000  # each game lasts 1000 steps
    totRewards = {}
    totActions = {}
    
    for k, epsilon in enumerate([0, 0.01, 0.1]):
        totRewards[k] = np.zeros(steps)
        totActions[k] = np.zeros(steps)
        
        for i in range(runs):
            bandit = Bandit(arms, epsilon) # initiate game
            
            for j in range(steps): # play
               action, reward = bandit.pull()
               bandit.updateQ(action)
               # record results
               totRewards[k][j] += reward
               if action == bandit.optimalA():
                   totActions[k][j] += 1
        # compute average over all runs for each step       
        totRewards[k] = totRewards[k]/runs
        totActions[k] = totActions[k]/runs
    
    # plot results
    plt.subplot(211)    
    plt.plot(totRewards[0], 'g--', totRewards[1], 'r--', totRewards[2], 'b--')
    plt.legend(['epsilon=0, greedy', 'epsilon=0.01', 'epsilon=0.1'])
    plt.xlabel('steps')
    plt.ylabel('average rewards')
    
    plt.subplot(212)    
    plt.plot(totActions[0], 'g--', totActions[1], 'r--', totActions[2], 'b--')
    plt.legend(['epsilon=0, greedy', 'epsilon=0.01', 'epsilon=0.1'])
    plt.xlabel('steps')
    plt.ylabel('fraction optimal action')
    plt.show()
                