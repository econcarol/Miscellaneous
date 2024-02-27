# RL: an Intro
# Ch 2: exercise 2.5

import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    def __init__(self, arms, alpha, epsilon):
        self.arms = arms         # # of handles
        self.alpha = alpha       # step size
        self.epsilon = epsilon   # fraction of time spent exploring
        self.N = np.zeros(arms)  # # of times each handle is pulled
        self.Q = np.zeros(arms)  # action-value fn for each handle
        self.R = np.ones(arms)   # equal true reward for all handles = 1
        
    def pull(self): 
        # epsilon-greedy policy
        rand = np.random.random()
        if rand < self.epsilon: # explore
            A = np.random.choice(self.arms)
        else: # exploit
            A = np.argmax(self.Q)
        # get reward
        reward = self.R[A] + np.random.randn() 
        # R takes independent random walks
        for i in range(self.arms):
            self.R[i] += (0.01 * np.random.randn())
        return A, reward 

    def updateQ1(self, A): # use sample average
        self.N[A] += 1
        self.Q[A] += (self.R[A] - self.Q[A]) / self.N[A]
    
    def updateQ2(self, A): # use constant step size
        self.Q[A] += self.alpha * (self.R[A] - self.Q[A])

    def optimalA(self):
        return np.argmax(self.R)

if __name__ == '__main__':
    arms    = 10    # 10 handles
    alpha   = 0.1   # step size
    epsilon = 0.1   # exploration factor
    
    runs  = 2000    # play 2000 games at each epsilon
    steps = 10000   # each game lasts 10000 steps
    totRewards = {}
    totActions = {}  
    
    # method 1: sample average
    totRewards[0] = np.zeros(steps)
    totActions[0] = np.zeros(steps)
    
    for i in range(runs):
        bandit = Bandit(arms, alpha, epsilon) # initiate game
        
        for j in range(steps): # play
           action, reward = bandit.pull()
           bandit.updateQ1(action)
           # record results
           totRewards[0][j] += reward
           if action == bandit.optimalA():
               totActions[0][j] += 1
    # compute average over all runs for each step       
    totRewards[0] = totRewards[0]/runs
    totActions[0] = totActions[0]/runs

    # method 2: constant step size
    totRewards[1] = np.zeros(steps)
    totActions[1] = np.zeros(steps)
    
    for i in range(runs):
        bandit = Bandit(arms, alpha, epsilon) # initiate game
        
        for j in range(steps): # play
           action, reward = bandit.pull()
           bandit.updateQ2(action)
           # record results
           totRewards[1][j] += reward
           if action == bandit.optimalA():
               totActions[1][j] += 1
    # compute average over all runs for each step       
    totRewards[1] = totRewards[1]/runs
    totActions[1] = totActions[1]/runs

    # plot results
    plt.subplot(211)    
    plt.plot(totRewards[0], 'r--', totRewards[1], 'b--')
    plt.legend(['sample avg', 'constant step size'])
    plt.xlabel('steps')
    plt.ylabel('average rewards')
    
    plt.subplot(212)    
    plt.plot(totActions[0], 'r--', totActions[1], 'b--')
    plt.legend(['sample avg', 'constant step size'])
    plt.xlabel('steps')
    plt.ylabel('fraction optimal action')
    plt.show()             
