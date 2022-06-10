import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        #self.V = {}
        return self.V[state]

        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if (state,action) in self.Q:
            return self.Q[(state,action)]
        else:
             return self.qInitValue

        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        actions = self.actionFunction(state)
        if len(actions) == 0:
            return "exit"
        else:  
            if state in self.Q:
                for a in actions:
                    Qval = {a:0}
                    Qval += self.getQValue(state,a)
                    return max(Qval, key=Qval.get)
            else:
                return self.getRandomAction(state)                   
            

        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions) 
            
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if state in self.Q:

            if np.random.rand(0, 1, dtype=np.float32) > self.epsilon:
                return self.getPolicy(state)
        
            else:
            # Select the action with max value
                return self.getRandomAction(state)
        else:
            return self.getRandomAction(state)

        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        
        if (state) in self.Q:
            for i in range (100):
                self.Q[state, action] = self.Q[state, action] + self.learningRate * (reward + self.discount * np.max(self.Q[nextState, :]) — self.Q[state, action])


        else:
             self.Q[state, action] = self.qInitValue

        # *********
