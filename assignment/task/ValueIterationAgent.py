from agent import Agent
import numpy as np

# TASK 2

class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {s: 0 for s in states}
        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}
        
        # ************

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if mdp.isTerminal(s) == True:
                    newV[s] = 0.0
                   
                else: 
                    v_pi = {a: 0 for a in actions}
                    count=0
                    for a in actions: # iterate through every possible action : where is the best value, i.e., a = argmax_a q(s,a)
                    # update value estimate
                        reward = self.mdp.getReward(s,a,None)
                        successors = mdp.getTransitionStatesAndProbs(s,a)
                        for next_state, prob in successors: #looping over all the possible transitions from state s
                            v_pi[a] += prob*(reward + discount*self.V[next_state])
                            count +=1
                    newV[s] = max(v_pi.values())
                   
                # Update value function with new estimate
                self.V[s] = newV[s]
                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        successors = self.mdp.getTransitionStatesAndProbs(state, action) #get all successor states and probabilities
        reward = self.mdp.getReward(state,action,None) 
        q_value = 0.0
        for next_state, prob in successors: #looping over all the possible transitions from state s
            q_value += prob*(reward + self.discount*self.V[next_state])
        return q_value
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:
            
        # **********
        # TODO 2.4
            q_value = {a: 0 for a in actions}
            for a in actions:
                q_value[a] += self.getQValue(state,a)
            self.pi[state] = max(q_value, key=q_value.get)
        return self.pi[state]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
