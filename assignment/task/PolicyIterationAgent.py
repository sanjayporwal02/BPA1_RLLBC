import numpy as np
from agent import Agent
import operator

# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        self.V = {s: 0 for s in states}
        
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}
        
        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations): # for every itereation
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    if mdp.isTerminal(s) == True:
                    # if a is None:  
                        newV[s] = 0.0                        
                    else:
                        # update value estimate
                        reward = self.mdp.getReward(s,a,None)
                        successors = mdp.getTransitionStatesAndProbs(s,a)
                        newV[s] = 0.0 
                        for next_state, prob in successors: #looping over all the possible transitions from state s
                            newV[s] += prob*(reward + discount*self.V[next_state]) #summing state value over all possible transition
                    self.V[s] = newV[s]
                        # ******************
            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    v_pi = {a: 0 for a in actions}
                    for a in actions: # iterate through every possible action : where is the best value, i.e., a = argmax_a q(s,a)
                        reward = self.mdp.getReward(s,a,None)
                        successors = mdp.getTransitionStatesAndProbs(s,a)
                        for next_state, prob in successors: #looping over all the possible transitions from state s
                            v_pi[a] += prob*(reward + discount*self.V[next_state])
                    
                    self.pi[s] = max(v_pi, key=v_pi.get)
                   
                    if old_action != self.pi[s]:
                        policy_stable = False
                   
                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 
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

        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
