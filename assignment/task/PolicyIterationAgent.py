import numpy as np
from agent import Agent


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
        self.V = np.zeros_like(states)

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
                    Terminal_state = mdp.isTerminal(s)
                    next_state, prob = [], []
                    next_state, prob = self.mdp.getTransitionStatesAndProbs(s, a)
                    reward = self.mdp.getReward(s, a, next_state)
                    possible_transitions = len(next_state)
                    
                    if Terminal_state == True:
                        newV[s] = 0.0
                        return None
                    else:
                        # update value estimate
                        for t in range(possible_transitions): #looping over all the possible transitions from state s
                            newV[s] += self.pi[s,t]*prob[t]*(reward + discount*self.V[next_state[t]]) #summing state value over all possible transition
                self.V = newV
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
                    q_pi = -np.inf
                    for a in actions: # iterate through every possible action : where is the best value, i.e., a = argmax_a q(s,a)
                        v_pi = 0
                        for t in range(possible_transitions): #looping over all the possible transitions from state s
                            v_pi += prob[t]*(reward + discount*self.V[next_state[t]])
                            if q_pi<=v_pi:
                                q_pi = v_pi
                                best_action = a
                    if best_action != old_action:
                        policy_stable = False
                        # changing the policy to pi_prime to avoid infinite loop
                        self.pi[s] = best_action

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
        # TODO 1.3.
        next_state, prob = [], []
        next_state, prob = self.gridWorld.getTransitionStatesAndProbs(state, action) #get all successor states and probabilities
        reward = self.mdp.getReward(state,action,next_state) 
        possible_transitions = len(next_state)
        q_value = 0
        for t in range(possible_transitions): #looping over all the possible transitions from state s
            q_value += prob[t]*(reward + self.discount*self.V[next_state[t]])
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
