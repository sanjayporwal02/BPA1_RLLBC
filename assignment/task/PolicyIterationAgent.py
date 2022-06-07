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
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    actions = self.mdp.getPossibleActions(s)
                    # *****************
                    # TODO 1.1.b)
                    
                    if self.mdp.isTerminal(s):
                        newV[s] = 0
                        return None
                    #
                    else:
                        for action in actions:
                        nextStates = self.mdp.getTransitionStatesAndProbs(s, action)[0]
                        probs = self.mdp.getTransitionStatesAndProbs(s, action)[1]
                        for nextState in nextStates:
                            rewards = self.mdp.getReward(s, action, nextState)
                            for reward in rewards:
                                newV[s] = a * probs * (reward + discount*newV[nextState])

                # update value estimate
                self.V = newV 

                # ******************

            policy_stable = True
            val = {}
            initial_val = -np.inf
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    for action in actions:
                        val[s] = 0
                        nextStates = self.mdp.getTransitionStatesAndProbs(s, action)[0]
                        probs = self.mdp.getTransitionStatesAndProbs(s, action)[1]
                        for nextState in nextStates:
                            rewards = self.mdp.getReward(s, action, nextState)
                            for reward in rewards:
                                val[s] = probs * (reward + discount*newV[nextState])
                        if val[s] > initial_val:
                            initial_val = val[s]
                            best_action = action

                if best_action != old_action:
                    policy_stable = False

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
        actions = self.mdp.getPossibleActions(s)
        QValue = 0
        for action in actions:
                        nextStates = self.mdp.getTransitionStatesAndProbs(s, action)[0]
                        probs = self.mdp.getTransitionStatesAndProbs(s, action)[1]
                        for nextState in nextStates:
                            rewards = self.mdp.getReward(s, action, nextState)
                            for reward in rewards:
                                QValue = probs * (reward + self.discount*self.V[nextState])


        return QValue

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
