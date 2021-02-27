# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # loop through each iteration then update values at the end of each loop (iteration)
        for iteration in range(self.iterations):

            # assign an empty set
            values = util.Counter()

            # loop through each state and its action
            for state in self.mdp.getStates():

                # get action for that state
                action = self.getAction(state)

                # fill variable values with a list of q values based on action (if action is not none)
                if action is not None:
                    values[state] = self.computeQValueFromValues(state, action)

            # update list of q values
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        """Formula: Q*(s) = sum [ probability * (reward + (discount value *  
                * [reward + (discount value * q value)) """

        # used for calculating the Q state of actions
        total = 0

        # loop through list of transitions and calculate the total Q values from performing actions
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):

            # probability value
            probability = transition[1]

            # reward value
            reward = self.mdp.getReward(state, action, transition)

            # current value
            value = self.getValue(transition[0])

            # calculating the sum of Q values by using the formula
            total = total + (probability * (reward + self.discount * value))

        # return the sum of Q values
        return total

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # return none if the state that agent is on is the terminal state
        if self.mdp.isTerminal(state):
            return None

        # if not, then get a list of Q values that the agent can take
        else:

            # store Q values
            value = util.Counter()

            # list of legal actions from state
            legalActions = self.mdp.getPossibleActions(state)

            # loop through possible actions and fill in list with Q values
            for action in legalActions:
                value[action] = self.getQValue(state, action)

            # return the largest Q value
            return value.argMax()

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # get list of states
        states = self.mdp.getStates()

        # all the states start with the value of zero
        maximum = 0

        # loop through each iteration
        for iteration in range(self.iterations):

            # update a single state using the number of iteration % total number of states
            state = states[iteration % len(states)]

            # check to see if state is not terminal
            if not self.mdp.isTerminal(state):

                # get possible actions from state
                actions = self.mdp.getPossibleActions(state)

                # get the maximum value
                maximum = max(self.getQValue(state, action) for action in actions)

                # update self.values[state] with the maximum value
                self.values[state] = maximum

        # return maximum value
        return maximum


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # initialize a priority queue
        priorityQueue = util.PriorityQueue()

        # predecessor implemented with a set to avoid duplications
        predecessor = {}

        # loop through states
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = abs(self.value[state] - max())
