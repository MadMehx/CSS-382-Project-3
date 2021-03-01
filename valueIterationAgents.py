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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # loop through each iteration then update values at the end of each loop (iteration)
        for iteration in range(self.iterations):

            # assign an empty dictionary
            values = util.Counter()

            # loop through each state from list of states
            for state in self.mdp.getStates():

                # returns the policy at the state; calls computeActionFromValues for current state
                action = self.getAction(state)

                # fill dictionary with q values if action is not None
                if action is not None:
                    values[state] = self.computeQValueFromValues(state, action)

            # update q values
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

        """Formula: Q*(s) = sum [ probability * (reward + (discount value * q value)] """

        # used for calculating the sum of Q values
        total = 0

        # loop through list of transitions and calculate the sum of Q values
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):

            # reward value
            reward = self.mdp.getReward(state, action, nextState)

            # future state
            value = self.getValue(nextState)

            # calculating the sum of Q values by using the formula:
            total = total + (probability * (reward + (self.discount * value)))

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

        # return none if the state is a terminal state
        if self.mdp.isTerminal(state):
            return None

        # if not, then get a list of Q values that the agent can take
        else:

            # store Q values
            value = util.Counter()

            # loop through possible actions and fill in list with Q values
            for action in self.mdp.getPossibleActions(state):
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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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

        # loop through each iteration
        for iteration in range(self.iterations):

            # update a single state using the number of iteration % total number of states
            state = self.mdp.getStates()[iteration % len(self.mdp.getStates())]

            # check to see if state is not terminal
            if not self.mdp.isTerminal(state):

                # get the maximum q value from possible actions
                maximum = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))

                # update self.values[state] with the maximum value
                self.values[state] = maximum

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # predecessor implemented with a set to avoid duplications
        predecessors = {}

        # compute predecessor of all states
        for state in self.mdp.getStates():

            # check to see if see current state is a terminal state
            if not self.mdp.isTerminal(state):

                # loop through possible actions from that state
                for action in self.mdp.getPossibleActions(state):

                    # loop through transition states and their probabilities
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):

                        # add states to the list of predecessors[nextState]
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}

        # initialize a priority queue
        priorityQueue = util.PriorityQueue()

        # loop through states
        for state in self.mdp.getStates():

            # check to see if current state is not a terminal state
            if not self.mdp.isTerminal(state):
                # find the maximum q values
                maximum = max(
                    self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))

                # find the absolute value of the difference between the current value of state in self.values
                # and the highest Q-value across all possible actions from state
                diff = abs(self.values[state] - maximum)

                # push state into the priorityQueue with a priority on diff
                priorityQueue.push(state, -diff)

        # loop through iterations
        for iteration in range(self.iterations):

            # check to see if the priorityQueue is empty, if so, then terminate loop
            if priorityQueue.isEmpty():
                break

            # otherwise, pop off a state from the priority queue and update state value given it's not a terminal state
            else:

                # pop state off from priority queue
                state = priorityQueue.pop()

                # check to see if state is not a terminal state and proceed with the self.value update
                if not self.mdp.isTerminal(state):

                    # calculate the maximum Q value from all states
                    maximum = max(self.computeQValueFromValues(state, action)
                                  for action in self.mdp.getPossibleActions(state))

                    # update Q values for state
                    self.values[state] = maximum

                # loop through each predecessor
                for predecessor in predecessors[state]:

                    # calculate the maximum q value from all predecessors
                    maxValue = max(self.computeQValueFromValues(predecessor, action)
                                   for action in self.mdp.getPossibleActions(predecessor))

                    # find the absolute value of the difference between the current value of predecessors in self.values
                    # and the highest Q-value across all possible actions from predecessors
                    diff = abs(self.values[predecessor] - maxValue)

                    # update priority queue with a priority on diff
                    if diff > self.theta:
                        priorityQueue.update(predecessor, -diff)
