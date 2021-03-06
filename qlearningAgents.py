# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # returns 0 if the state has never been before (dictionary), otherwise returns Q node value
        return self.values[state, action]

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # check for legal actions
        legalActions = self.getLegalActions(state)

        # if no legal actions available, then return 0
        if len(legalActions) == 0:
            return 0

        # from legal actions add the Q values into the list and calculate the maximum
        else:
            maximum = max(self.getQValue(state, action) for action in legalActions)

        # return max of q values
        return maximum

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # check for legal moves
        legalActions = self.getLegalActions(state)

        # if no legal moves possible, return None
        if len(legalActions) == 0:
            return None

        # Otherwise, find the best Action based on Q values
        else:

            # list to store q values
            values = util.Counter()

            # loop through list of legal (possible) actions and fill list
            for action in legalActions:
                values[action] = self.getQValue(state, action)

            # return largest q value from list
            return values.argMax()

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # if no legal moves, then return none since it is terminal state
        if len(legalActions) == 0:
            return None

        # with probability epsilon, we should take a random legal action
        else:

            # do a random action based on legal actions
            if util.flipCoin(self.epsilon):
                return random.choice(legalActions)

            # perform an optimal action based on current state
            else:
                return self.computeActionFromQValues(state)

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        """Formula for updating the Qvalue - alpha as learning rate(1 - alpha) * oldValue + alpha
         * (reward + discounted future reward) """

        # old q value
        oldValue = self.getQValue(state, action)

        # maximum future reward for next state
        discountedFutureReward = self.discount * self.computeValueFromQValues(nextState)

        # calculating the next q value
        total = (1 - self.alpha) * oldValue + self.alpha * (reward + discountedFutureReward)

        # update q value
        self.values[state, action] = total

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # declare value for returning
        qValue = 0

        # weight
        weight = self.weights

        # featureVector
        featureVector = self.featExtractor.getFeatures(state, action)

        # loop through featureVector and calculate sum of weight scores
        for feature in featureVector:
            qValue = qValue + (weight[feature] * featureVector[feature])

        # return sum of weight scores
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # weight
        weight = self.weights

        # featureVector
        featureVector = self.featExtractor.getFeatures(state, action)

        # difference is based on [reward + discounted future value] - Qvalue
        difference = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state, action)

        # updates weight by adding old weight to alpha * difference * feature vector of state and action
        for feature in featureVector:
            weight[feature] = weight[feature] + self.alpha * difference * featureVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
