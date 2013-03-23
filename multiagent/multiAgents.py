# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

bigNumber = 99999999

def distanceToNearestFood(pos, allFood):
  minDistance = bigNumber

  for foodPos in getAllFoodPositions(allFood):
    dist = manhattanDistance(pos, foodPos)
    if dist < minDistance: 
      minDistance = dist

  if minDistance == bigNumber:
    # no food left, this is a good thing so return a small value
    return 0.00001

  return minDistance

def averageDistanceToFood(pos, allFood):
  cumDist = 0
  count = 0

  for foodPos in getAllFoodPositions(allFood):
    cumDist += manhattanDistance(pos, foodPos)
    count += 1

  if count == 0:
    # no food left, this is a good thing so return a small value
    return 0.00001

  return cumDist/count

def getAllFoodPositions(food):
  x = 0
  positions = []

  for fx in food:
    y = 0
    for fy in fx:
      if fy:
        positions.append((x,y))
      y += 1
    x += 1

  return positions

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print "sgs = ", successorGameState
        # print "new pos =", newPos
        # print "new food = ", newFood
        # print "new ghost states = ", newGhostStates
        # print "new scared times = ", newScaredTimes

        distToNearestGhost = 4

        for ghostState in newGhostStates:
          if ghostState.scaredTimer < 2:
            dist = manhattanDistance(newPos, ghostState.configuration.getPosition())
            if dist < distToNearestGhost:
              distToNearestGhost = dist

        dtnf = distanceToNearestFood(newPos, newFood)

        "*** YOUR CODE HERE ***"
        return (2 ** distToNearestGhost)/(dtnf) + successorGameState.getScore() * 1.5

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def value(self, state, depth, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        depth -= 1

      if state.isWin() or state.isLose() or (depth == 0):
        return self.evaluationFunction(state)  

      nextAgentIndex = (agentIndex + 1) % (state.getNumAgents())
      if nextAgentIndex == 0:
        return self.max_value(state, depth, nextAgentIndex)
      else:
        return self.min_value(state, depth, nextAgentIndex)

    def max_value(self, state, depth, agentIndex):
      v = -bigNumber
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        v = max(v, self.value(successor, depth, agentIndex))
      return v

    def min_value(self, state, depth, agentIndex):
      v = bigNumber
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        v = min(v, self.value(successor, depth, agentIndex))
      return v

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        bestSoFar = -bigNumber
        bestActionIndex = 0
        i = 0

        for action in legalActions:
          actionValue  = self.value(gameState.generateSuccessor(0, action), self.depth, 0)
          if actionValue > bestSoFar:
            bestSoFar = actionValue
            bestActionIndex = i
          i += 1

        return legalActions[bestActionIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, state, depth, agentIndex, A, B):

      ## print "value called: depth= ", depth, " agent= ", agentIndex, " alpha= ", A, " beta= ", B

      if agentIndex == state.getNumAgents() - 1: 
        depth -= 1
      if state.isWin() or state.isLose() or (depth == 0):
        return self.evaluationFunction(state)  

      nextAgentIndex = (agentIndex + 1) % (state.getNumAgents())
      if nextAgentIndex == 0:
        return self.max_value(state, depth, nextAgentIndex, A, B)
      else:
        return self.min_value(state, depth, nextAgentIndex, A, B)

    def max_value(self, state, depth, agentIndex, A, B):
      v = -bigNumber
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        v = max(v, self.value(successor, depth, agentIndex, A, B))
        if v > B: return v
        A = max(A, v)
      return v

    def min_value(self, state, depth, agentIndex, A, B):
      v = bigNumber
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        v = min(v, self.value(successor, depth, agentIndex, A, B))
        if v < A: return v
        B = min(B, v)
      return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        bestSoFar = -bigNumber
        bestActionIndex = 0
        i = 0
        A = -bigNumber
        B = bigNumber

        for action in legalActions:
          v  = self.value(gameState.generateSuccessor(0, action), self.depth, 0, A, B)
          if v > bestSoFar:
            bestSoFar = v
            bestActionIndex = i
          if v > B: break
          A = max(A, v)
          i += 1

        return legalActions[bestActionIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, state, depth, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        depth -= 1

      if state.isWin() or state.isLose() or (depth == 0):
        return self.evaluationFunction(state)  

      nextAgentIndex = (agentIndex + 1) % (state.getNumAgents())
      if nextAgentIndex == 0:
        return self.max_value(state, depth, nextAgentIndex)
      else:
        return self.min_value(state, depth, nextAgentIndex)

    def max_value(self, state, depth, agentIndex):
      v = -bigNumber
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        v = max(v, self.value(successor, depth, agentIndex))
      return v

    def min_value(self, state, depth, agentIndex):
      total = 0.0
      count = 0.0
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        total += self.value(successor, depth, agentIndex)
        count += 1.0
      return total/count


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        bestSoFar = -bigNumber
        bestActionIndex = 0
        i = 0

        for action in legalActions:
          actionValue  = self.value(gameState.generateSuccessor(0, action), self.depth, 0)
          if actionValue > bestSoFar:
            bestSoFar = actionValue
            bestActionIndex = i
          i += 1

        return legalActions[bestActionIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Priority 1, keep away from ghosts so use exponential value of distance to nearest unascared ghost
      The reciprocal of distance to food also ensures he goes towards food.
      Also worth encouraging a good score too!
    """
    "*** YOUR CODE HERE ***"
    
    # keep away from ghosts!
    pacmanPos = currentGameState.getPacmanPosition()
    dist = 4
    for ghostState in currentGameState.getGhostStates():
          if ghostState.scaredTimer < 3:
            dist = min(dist, manhattanDistance(pacmanPos, ghostState.configuration.getPosition()))
    
    dtnf = distanceToNearestFood(pacmanPos, currentGameState.getFood())

    return 1.5 * (2 ** dist)/dtnf + currentGameState.getScore() * 2.2

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

