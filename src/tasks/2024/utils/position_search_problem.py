
class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, state, goal, costFn = lambda x: 1, inp=None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startState = state
        self.goal = goal
        self.costFn = costFn
        self.inp = inp

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state.pos == self.goal

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        #  position, action, dummy_cost
        successors = []
        for name, dist in state.get_successors().items():
            s = state.copy()
            s.move_to_valve(name)

            succ = s, name, dist
            successors.append(succ)
        return successors
