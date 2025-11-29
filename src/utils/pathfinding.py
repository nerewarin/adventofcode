"""
copy of my solutions from CS188.1x: Artificial Intelligence course

"""

import heapq
from collections.abc import Generator
from typing import Any

from src.utils.position_search_problem import BaseState, PositionSearchProblem


def manhattan_distance(point1, point2):
    """
    Calculate Manhattan distance between two points.

    Parameters:
    - point1: Tuple (x1, y1)
    - point2: Tuple (x2, y2)

    Returns:
    - Manhattan distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


###########
# Data structures useful for implementing SearchAgents
###########


class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
        Dequeue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    Note that this PriorityQueue does not allow you to change the priority
    of an item.  However, you may insert the same item multiple times with
    different priorities.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        entry = (priority, self.count, item)
        # entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        #  (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0


class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def manhattan_heuristic(state: BaseState, problem: PositionSearchProblem):
    """The Manhattan distance heuristic for a PositionSearchProblem"""
    return state.pos.manhattan_to(problem.goal)


def distance_heuristic(state: BaseState, problem: PositionSearchProblem):
    return state.pos.distance_to(problem.goal)


def generic_search(problem, fringe, add_to_fringe_fn) -> tuple[BaseState, list[Any], Any] | None:
    closed = set()
    start = (problem.get_start_state(), 0, [])  # (node, cost, path)
    add_to_fringe_fn(fringe, start, 0)

    while not fringe.isEmpty():
        (node, cost, path) = fringe.pop()

        if problem.is_goal_state(node):
            return node, path, cost

        # STATE MUST BE HASHABLE BY POSITION!
        if node not in closed:
            closed.add(node)

            for child_node, child_action, child_cost in problem.get_successors(node):
                new_cost = cost + child_cost
                new_path = path + [child_action]
                new_state = (child_node, new_cost, new_path)
                add_to_fringe_fn(fringe, new_state, new_cost)

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    fringe = Stack()

    def add_to_fringe_fn(fringe, state, cost):
        fringe.push(state)

    return generic_search(problem, fringe, add_to_fringe_fn)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = Queue()

    def add_to_fringe_fn(fringe, state, cost):
        fringe.push(state)

    return generic_search(problem, fringe, add_to_fringe_fn)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    fringe = PriorityQueue()

    def add_to_fringe_fn(fringe, state, cost):
        fringe.push(state, cost)

    return generic_search(problem, fringe, add_to_fringe_fn)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = PriorityQueue()

    def add_to_fringe_fn(fringe: PriorityQueue, state, cost):
        new_cost = cost + heuristic(state[0], problem)
        fringe.push(state, new_cost)

    return generic_search(problem, fringe, add_to_fringe_fn)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

if __name__ == "__main__":
    from src.tasks.year_2024.tasks.day16 import State
    from src.utils.directions import ADJACENT_DIRECTIONS, DiagonalDirectionEnum, go
    from src.utils.position import Position2D

    maze = [
        # v (start)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #                  v (goal: (y=7, x=6))
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    end = Position2D(7, 6).reversed()

    class TestOrthogonalState(State):
        def _is_wall(self, yx) -> bool:
            return self._get(yx) == 1

        def get_path_coordinates(self):
            path = [self.pos]
            for action in actions:
                prev = path[-1]
                new = go(action, prev)
                path.append(new)
            return path

        def get_cost_of_actions(self, actions: list[Any]) -> int:
            return 1

    start_state = TestOrthogonalState(maze, start)
    problem = PositionSearchProblem(start_state, end, maze)

    final_state, actions, cost = astar(problem, heuristic=manhattan_heuristic)
    path = start_state.get_path_coordinates()
    assert len(path) == 14
    assert path[0] == start
    assert path[-1] == end

    class TestDiagonalState(TestOrthogonalState):
        @property
        def _directions(self) -> Generator[DiagonalDirectionEnum]:
            yield from ADJACENT_DIRECTIONS.items()

    start_state = TestDiagonalState(maze, start)
    problem = PositionSearchProblem(start_state, end, maze)

    final_state, actions, cost = astar(problem, heuristic=manhattan_heuristic)
    path = start_state.get_path_coordinates()

    # if (4, 3) is (x, y) here - there is a wall in maze at this place so its definitely
    expected_path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 5), (7, 6)]
    # assert path == expected_path, path
    # so we need to reverse positions to test first
    expected_path_reversed = [Position2D(*pos).reversed() for pos in expected_path]
    assert path == expected_path_reversed
