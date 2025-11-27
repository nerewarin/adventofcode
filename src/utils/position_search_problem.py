from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any

from src.tasks.year_2023.tasks.day10 import State
from src.utils.position import Position2D


@abstractmethod
class BaseState(ABC):
    """Abstract base class for any storage backend."""

    @property
    @abstractmethod
    def pos(self) -> Position2D: ...

    @abstractmethod
    def get_successors(self) -> Iterable["BaseState"]: ...

    @abstractmethod
    def copy(self) -> "BaseState": ...

    @abstractmethod
    def get_last_action(self) -> Any: ...


class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, state: BaseState, goal, cost_fn=lambda x: 1, inp=None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startState = state
        self.goal = goal
        self.cost_fn = cost_fn
        self.inp = inp

    def get_start_state(self):
        return self.startState

    def is_goal_state(self, state):
        return state.pos == self.goal

    def get_successors(self, state: State) -> Generator[tuple[BaseState, Any, Any]]:
        """Must return new state, last action and its cost"""
        yield from state.get_successors()
