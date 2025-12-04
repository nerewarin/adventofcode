from abc import ABC, abstractmethod
from collections.abc import Generator, Sized
from typing import Any

from src.utils.directions import ORTHOGONAL_DIRECTIONS, OrthogonalDirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.position import Position2D, get_value_by_position


@abstractmethod
class BaseState(ABC):
    """Abstract base class for any storage backend."""

    @property
    @abstractmethod
    def pos(self) -> Any: ...

    @abstractmethod
    def get_successors(self) -> Generator[tuple["BaseState", Any, Any]]: ...

    def __hash__(self):
        # hash only by position
        return hash(self.pos)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseState):
            return NotImplemented
        # equality only by position
        return self.pos == other.pos

    def get_cost_of_actions(self, actions: Sized) -> Any:
        """
        Returns the cost of a particular sequence of actions. Very basic logic. Override it if needed.
        """
        return len(actions)


class OrthogonalPositionState(BaseState):
    def __init__(self, inp: list[list[str]], pos, step=None, path=None, actions=None, cost=0, wall_symbol="#"):
        self.inp = inp
        self.x, self.y = pos
        self.symbol = self.inp[self.y][self.x]

        self.step = step or 0
        self.path = path or [pos]
        self.actions = actions or []
        self.cost = cost
        self.wall_symbol = wall_symbol

    def __str__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"(pos={self.pos}, step={self.step}, symbol={self.symbol}, last_action={self.get_last_action()},"
            f"cost={self.cost})"
        )

    @property
    def pos(self) -> Position2D:
        return Position2D(self.x, self.y)

    def _get(self, pos: Position2D) -> str:
        return get_value_by_position(pos, self.inp)

    def _is_wall(self, yx: Position2D) -> bool:
        return self._get(yx) == self.wall_symbol

    @property
    def _directions(self) -> Generator[OrthogonalDirectionEnum]:
        yield from ORTHOGONAL_DIRECTIONS.items()

    def get_successors(self) -> Generator[tuple[BaseState, OrthogonalDirectionEnum, int]]:
        prior_action = self.get_last_action()
        for yx, direction in self._directions:
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue
            if self._is_wall(pos):
                continue

            new_path = self.path + [pos]
            actions = [direction]
            new_actions = self.actions + actions
            cost = self.get_cost_of_actions(actions)

            state = self.__class__(
                self.inp, pos, self.step + 1, new_path, new_actions, cost=self.cost + cost, wall_symbol=self.wall_symbol
            )
            action = direction

            yield state, action, cost

    def get_last_action(self) -> OrthogonalDirectionEnum | None:
        if not self.actions:
            return None
        return self.actions[-1]


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

    @staticmethod
    def get_successors(state: BaseState) -> Generator[tuple[BaseState, Any, Any]]:
        """Must return new state, last action and its cost"""
        yield from state.get_successors()
