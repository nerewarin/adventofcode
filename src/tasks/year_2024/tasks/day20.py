"""
--- Day 20: Race Condition ---
https://adventofcode.com/2024/day/20
"""

import logging
from collections.abc import Generator
from typing import Any

from src.utils.directions import OrthogonalDirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.logger import get_logger, get_message_only_logger
from src.utils.maze import draw_maze, parse_maze
from src.utils.pathfinding import PriorityQueue, astar, manhattan_heuristic
from src.utils.position import Position2D, set_value_by_position
from src.utils.position_search_problem import BaseState, OrthogonalPositionState, PositionSearchProblem
from src.utils.test_and_run import run, test

_logger = get_logger()
_grid_logger = get_message_only_logger()

WALL = "#"
SPACE = "."
START = "S"
END = "E"
PATH_MEMBER = "O"
CHEAT_START = "1"
CHEAT_END = "2"


class RaceConditionState(OrthogonalPositionState):
    def __init__(self, *args, cheat_made_at: Position2D | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cheat_made_at = cheat_made_at
        self.is_final = False

    def get_successors(self) -> Generator[tuple["RaceConditionState", OrthogonalDirectionEnum, int]]:
        """
        Extends base functionality allowing one cheat
        """
        prior_action = self.get_last_action()
        for yx, direction in self._directions:
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue

            cheat_made_at = self.cheat_made_at
            if self._is_wall(pos):
                if cheat_made_at is None:
                    cheat_made_at = pos
                else:
                    continue

            new_path = self.path + [pos]
            actions = [direction]
            new_actions = self.actions + actions
            cost = self.get_cost_of_actions(actions)

            state = self.__class__(
                self.inp,
                pos,
                self.step + 1,
                new_path,
                new_actions,
                cost=self.cost + cost,
                wall_symbol=self.wall_symbol,
                cheat_made_at=cheat_made_at,
            )
            action = direction

            yield state, action, cost


class LimitedPositionSearchProblem(PositionSearchProblem):
    """
    Extends position search with cost limit
    """

    def __init__(self, *args, path_costs_cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cost_threshold = cost_threshold
        self.path_costs_cache = path_costs_cache or {}
        self.considered_successors: set[Position2D] = set()

    def is_goal_state(self, state):
        return state.pos == self.goal

    def get_successors(self, state: RaceConditionState) -> Generator[tuple[BaseState, Any, Any]]:
        """Must return new state, last action and its cost"""
        for state, action, cost in state.get_successors():
            pos = state.pos
            if pos not in self.considered_successors:
                self.considered_successors.add(pos)
                # _logger.debug(f"Considering succ at {pos}")
                # state.draw()

                if pos in self.path_costs_cache:
                    state.is_final = True
                    state.cost += self.path_costs_cache[pos]
                yield state, action, cost


def limited_search(problem, fringe, add_to_fringe_fn) -> tuple[BaseState, list[Any], Any] | None:
    """generic search rework"""
    closed = set()
    start = (problem.get_start_state(), 0, [])  # (state, cost, actions)
    add_to_fringe_fn(fringe, start, 0)

    while not fringe.isEmpty():
        (state, cost, actions) = fringe.pop()

        if problem.is_goal_state(state):
            return state, actions, cost

        # STATE MUST BE HASHABLE BY POSITION!
        if state not in closed:
            closed.add(state)

            for child_node, child_action, child_cost in problem.get_successors(state):
                new_cost = cost + child_cost
                new_actions = actions + [child_action]
                new_state = (child_node, new_cost, new_actions)

                if child_node.is_final:
                    # TODO not sure..
                    # and now update cache

                    cheat_made_at = child_node.cheat_made_at  # noqa

                    return new_state, new_actions, new_cost

                add_to_fringe_fn(fringe, new_state, new_cost)

    return None


def count_cheap_paths_with_astar(problem, cost_threshold, heuristic):
    """Like astar but
    1. does not add state to fringe if cost reaches threshold
    2. consider
    """
    fringe = PriorityQueue()
    res = 0

    def add_to_fringe_fn(fringe: PriorityQueue, state, cost):
        new_cost = cost + heuristic(state[0], problem)
        if new_cost > cost_threshold:
            return None
        fringe.push(state, new_cost)

    while new_solution := limited_search(problem, fringe, add_to_fringe_fn):
        (state, actions, cost) = new_solution
        assert len(actions) == cost  # sanity check for this problem only!
        res += 1
        _logger.debug(f"New solution #{res} found with {cost=}")

    return res


class RaceCondition:
    def __init__(self, maze: list[list[str]], start: Position2D, goal: Position2D, task_num: int, target_savings: int):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.task_num = task_num
        self.target_savings = target_savings

        self._width = len(self.maze[0])
        self._height = len(self.maze)
        self._heuristic = manhattan_heuristic

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.maze}. task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num, target_savings):
        return cls(*parse_maze(inp), task_num, target_savings)

    @staticmethod
    def _get(pos: Position2D, grid: list[list[int]]) -> int:
        x, y = pos
        return grid[y][x]

    def _show_grid(self, grid: list[list[str]] | None = None, level=logging.DEBUG) -> None:
        if grid is None:
            grid = self.maze
        draw_maze(grid)

    def _get_problem(
        self, state_cls, grid, path_costs_cache: dict[Position2D, int] | None = None
    ) -> PositionSearchProblem:
        state = state_cls(grid, self.start, 0)
        return LimitedPositionSearchProblem(state=state, goal=self.goal, inp=grid, path_costs_cache=path_costs_cache)

    def _copy_grid(self) -> list[list[str]]:
        return [lst.copy() for lst in self.maze]

    def _initial_run(self):
        problem = self._get_problem(OrthogonalPositionState, self.maze)
        initial_res = astar(problem, self._heuristic)
        if not initial_res:
            _logger.error("Path finding failed")
            return None

        _initial_final_state, *_ = initial_res
        return _initial_final_state

    def solve(self) -> int | None:
        _logger.debug("Initial grid:")
        self._show_grid()

        _initial_final_state = self._initial_run()

        min_actions_with_no_cheat = len(_initial_final_state.actions)
        grid_copy = self._copy_grid()
        for pos in _initial_final_state.path[1:-1]:  # leave start and end as is
            set_value_by_position(pos, PATH_MEMBER, grid_copy)

        _logger.debug("Final grid:")
        # self._show_grid(grid_copy)
        _initial_final_state.draw()

        if self.task_num != 1:
            raise NotImplementedError(f"task_num = {self.task_num} is not implemented")

        # from initial run:
        # 1. use length of the best path with no cheats to compute cost_threshold
        target_savings = self.target_savings
        if min_actions_with_no_cheat <= target_savings:
            raise ValueError(f"{min_actions_with_no_cheat=} <= {target_savings=}!")
        cost_threshold = min_actions_with_no_cheat - target_savings
        # 2. cache path length from every initial path to goal not to compute it again
        # we can then update this cache fot new points found AFTER the cheat
        path_costs_cache = {}
        for i, pos in _initial_final_state.path:  # leave start and end as is
            path_costs_cache[pos] = min_actions_with_no_cheat - i

        problem = self._get_problem(RaceConditionState, self.maze, path_costs_cache)
        res = count_cheap_paths_with_astar(problem, cost_threshold, self._heuristic)
        return res


def task(inp: list[str], task_num: int | None = 1, target_savings: int | None = 100) -> int:
    return RaceCondition.from_multiline_input(inp, task_num, target_savings).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


class Tests:
    @classmethod
    def test1(cls):
        final_state_for_initial_run = RaceCondition.from_multiline_input(
            [
                [x for x in line.strip()]
                for line in """
                    ###############
                    #...#...#.....#
                    #.#.#.#.#.###.#
                    #S#...#.#.#...#
                    #######.#.#.###
                    #######.#.#...#
                    #######.#.###.#
                    ###..E#...#...#
                    ###.#######.###
                    #...###...#...#
                    #.#####.#.###.#
                    #.#...#.#.#...#
                    #.#.#.#.#.#.###
                    #...#...#...###
                    ###############
                """.split("\n")
                if line
            ],
            1,
            1,
        )._initial_run()

        assert len(final_state_for_initial_run.actions) == 84


if __name__ == "__main__":
    Tests.test1()
    test(task1, 1, target_savings=64)
    run(task1)
