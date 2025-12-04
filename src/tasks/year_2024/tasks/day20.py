"""
--- Day 20: Race Condition ---
https://adventofcode.com/2024/day/20
"""

import logging

from src.utils.logger import get_logger, get_message_only_logger
from src.utils.maze import parse_maze
from src.utils.pathfinding import astar, manhattan_heuristic
from src.utils.position import Position2D, set_value_by_position
from src.utils.position_search_problem import OrthogonalPositionState, PositionSearchProblem
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


class RaceCondition:
    def __init__(self, maze: list[list[str]], start: Position2D, goal: Position2D, task_num: int, threshold: int):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.task_num = task_num
        self.threshold = threshold

        self._width = len(self.maze[0])
        self._height = len(self.maze)
        self._heuristic = manhattan_heuristic

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.maze}. task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num, threshold):
        return cls(*parse_maze(inp), task_num, threshold)

    @staticmethod
    def _get(pos: Position2D, grid: list[list[int]]) -> int:
        x, y = pos
        return grid[y][x]

    def _show_grid(self, grid: list[list[str]] | None = None, level=logging.DEBUG) -> None:
        if grid is None:
            grid = self.maze
        for i, line in enumerate(grid):
            line_str = "".join(line)
            _grid_logger.log(level, line_str)

    def _get_problem(self, grid) -> PositionSearchProblem:
        state = OrthogonalPositionState(grid, self.start, 0)
        return PositionSearchProblem(state=state, goal=self.goal, inp=grid)

    def _copy_grid(self) -> list[list[str]]:
        return [lst.copy() for lst in self.maze]

    def solve(self) -> int | None:
        _logger.debug("Initial grid:")
        self._show_grid()

        problem = self._get_problem(self.maze)
        initial_res = astar(problem, self._heuristic)
        if not initial_res:
            _logger.error("Path finding failed")
            return None

        _initial_final_state, _initial_actions, _ = initial_res
        initial_path_len = len(_initial_actions)
        grid_copy = self._copy_grid()
        for pos in _initial_final_state.path[1:-1]:  # leave start and end as is
            set_value_by_position(pos, PATH_MEMBER, grid_copy)

        _logger.debug("Final grid:")
        self._show_grid(grid_copy)

        return initial_path_len


def task(inp: list[str], task_num: int | None = 1, threshold: int | None = 100) -> int:
    return RaceCondition.from_multiline_input(inp, task_num, threshold).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 1, threshold=64)
    run(task1)
