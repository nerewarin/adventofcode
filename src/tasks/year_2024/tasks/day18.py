"""
--- Day 18: RAM Run ---
https://adventofcode.com/2024/day/18
"""

import logging
from typing import cast

from src.utils.logger import get_logger, get_message_only_logger
from src.utils.pathfinding import astar, manhattan_heuristic
from src.utils.position import Position2D
from src.utils.position_search_problem import OrthogonalPositionState, PositionSearchProblem
from src.utils.test_and_run import run, test

_logger = get_logger()
_grid_logger = get_message_only_logger()

WALL = "#"
SPACE = "."
PATH_MEMBER = "O"
LAST_BLOCK_SYMBOL = "X"


def pretty_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")


class RAMRun:
    def __init__(self, data, task_num, goal, steps_to_simulate):
        self.data = data
        self.task_num = task_num
        self.start = Position2D(0, 0)
        self.goal = goal
        self.steps_to_simulate = steps_to_simulate

        self._width, self._height = self.goal + Position2D(1, 1)
        self._heuristic = manhattan_heuristic

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data}. task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num, goal, steps_to_simulate):
        return cls(cls._parse_input(inp), task_num, goal, steps_to_simulate)

    @staticmethod
    def _parse_input(inp: list[str]) -> list[tuple[int, int]]:
        return cast(
            list[tuple[int, int]],
            [tuple(map(int, line.split(","))) for line in inp],
        )

    @staticmethod
    def _get(pos: Position2D, grid: list[list[int]]) -> int:
        x, y = pos
        return grid[y][x]

    @staticmethod
    def _show_grid(grid: list[list[int]], level=logging.DEBUG) -> None:
        symbols = {
            0: SPACE,
            1: WALL,
            2: PATH_MEMBER,
            3: LAST_BLOCK_SYMBOL,
        }
        for i, line in enumerate(grid):
            line_str = "".join([symbols.get(value, value) for value in line])

            _grid_logger.log(level, line_str)

    def _get_problem(self, grid, wall) -> PositionSearchProblem:
        state = OrthogonalPositionState(grid, self.start, 0, wall_symbol=wall)
        return PositionSearchProblem(state=state, goal=self.goal, inp=grid)

    def solve(self) -> int | None:
        simulation_list = self.data[: self.steps_to_simulate]
        last_block = simulation_list[-1]
        _logger.debug(f"last block: {last_block}")

        next_10_blocks = self.data[self.steps_to_simulate :][:10]

        space = 0
        wall = 1
        path_member = 2
        # last_block_symbol = 3

        grid = [[space for _ in range(self._height)] for _ in range(self._width)]
        for x, y in simulation_list:
            grid[y][x] = wall
            # if (x, y) == last_block:
            #     grid[y][x] = last_block_symbol

        _logger.debug("Initial grid:")
        self._show_grid(grid)

        problem = self._get_problem(grid, wall)
        res = astar(problem, self._heuristic)
        if not res:
            _logger.debug(f"Final grid for steps_to_simulate={self.steps_to_simulate} (None):")

            self._show_grid(grid, level=logging.WARNING)
            return None
        state, actions, score = res

        for x, y in state.path:
            grid[y][x] = path_member
        for i, (x, y) in enumerate(next_10_blocks):
            grid[y][x] = str(i)

        _logger.debug("Final grid:")
        self._show_grid(grid)

        return len(actions)


def task(inp: list[str], task_num: int = 1, goal=Position2D(70, 70), steps_to_simulate=1024) -> int:
    return RAMRun.from_multiline_input(inp, task_num, goal=goal, steps_to_simulate=steps_to_simulate).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    failures = 0
    # for steps_to_simulate in range(2959, 3450):
    for steps_to_simulate in (2961,):
        _logger.info(f"{steps_to_simulate=}")
        if _ := run(task1, steps_to_simulate=steps_to_simulate):
            continue
        else:
            failures += 1
            _logger.info(f"{failures=}")
            if failures == 10:
                return steps_to_simulate

    return None


if __name__ == "__main__":
    test(task1, 22, goal=Position2D(6, 6), steps_to_simulate=12)
    run(task1)  # 436

    # 2960 is wrong
    # 2961 is wrong
    # 2962   is wrong
    # 2964 wrong
    # 2965 wrong
    # 2966 wrong
    # 2967 wrong
    # 2968 # wrong?

    # run(task2)
    # run(task1, steps_to_simulate=2959)  # 436
