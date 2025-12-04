"""
--- Day 4: Printing Department ---
https://adventofcode.com/2025/day/4
"""

import logging

from src.utils.directions import ADJACENT_DIRECTIONS, go, out_of_borders
from src.utils.logger import get_logger, get_message_only_logger
from src.utils.position import Position2D, get_value_by_position
from src.utils.profiler import timeit_deco
from src.utils.test_and_run import run, test

_logger = get_logger()


class PrintingDepartment:
    def __init__(self, inp: list[str], diagram: list[list[int]], task_num: int):
        self._grid = [[symbol for symbol in line] for line in inp]
        self.diagram = diagram
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(inp, cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> list[list[int]]:
        symbol_to_int = {".": 0, "@": 1}
        return [[symbol_to_int[symbol] for symbol in line] for line in inp if line]

    def _show_grid(self, level=logging.INFO) -> None:
        for line in self._grid:
            get_message_only_logger().log(level, "".join(line))

    @timeit_deco
    def solve(self) -> int:
        res = 0
        for row_idx, row in enumerate(self.diagram):
            for col_idx, value in enumerate(row):
                position = Position2D(col_idx, row_idx)
                value = get_value_by_position(position, self.diagram)
                if not value:
                    continue
                rolls_nearby = 0
                for coords, direction in ADJACENT_DIRECTIONS.items():
                    neighbor = go(direction, position)
                    if out_of_borders(*neighbor, self.diagram):
                        continue
                    rolls_nearby += get_value_by_position(neighbor, self.diagram)
                    if rolls_nearby >= 4:
                        break
                if rolls_nearby < 4:
                    self._grid[row_idx][col_idx] = "x"
                    res += 1
        self._show_grid()
        return res


def task(inp: list[str], task_num: int = 1) -> int:
    return PrintingDepartment.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 13)
    run(task1)
