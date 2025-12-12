"""
--- Day 9: Movie Theater ---
https://adventofcode.com/2025/day/*
"""

import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run, test

_logger = get_logger()


@dataclass
class Problem:
    points: list[Position2D]
    task_num: int

    def solve(self) -> int:
        max_square = 0
        for a, b in itertools.combinations(self.points, 2):
            w, h = a - b
            square = (abs(w) + 1) * (abs(h) + 1)
            max_square = max(max_square, square)

        return max_square


class MovieTheater:
    def __init__(self, problems: Iterable[Problem], task_num: int):
        self.problems = problems
        if _logger.level <= logging.DEBUG:
            # make sure parsing part2 returns same problems len as part1 with an eye
            problems_copy = list(problems)
            _logger.debug("len(problems) = %d", len(problems_copy))
            self.problems = iter(problems_copy)

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(problems={self.problems}, task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp, task_num), task_num)

    @staticmethod
    def _parse_input(inp: list[str], task_num: int) -> Iterable[Problem]:
        yield Problem(
            points=[Position2D.integer_point_from_tuple_of_strings(tuple(line.split(","))) for line in inp],
            task_num=task_num,
        )

    def solve(self) -> int:
        return sum(problem.solve() for problem in self.problems)


def task(inp: list[str], task_num: int = 1, **kw) -> int:
    return MovieTheater.from_multiline_input(inp, task_num).solve(**kw)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


if __name__ == "__main__":
    test(task1, 50)
    assert run(task1) > 4679487087  # 2700162690, 3005828594 and 4679487087: too low
