"""
--- Day 7: Laboratories ---
https://adventofcode.com/2025/day/7
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import Literal

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()
_rexp = re.compile(r"\s+")


@dataclass
class Operation:
    symbol: Literal["+", "*"]

    def apply(self, operands: Iterable[int]) -> int:
        if self.symbol == "+":
            operation = sum
        elif self.symbol == "*":
            operation = prod
        else:
            raise NotImplementedError(f"Unknown operation {self.symbol!r}")
        return operation(operands)


@dataclass
class Problem:
    manifold: list[list[int]]

    def solve(self) -> int:
        beams = []
        res = 0
        for i, line in enumerate(self.manifold):
            if not beams:
                beams = line
                _logger.info(f"initial: {beams}")
                continue

            for j, value in enumerate(line):
                if value:
                    if beams[j]:
                        res += 1

                        width = len(beams)

                        for step in (j + 1, j - 1):
                            if not 0 <= step < width:
                                continue

                            beams[step] = 1
                        beams[j] = 0

            _logger.info(f"step {i}/{len(self.manifold)}: {line=}, {beams=}")

        return res


class TrashCompactor:
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
        yield Problem(manifold=[[(symbol in ("S", "^")) * 1 for symbol in line] for line in inp])

    def solve(self) -> int:
        return sum(problem.solve() for problem in self.problems)


def task(inp: list[str], task_num: int = 1) -> int:
    return TrashCompactor.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 21)
    run(task1)
