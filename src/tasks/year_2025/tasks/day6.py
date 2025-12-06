"""
--- Day 6: Trash Compactor ---
https://adventofcode.com/2025/day/6
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import Literal, Self

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
    operands: Iterable[int]
    operation: Operation

    @classmethod
    def from_strings(cls, *operands: str, operation: Literal["+", "*"]) -> Self:
        return cls(map(int, operands), Operation(operation))

    def solve(self) -> int:
        return self.operation.apply(self.operands)


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
        if task_num == 1:
            return (
                Problem.from_strings(*operands, operation=op)
                for *operands, op in zip(*(_rexp.split(line.strip()) for line in inp))
            )
        elif task_num == 2:
            # lines = len(inp)
            max_width = max(len(line) for line in inp)
            problems = []
            operation = None
            operands = []
            for col_idx in range(max_width):
                column_symbols = ""
                for i, line in enumerate(inp):
                    if col_idx >= len(line):
                        continue

                    symbol = line[col_idx]
                    if symbol == " ":
                        continue

                    if i == len(inp) - 1:
                        assert symbol == "+" or symbol == "*", f"Invalid operator {symbol=!r}"
                        if operation:
                            assert operands
                            # add problem
                            problem = Problem(operands, operation=operation)
                            problems.append(problem)
                            # reset accumulators
                            operands = []
                        operation = Operation(symbol)
                    else:
                        column_symbols += symbol

                if not column_symbols:
                    # blank column
                    continue
                operands.append(int(column_symbols))

            assert operation
            assert operands
            last_problem = Problem(operands, operation=operation)
            problems.append(last_problem)
            return problems
        else:
            raise ValueError(f"Invalid task number: {task_num}")

    def solve(self) -> int:
        return sum(problem.solve() for problem in self.problems)


def task(inp: list[str], task_num: int = 1) -> int:
    return TrashCompactor.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 4277556)
    run(task1)

    test(task2, 3263827)
    run(task2)
