"""
--- Day 6: Trash Compactor ---
https://adventofcode.com/2025/day/6
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import Literal, Self

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco
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

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(problems={self.problems}, task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> Iterable[Problem]:
        return (
            Problem.from_strings(*operands, operation=op)
            for *operands, op in zip(*(_rexp.split(line.strip()) for line in inp))
        )

    @timeit_deco
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
