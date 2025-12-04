"""
--- Day 19: Linen Layout ---
https://adventofcode.com/2024/day/19
"""

from collections import defaultdict
from functools import cached_property

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco
from src.utils.test_and_run import run, test

_logger = get_logger()

WHITE = "w"
BLUE = "u"
BLACK = "b"
RED = "r"
GREEN = "g"


class LinenLayout:
    def __init__(self, available_towel_patterns, design_orders, task_num):
        self.available_towel_patterns = sorted(available_towel_patterns)
        self.design_orders = design_orders

        if task_num == 1:
            pass
        elif task_num == 2:
            pass
        else:
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"(patterns={len(self.available_towel_patterns)}, design_orders={len(self.design_orders)})"
        )

    @cached_property
    def patterns_by_len(self) -> dict[int, list[str]]:
        patterns_by_len = defaultdict(list)
        for pattern in self.available_towel_patterns:
            patterns_by_len[len(pattern)].append(pattern)
        return dict(patterns_by_len)

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(*cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> tuple[list[str], list[str]]:
        available_towel_patterns = inp[0].split(", ")
        design_orders = [line for line in inp[2:] if line]
        return available_towel_patterns, design_orders

    # @lru_cache
    def _is_possible(self, design_order: str) -> bool:
        for length, patterns in self.patterns_by_len.items():
            start = design_order[:length]
            for pattern in patterns:
                # TODO use bisect or dict by prefixes or smth like that
                # since we have patterns sorted, stop search when we reach next letter
                if ord(pattern[0]) > ord(start[0]):
                    break

                end = design_order[length:]
                if start == pattern:
                    if not end:
                        return True
                    is_possible = self._is_possible(end)
                    if is_possible:
                        return True

        return False

    @timeit_deco
    def solve(self) -> int:
        res = [0] * len(self.design_orders)
        for i, order in enumerate(self.design_orders):
            if self.task_num == 1:
                if self._is_possible(order):
                    # just for debug purposes keep track of which design is possible
                    res[i] = 1
            else:
                res[i] = self._count_possibilities(order)

        return sum(res)


def task(inp: list[str], task_num: int = 1) -> int:
    return LinenLayout.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 6)
    run(task1)
