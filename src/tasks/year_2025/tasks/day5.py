"""
--- Day 4: Cafeteria  ---
https://adventofcode.com/2025/day/5
"""

from functools import cached_property
from typing import NamedTuple, Self

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco
from src.utils.test_and_run import run, test

_logger = get_logger()


class IngredientRange(NamedTuple):
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start + 1

    def contains(self, value: int):
        return self.start <= value <= self.end

    def intersects(self, other: Self) -> bool:
        #     a---b
        #       c-----d
        if self.start <= other.start <= self.end:
            return True
        #     a---b
        #    c-----d
        if other.start <= self.start <= other.end:
            return True
        return False

    def merge(self, other: Self) -> Self | None:
        if self.intersects(other):
            return self.__class__(min(self.start, other.start), max(self.end, other.end))
        return None

    def __lt__(self, other: Self) -> bool:
        return self.start < other.start


IngredientRangeList = list[IngredientRange]
IngredientIdList = list[int]


class Cafeteria:
    def __init__(
        self, fresh_ingredient_ranges: IngredientRangeList, available_ingredient_ids: IngredientIdList, task_num: int
    ):
        self._fresh_ingredient_ranges = fresh_ingredient_ranges
        self._available_ingredient_ids = available_ingredient_ids

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    @cached_property
    def fresh_ingredient_ranges(self) -> IngredientRangeList:
        # return merged ranges
        fresh_ingredient_ranges = sorted(self._fresh_ingredient_ranges)

        for i in range(len(fresh_ingredient_ranges) - 1):
            left = fresh_ingredient_ranges[i]
            right = fresh_ingredient_ranges[i + 1]

            merged = left.merge(right)
            if merged:
                fresh_ingredient_ranges[i + 1] = merged
                fresh_ingredient_ranges[i] = None

        return [r for r in fresh_ingredient_ranges if r is not None]

    @cached_property
    def available_ingredient_ids(self) -> IngredientIdList:
        return sorted(self._available_ingredient_ids)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(*cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> tuple[IngredientRangeList, IngredientIdList]:
        fresh_ingredient_ranges = []
        i = 0
        for i, line in enumerate(inp):
            if not line:
                break
            start, end = line.split("-")
            fresh_ingredient_ranges.append(IngredientRange(int(start), int(end)))

        available_ingredient_ids = [int(value) for value in inp[i + 1 :]]
        return fresh_ingredient_ranges, available_ingredient_ids

    @timeit_deco
    def solve(self) -> int:
        if self.task_num == 2:
            return sum(r.length for r in self.fresh_ingredient_ranges)

        res = 0
        fresh_ingredient_ranges = self.fresh_ingredient_ranges

        last_considered_range_idx = 0
        for ingredient_id in self.available_ingredient_ids:
            for i, fresh_ingredient_range in enumerate(fresh_ingredient_ranges[last_considered_range_idx:]):
                if ingredient_id < fresh_ingredient_range.start:
                    last_considered_range_idx += i
                    break

                if fresh_ingredient_range.contains(ingredient_id):
                    res += 1
                    last_considered_range_idx += i
                    break

        return res


def task(inp: list[str], task_num: int = 1) -> int:
    return Cafeteria.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 3)
    run(task1)

    test(task2, 14)
    run(task2)
