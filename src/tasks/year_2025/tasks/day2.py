"""
--- Day 2: Gift Shop ---
https://adventofcode.com/2025/day/2
"""

from typing import cast

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


class GiftShop:
    def __init__(self, data, task_num):
        self.data = data
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> list[tuple[int, int]]:
        return cast(
            list[tuple[int, int]],
            [tuple(map(int, pair.split("-"))) for line in inp for pair in line.split(",") if pair],
        )

    def solve(self) -> int:
        invalid_ids = []
        for left, right in self.data:
            for num in range(left, right + 1):
                id_str = str(num)
                middle = len(id_str) // 2
                left = id_str[:middle]
                right = id_str[middle:]
                if left == right:
                    invalid_ids.append(num)

        return sum(invalid_ids)


def task(inp: list[str], task_num: int = 1) -> int:
    return GiftShop.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 1227775554)
    run(task1)
