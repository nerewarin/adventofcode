"""
--- Day 2: Gift Shop ---
https://adventofcode.com/2025/day/2
"""

from typing import cast

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco
from src.utils.test_and_run import run

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

    @timeit_deco
    def solve(self) -> int:
        invalid_ids = []
        for left, right in self.data:
            for num in range(left, right + 1):
                id_str = str(num)

                if self.task_num == 1:
                    middle = len(id_str) // 2
                    left = id_str[:middle]
                    right = id_str[middle:]
                    if left == right:
                        invalid_ids.append(num)
                elif self.task_num == 2:
                    id_len = len(id_str)

                    for part_size in range(1, id_len):
                        parts_amount, reminder = divmod(id_len, part_size)
                        if reminder:
                            continue

                        parts = [
                            id_str[part_num * part_size : (part_num + 1) * part_size]
                            for part_num in range(parts_amount)
                        ]
                        if len(set(parts)) == 1:
                            invalid_ids.append(num)
                            break
                else:
                    raise ValueError(f"Invalid task number: {self.task_num}")

        return sum(invalid_ids)


def task(inp: list[str], task_num: int = 1) -> int:
    return GiftShop.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    # test(task1, 1227775554)
    # run(task1)

    # test(task2, 4174379265)
    run(task2)
