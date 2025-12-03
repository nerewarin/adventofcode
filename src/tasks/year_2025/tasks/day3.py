"""
--- Day 3: Lobby ---
https://adventofcode.com/2025/day/3
"""

from functools import lru_cache

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco
from src.utils.test_and_run import run, test

_logger = get_logger()


class Lobby:
    def __init__(self, data, task_num):
        self.data = data
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp), task_num)

    @staticmethod
    def _parse_input(inp: list[str]) -> list[list[int]]:
        return [[int(symbol) for symbol in line] for line in inp if line]

    @lru_cache
    def _is_invalid(self, id_str):
        id_len = len(id_str)

        for part_size in range(1, id_len):
            parts_amount, reminder = divmod(id_len, part_size)
            if reminder:
                continue

            parts = [id_str[part_num * part_size : (part_num + 1) * part_size] for part_num in range(parts_amount)]
            if len(set(parts)) == 1:
                return True
        return False

    @timeit_deco
    def solve(self) -> int:
        joltages = []
        for bank in self.data:
            if self.task_num == 1:
                # 2 pointers
                i = 0
                j = len(bank) - 1
                max_left, max_right = 0, 0
                max_i = None
                while i < j:
                    left = bank[i]
                    if left > max_left:
                        max_i = i
                    max_left = max(left, max_left)
                    if left == 9:
                        # no more i shifts
                        break
                    else:
                        i += 1
                        continue

                assert max_i is not None
                max_right = max(bank[max_i + 1 :])
                joltages.append(max_left * 10 + max_right)
            else:
                # raise ValueError(f"Invalid task number: {self.task_num}")
                raise NotImplementedError(f"Invalid task number: {self.task_num}")

        return sum(joltages)


def task(inp: list[str], task_num: int = 1) -> int:
    return Lobby.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 357)
    run(task1)
