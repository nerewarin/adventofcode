"""
--- Day 1: Secret Entrance ---
https://adventofcode.com/2025/day/1
"""

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


class SecretEntrance:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.data})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp))

    @staticmethod
    def _parse_input(inp: list[str]) -> list[tuple[str, int]]:
        data = []
        for line in inp:
            if line:
                direction = line[0]
                if direction not in ("L", "R"):
                    continue
                value = line[1:]
                data.append((direction, int(value)))
        return data

    def solve(self) -> int:
        dial = 50
        zero_times = 0
        for direction, value in self.data:
            if direction == "L":
                dial -= value
            elif direction == "R":
                dial += value
            dial %= 100
            if dial == 0:
                zero_times += 1
        return zero_times


def task(inp: list[str], task_num: int = 1) -> int:
    return SecretEntrance.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 3)
    run(task1)
