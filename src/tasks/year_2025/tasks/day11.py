"""
--- Day 11: Reactor ---
https://adventofcode.com/2025/day/11
"""

import collections
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


class Indicator(StrEnum):
    off = "."
    on = "#"

    @property
    def value(self):
        return int(self == Indicator.on)


@dataclass
class Device:
    name: str
    connections: list["Device"] = field(default_factory=list)
    path: list[str] = field(default_factory=list)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "Device | str"):
        return self.name == str(other)


@dataclass
class Problem:
    device_by_name: dict[str, Device]
    task_num: int
    _start = "you"
    _end = "out"

    def solve(self) -> int:
        if self.task_num == 1:
            return self._solve_part1()
        else:
            return self._solve_part2()

    def _solve_part1(self):
        fringe = collections.deque()

        start = self.device_by_name[self._start]

        fringe.append(start)

        paths = 0
        while fringe:
            device = fringe.popleft()

            if device == self._end:
                paths += 1
                continue

            for connected in self.device_by_name[device.name].connections:
                fringe.append(connected)

        return paths


class Factory:
    def __init__(self, problems: Iterable[Problem], task_num: int):
        self.problems = problems

        problems_copy = list(problems)
        self._problems_len = len(problems_copy)
        self.problems = iter(problems_copy)

        if _logger.level <= logging.DEBUG:
            # make sure parsing part2 returns same problems len as part1 with an eye
            _logger.debug("len(problems) = %d", self._problems_len)

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(problems={self.problems}, task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp, task_num), task_num)

    @classmethod
    def _parse_line(cls, line: str) -> tuple[str, list[str]]:
        device, connections = line.split(": ")
        return device, connections.split(" ")

    @staticmethod
    def to_dot(graph: dict[str, list[str]]) -> str:
        """Return a Graphviz DOT representation."""
        lines: list[str] = ["digraph G {"]
        lines.append("    rankdir=LR;")  # left-to-right layout; remove if not needed

        # optional styling for 'you' and 'out'
        lines.append('    "you" [shape=doublecircle, style=filled, fillcolor=lightblue];')
        lines.append('    "out" [shape=Msquare, style=filled, fillcolor=lightgray];')

        for src, targets in graph.items():
            for dst in targets:
                lines.append(f'    "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def _parse_input(cls, inp: list[str], task_num: int) -> Iterable[Problem]:
        graph = {}
        devices = {}
        for line in inp:
            device_name, connections_raw = cls._parse_line(line)
            devices[device_name] = Device(
                device_name,
                connections=[Device(x) for x in connections_raw],
            )

            graph[device_name] = connections_raw

        for src, targets in graph.items():
            for dst in targets:
                _logger.debug("%s -> %s", src, dst)
        _logger.debug("%s", cls.to_dot(graph))

        yield Problem(devices, task_num)

    def solve(self) -> int:
        res = 0
        for problem in tqdm(self.problems, desc=f"Solving task{self.task_num}", total=self._problems_len):
            res += problem.solve()
        return res


def task(inp: list[str], task_num: int = 1) -> int:
    return Factory.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 5)
    run(task1)
