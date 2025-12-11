"""
--- Day 7: Playground  ---
https://adventofcode.com/2025/day/8
"""

import logging
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


class Box(NamedTuple):
    x: int
    y: int
    z: int

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other) -> bool:
        for a in range(3):
            if self[a] < other[a]:
                return True
            elif self[a] > other[a]:
                return False
        return False

    def __eq__(self, other) -> bool:
        for a in range(3):
            if self[a] != other[a]:
                return False
        return True

    @lru_cache
    def _distance_to(self, other: "Box") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def distance_to(self, other: "Box") -> float:
        r"""
        d(p, q) = {\sqrt
        {(p_{1}-q_{1}) ^ {2} + (p_{2}-q_{2}) ^ {2} + (p_{3}-q_{3}) ^ {2}}}.}
        """
        if self == other:
            return 0
        if self > other:
            return other.distance_to(self)
        return self._distance_to(other)


def connected_components(edges):
    graph = defaultdict(set)

    # Build adjacency list (undirected)
    for u, v, *_ in edges:  # ignore weights
        graph[u].add(v)
        graph[v].add(u)

    visited = set()
    connected = []

    for node in graph:
        if node in visited:
            continue

        # BFS / DFS to determine component size
        q = deque([node])
        visited.add(node)
        nbrs = {node}

        while q:
            cur = q.popleft()
            for nbr in graph[cur]:
                if nbr not in visited:
                    visited.add(nbr)
                    nbrs.add(nbr)
                    q.append(nbr)

        connected.append(nbrs)

    return connected


@dataclass
class Problem:
    boxes: list[list[int]]
    task_num: int

    def solve(self, connections: int | None = None) -> int:
        if connections is None:
            connections = 1000

        boxes = [Box(*box) for box in self.boxes]

        mat = np.array([[box.distance_to(another) for box in boxes] for another in boxes])
        n = mat.shape[0]

        # take only upper triangle, k=1 excludes the diagonal
        # take only upper triangle, k=1 excludes the diagonal
        i_idx, j_idx = np.triu_indices(n, k=1)

        # distances for those pairs
        vals = mat[i_idx, j_idx]

        # sort by distance
        order = np.argsort(vals)

        # get sorted unique edges (i < j)
        edges = [(int(i_idx[k]), int(j_idx[k]), float(vals[k])) for k in order[:connections]]

        graphs = connected_components(edges)
        graphs.sort(key=lambda x: -len(x))

        if self.task_num == 1:
            top3 = [len(g) for g in graphs[:3]]
            return math.prod(top3)

        for k in order[connections:]:
            edge = (int(i_idx[k]), int(j_idx[k]))

            a, b = edge
            a_x = boxes[a].x
            b_x = boxes[b].x

            graph_a, graph_b = None, None
            for i, graph in enumerate(graphs):
                if a in graph:
                    graph_a = i
                if b in graph:
                    graph_b = i
                if graph_a and graph_b:
                    break

            if graph_a is not None and graph_a == graph_b:
                continue

            if graph_a is None and graph_b is None:
                graphs.append({a, b})
            elif graph_a is None:
                graphs[graph_b].add(a)
            elif graph_b is None:
                graphs[graph_a].add(b)
            else:
                graphs[graph_a] = graphs[graph_a].union(graphs[graph_b])

                if len(graphs) == 1:
                    return a_x * b_x

                graphs.pop(graph_b)

            graphs.sort(key=lambda x: -len(x))

            _logger.debug(f"connected={sum(len(g) for g in graphs)}")

            if len(graphs[0]) == len(boxes):
                return a_x * b_x

        return 0


class Playground:
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
        yield Problem(boxes=[[int(symbol) for symbol in line.split(",")] for line in inp], task_num=task_num)

    def solve(self, connections=None) -> int:
        return sum(problem.solve(connections) for problem in self.problems)


def task(inp: list[str], task_num: int = 1, **kw) -> int:
    return Playground.from_multiline_input(inp, task_num).solve(**kw)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


if __name__ == "__main__":
    test(task1, 40, connections=10)
    run(task1)

    test(task2, 25272, connections=10)
    run(task2)
