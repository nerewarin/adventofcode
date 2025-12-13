"""
--- Day 9: Movie Theater ---
https://adventofcode.com/2025/day/*
"""

import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run, test

_logger = get_logger()


@dataclass
class Problem:
    points: list[Position2D]
    task_num: int

    def _show_plot(self, rectangles: Iterable[tuple[Position2D, Position2D]]) -> None:
        pts = []
        for point in self.points:
            pts.append(point)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xs_closed = xs + [xs[0]]
        ys_closed = ys + [ys[0]]

        plt.figure(figsize=(8, 8))
        plt.plot(xs_closed, ys_closed, marker="o", markersize=2, linewidth=1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"Polyline through {len(pts)} points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # add rectangle

        colors = "red", "green"
        for i, rectangle_corners in enumerate(rectangles):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.fill(
                xs_closed,
                ys_closed,
                alpha=0.25,
                zorder=1,
            )
            ax.plot(
                xs_closed,
                ys_closed,
                marker="o",
                markersize=2,
                linewidth=1,
                zorder=3,
            )
            (x1, y1), (x2, y2) = rectangle_corners
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            rect = Rectangle(
                (x_min, y_min),
                width,
                height,
                fill=False,
                edgecolor=colors[i % 2],
                linewidth=2,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.grid(True)

        out = f"day9_points_plot_{len(self.points)}_points.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.show()

    def solve(self) -> int:
        max_square = 0
        rectangle_corners = None

        min_values = [min(point[axis] for point in self.points) for axis in range(2)]
        max_values = [max(point[axis] for point in self.points) for axis in range(2)]
        # middles = [(max_values[i] - min_values[i] / 2) + min_values[i] for i in range(2)]

        if self.task_num == 1:
            points = self.points
        else:
            points = set()
            wight_and_height_halves = [(max_values[i] - min_values[i]) / 2 for i in range(2)]
            for i, point in enumerate(self.points):
                next_point = self.points[(i + 1) % len(self.points)]
                diff = next_point - point

                for axis in range(2):
                    if abs(diff[axis]) > wight_and_height_halves[axis]:
                        points.add(point)
                        points.add(next_point)
                        break

        for a, b in itertools.combinations(points, 2):
            w, h = a - b

            square = (abs(w) + 1) * (abs(h) + 1)
            if square > max_square:
                max_square = square
                rectangle_corners = (a, b)

        assert rectangle_corners is not None
        self._show_plot([rectangle_corners])
        return max_square


class MovieTheater:
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
        yield Problem(
            points=[Position2D.integer_point_from_tuple_of_strings(tuple(line.split(","))) for line in inp],
            task_num=task_num,
        )

    def solve(self) -> int:
        return sum(problem.solve() for problem in self.problems)


def task(inp: list[str], task_num: int = 1, **kw) -> int:
    return MovieTheater.from_multiline_input(inp, task_num).solve(**kw)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


if __name__ == "__main__":
    test(task1, 50)
    assert run(task1) == 4737096935  # 2700162690, 3005828594 and 4679487087: too low

    test(task2, 24)
    run(task2)
