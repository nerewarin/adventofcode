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
from tqdm import tqdm

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
        if self.task_num == 2:
            colors = colors[::-1]
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

        out = f"day9_points_plot_{len(self.points)}_points_part{self.task_num}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.show()

    def solve(self) -> int:
        max_square = 0
        rectangle_corners = None

        min_values = [min(point[axis] for point in self.points) for axis in range(2)]
        max_values = [max(point[axis] for point in self.points) for axis in range(2)]

        if self.task_num == 1:
            for a, b in itertools.combinations(self.points, 2):
                w, h = a - b
                square = (abs(w) + 1) * (abs(h) + 1)
                if square > max_square:
                    max_square = square
                    rectangle_corners = (a, b)
        else:
            reference_points_data = self._get_reference_points(max_values, min_values)
            for ref_point, axis, _ in reference_points_data:
                if axis != 0:
                    raise NotImplementedError("vertical reference_points_lines")

            ref_points_amount = len(reference_points_data)
            for i, (ref_point, ref_axis, limits) in enumerate(reference_points_data):
                if ref_axis != 0:
                    raise NotImplementedError("reference_points considered ")

                for point in tqdm(
                    self.points,
                    total=len(self.points),
                    desc=f"ref_point {i}/{ref_points_amount}: considering {len(self.points)} pairs",
                ):
                    # if other_axis_value_above_avg:
                    #     raise NotImplementedError(f"{other_axis_value_above_avg=}")
                    for axis in range(2):
                        # look for other side
                        limits_for_axis = limits[axis]
                        callable_, value = limits_for_axis
                        if not callable_(point[axis], value):
                            break
                        # if ref_points_amount == 1:
                        #     ...
                        # elif ref_points_amount == 2:
                        #     if axis == another_axis:
                        #         # look to the end of your side instead
                        #         limits_for_axis *= -1
                        # else:
                        #     raise NotImplementedError(f"not considered {ref_points_amount=} case")
                        # if limits_for_axis < 0:
                        #     if point[axis] >= ref_point[axis]:
                        #         break
                        # else:
                        #     if point[axis] < ref_point[axis]:
                        #         break
                    else:
                        w, h = ref_point - point
                        square = (abs(w) + 1) * (abs(h) + 1)
                        if square > max_square:
                            max_square = square
                            rectangle_corners = (ref_point, point)

            # for pair in tqdm(product(reference_points_data, self.points), total=len(self.points) * len(reference_points_data)):
            #     (reference_point, axis, size_by_axis), point = pair
            #     if reference_point == point:
            #         continue

        assert rectangle_corners is not None
        self._show_plot([rectangle_corners])
        return max_square

    def _get_reference_points(
        self, max_values: list[int], min_values: list[int]
    ) -> list[tuple[Position2D, int, tuple]]:
        lines = []
        wight_and_height = [(max_values[i] - min_values[i]) for i in range(2)]
        middles = [(max_values[i] - min_values[i]) / 2 + min_values[i] for i in range(2)]
        for i, point in enumerate(self.points):
            next_point = self.points[(i + 1) % len(self.points)]
            diff = next_point - point

            for axis in range(2):
                size_by_axis = abs(diff[axis])
                limit = wight_and_height[axis]
                if limit > size_by_axis > 3 / 4 * limit:
                    offsets_from_center = (
                        abs(point[axis] - middles[axis]),
                        abs(next_point[axis] - middles[axis]),
                    )
                    max_offset_idx = offsets_from_center.index(min(offsets_from_center))
                    ref_point = [point, next_point][max_offset_idx]

                    axis_value = ref_point[axis]
                    if axis_value > middles[axis]:
                        axis_limit = (lambda a, b: a < b), ref_point[axis]
                    else:
                        axis_limit = (lambda a, b: a > b), ref_point[axis]

                    another_axis = (axis + 1) % 2
                    another_axis_value = ref_point[another_axis]
                    if another_axis_value > middles[another_axis]:
                        other_axis_limit = (lambda a, b: a < b), ref_point[another_axis]
                    else:
                        other_axis_limit = (lambda a, b: a > b), ref_point[another_axis]

                    limits = axis_limit, other_axis_limit  # e.g. (-1, -1) left top

                    lines.append((ref_point, axis, limits))
                    break

                # TODO find another axis limit
        return lines


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
    # test(task1, 50)
    # assert run(task1) == 4737096935  # 2700162690, 3005828594 and 4679487087: too low

    test(task2, 24)
    run(task2)
#
