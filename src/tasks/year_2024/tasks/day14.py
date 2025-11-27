import logging
import re
from dataclasses import dataclass
from functools import cached_property
from math import prod

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run

_logger = get_logger()

_coordinate_pattern = r"(-?\d+)"
_point_pattern = f"{_coordinate_pattern},{_coordinate_pattern}"

_input_rexp = re.compile(rf"p={_point_pattern} v={_point_pattern}")


class AbstractSpace:
    x: int
    y: int

    @cached_property
    def mid_x(self):
        return self.x // 2

    @cached_property
    def mid_y(self):
        return self.y // 2

    @cached_property
    def mid(self):
        return self.mid_x, self.mid_y

    def get_quarter(self, pos: Position2D) -> int:
        """
        0 1
        2 3
        """
        is_right = pos.x > self.mid_x
        is_down = pos.y > self.mid_y
        return is_right + is_down * 2


@dataclass
class TestSpace(AbstractSpace):
    x: int = 11
    y: int = 7


@dataclass
class RuntimeSpace(AbstractSpace):
    x: int = 101
    y: int = 103


@dataclass
class Robot:
    x: int
    y: int
    vx: int
    vy: int
    idx: int


def _parse_input(inp: list[str], task_num: int) -> list[Robot]:
    robots: list[Robot] = []

    for i, line in enumerate(inp):
        if not line:
            continue

        robots.append(Robot(*map(int, _input_rexp.match(line).groups()), idx=i))

    return robots


def simulate(robot: Robot, space: AbstractSpace, seconds: int) -> Position2D:
    x = (robot.x + robot.vx * seconds) % space.x
    y = (robot.y + robot.vy * seconds) % space.y
    return Position2D(x, y)


def _draw(positions: list[Position2D], space: AbstractSpace, turn: int):
    if _logger.level <= logging.DEBUG:
        grid = [["." for _ in range(space.x)] for _ in range(space.y)]
        for x, y in positions:
            if grid[y][x] == ".":
                grid[y][x] = "1"
            else:
                grid[y][x] = str(int(grid[y][x]) + 1)

        filled_middle_rows = 0
        for row in grid:
            if row[space.mid_x] != ".":
                filled_middle_rows += 1

        is_tree = filled_middle_rows > (space.y // 3)
        if is_tree:
            _logger.debug(f"{turn=}")
            for row in grid:
                _logger.debug("".join(row))

        return is_tree


def task(inp: list[str], space_cls: type[AbstractSpace] = RuntimeSpace, seconds: int | None = 100) -> int:
    space = space_cls()
    robots = _parse_input(inp, 1)

    if seconds:
        res_by_quarter = [0, 0, 0, 0]
        for robot in robots:
            final_position = simulate(robot, space, seconds)
            if final_position.x == space.mid_x or final_position.y == space.mid_y:
                continue

            quarter = space.get_quarter(final_position)
            res_by_quarter[quarter] += 1

        return prod(res_by_quarter)

    else:
        # task 2
        seconds = 0
        while True:
            positions = []
            seconds += 1
            for i, robot in enumerate(robots):
                position = simulate(robot, space, 1)
                positions.append(position)

                robots[i] = Robot(*position, robot.vx, robot.vy, idx=i)

            is_tree = _draw(positions, space, seconds)
            if is_tree is None:
                raise NotImplementedError

            # if is_tree:
            #     return seconds


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, seconds=None, **kw)


if __name__ == "__main__":
    # test(task1, 12, space_cls=TestSpace)
    # assert run(task1)  # 226201248

    # test(task2, 0, space_cls=TestSpace)
    run(task2)
