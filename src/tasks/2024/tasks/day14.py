import re
from dataclasses import dataclass
from functools import cached_property
from math import prod

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run, test

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
    x: int = 103
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


def task1(inp: list[str], space_cls: type[AbstractSpace] = RuntimeSpace, seconds=100) -> int:
    space = space_cls()
    robots = _parse_input(inp, 1)

    res_by_quarter = [0, 0, 0, 0]
    for robot in robots:
        final_position = simulate(robot, space, seconds)
        if final_position.x == space.mid_x or final_position.y == space.mid_y:
            continue

        quarter = space.get_quarter(final_position)
        res_by_quarter[quarter] += 1

    return prod(res_by_quarter)


if __name__ == "__main__":
    test(task1, 12, space_cls=TestSpace)
    assert run(task1) > 221096928
