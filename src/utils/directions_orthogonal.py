"""
Instruments to work with 2D points.
Take care: every position is represented as (Y, X) instead of (X, Y) for easier access to 2d-lists

from math lessons:

y
^
|
|
|
.----> x

from computer history (first display)
0...-> v
---->
...... >    ..so

.----> x
|
|
|
v
y

"""

from collections.abc import Iterable
from enum import StrEnum

from src.utils.position import Position2D


class DirectionEnum(StrEnum):
    up = "up"
    right = "right"
    down = "down"
    left = "left"


SYMBOL_UP = "^"
SYMBOL_RIGHT = ">"
SYMBOL_DOWN = "v"
SYMBOL_LEFT = "<"

DIRECTION_SYMBOLS = {
    SYMBOL_DOWN: DirectionEnum.down,
    SYMBOL_UP: DirectionEnum.up,
    SYMBOL_RIGHT: DirectionEnum.right,
    SYMBOL_LEFT: DirectionEnum.left,
}

DIRECTIONS = {
    # y, x
    (-1, 0): DirectionEnum.up,
    (0, 1): DirectionEnum.right,
    (1, 0): DirectionEnum.down,
    (0, -1): DirectionEnum.left,
}

DIRECTIONS_BY_ENUM = {v: k for k, v in DIRECTIONS.items()}


def get_abs(y, x):
    return abs(y), abs(x)


def get_2d_diff(pos1: tuple[int, int], pos2: tuple[int, int], absolute=False):
    y1, x1 = pos1
    y2, x2 = pos2
    res = y2 - y1, x2 - x1
    if absolute:
        res = get_abs(*res)
    return res


def go(
    directions: Iterable[DirectionEnum] | Iterable[tuple[int, int]] | DirectionEnum | tuple[int, int],
    pos: tuple[int, int] | Position2D | None = None,
    y: int | None = None,
    x: int | None = None,
    return_reversed: bool | None = None,
) -> tuple[int, int] | Position2D:
    _directions = []
    if isinstance(directions, DirectionEnum):
        _directions = [directions]
    elif isinstance(directions, tuple) and len(directions) == 2:
        _directions.append(DIRECTIONS[directions])
    else:
        for d in directions:
            if isinstance(d, DirectionEnum):
                _directions.append(d)
            elif isinstance(d, tuple):
                _directions.extend(d)
                break
            else:
                raise ValueError(f"unknown branch for {d=}")

    is_given_separately = x is not None and y is not None
    is_given_as_tuple = pos is not None
    if is_given_as_tuple:
        if isinstance(pos, Position2D):
            x, y = pos
            if return_reversed is None:
                return_reversed = False
        else:
            y, x = pos
            if return_reversed is None:
                return_reversed = True
    if return_reversed is None:
        raise ValueError("could not determine return_reversed value!")

    if not is_given_separately and not is_given_as_tuple:
        raise ValueError("either is_given_separately or is_given_as_tuple must be passed!")

    if is_given_separately and is_given_as_tuple:
        raise ValueError("either is_given_separately or is_given_as_tuple mus be passed, not both!")

    for d in _directions:
        dy, dx = DIRECTIONS_BY_ENUM[d]
        y = y + dy
        x = x + dx

    if return_reversed:
        return y, x
    return Position2D(x, y)


def out_of_borders(x, y, grid, is_reversed=True):
    if is_reversed:
        x, y = y, x

    max_y = len(grid)
    if y < 0 or y >= max_y:
        return True

    max_x = len(grid[y])
    if x < 0 or x >= max_x:
        return True

    return False


def is_horizontal(direction: DirectionEnum) -> bool:
    y, x = DIRECTIONS_BY_ENUM[direction]
    if not y and not x:
        raise ValueError("y or x must be passed!")
    if y and x:
        raise ValueError("either y or x must be zero!")
    return bool(x)


def reverse_coordinates(coordinates: tuple[int, int]) -> tuple[int, int]:
    x, y = coordinates
    return x * -1, y * -1


def is_a_way_back(direction1: DirectionEnum, direction2: DirectionEnum) -> bool:
    coordinates1 = DIRECTIONS_BY_ENUM[direction1]
    coordinates2 = DIRECTIONS_BY_ENUM[direction2]

    reversed_coordinates2 = reverse_coordinates(coordinates2)
    return coordinates1 == reversed_coordinates2
