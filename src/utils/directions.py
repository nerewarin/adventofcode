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


class AbstractDirectionEnum:
    pass


class OrthogonalDirectionEnum(AbstractDirectionEnum, StrEnum):
    up = "up"
    right = "right"
    down = "down"
    left = "left"


SYMBOL_UP = "^"
SYMBOL_RIGHT = ">"
SYMBOL_DOWN = "v"
SYMBOL_LEFT = "<"

ORTHOGONAL_DIRECTION_SYMBOLS = {
    SYMBOL_DOWN: OrthogonalDirectionEnum.down,
    SYMBOL_UP: OrthogonalDirectionEnum.up,
    SYMBOL_RIGHT: OrthogonalDirectionEnum.right,
    SYMBOL_LEFT: OrthogonalDirectionEnum.left,
}
ORTHOGONAL_DIRECTION_SYMBOLS_BY_ENUM = {v: k for k, v in ORTHOGONAL_DIRECTION_SYMBOLS.items()}


class DiagonalDirectionEnum(AbstractDirectionEnum, StrEnum):
    up_left = "up_left"
    up_right = "up_right"
    down_left = "down_left"
    down_right = "down_right"


SYMBOL_UP_LEFT = "_/"
SYMBOL_UP_RIGHT = "-\\"
SYMBOL_DOWN_LEFT = "/-"
SYMBOL_DOWN_RIGHT = "/-"

DIAGONAL_DIRECTION_SYMBOLS = {
    SYMBOL_UP_LEFT: DiagonalDirectionEnum.up_left,
    SYMBOL_UP_RIGHT: DiagonalDirectionEnum.up_right,
    SYMBOL_DOWN_LEFT: DiagonalDirectionEnum.down_left,
    SYMBOL_DOWN_RIGHT: DiagonalDirectionEnum.down_right,
}
DIAGONAL_DIRECTION_SYMBOLS_BY_ENUM = {v: k for k, v in DIAGONAL_DIRECTION_SYMBOLS.items()}

ADJACENT_DIRECTION_SYMBOLS = {
    **ORTHOGONAL_DIRECTION_SYMBOLS,
    **DIAGONAL_DIRECTION_SYMBOLS,
}

ORTHOGONAL_DIRECTIONS = {
    # y, x
    (-1, 0): OrthogonalDirectionEnum.up,
    (0, 1): OrthogonalDirectionEnum.right,
    (1, 0): OrthogonalDirectionEnum.down,
    (0, -1): OrthogonalDirectionEnum.left,
}
DIAGONAL_DIRECTIONS = {
    # y, x
    (-1, -1): DiagonalDirectionEnum.up_left,
    (-1, 1): DiagonalDirectionEnum.up_right,
    (1, 1): DiagonalDirectionEnum.down_right,
    (1, -1): DiagonalDirectionEnum.down_left,
}
ADJACENT_DIRECTIONS = {**DIAGONAL_DIRECTIONS, **ORTHOGONAL_DIRECTIONS}

ORTHOGONAL_DIRECTIONS_BY_ENUM = {v: k for k, v in ORTHOGONAL_DIRECTIONS.items()}
DIAGONAL_DIRECTIONS_BY_ENUM = {v: k for k, v in DIAGONAL_DIRECTIONS.items()}
ADJACENT_DIRECTIONS_BY_ENUM = {v: k for k, v in ADJACENT_DIRECTIONS.items()}


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
    directions: Iterable[AbstractDirectionEnum] | Iterable[tuple[int, int]] | AbstractDirectionEnum | tuple[int, int],
    pos: tuple[int, int] | Position2D | None = None,
    y: int | None = None,
    x: int | None = None,
    return_reversed: bool | None = None,
) -> tuple[int, int] | Position2D:
    _directions = []
    if isinstance(directions, AbstractDirectionEnum):
        _directions = [directions]
    elif isinstance(directions, tuple) and len(directions) == 2:
        _directions.append(ADJACENT_DIRECTIONS[directions])
    else:
        for d in directions:
            if isinstance(d, AbstractDirectionEnum):
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
        dy, dx = ADJACENT_DIRECTIONS_BY_ENUM[d]
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


def is_horizontal(direction: OrthogonalDirectionEnum) -> bool:
    y, x = ADJACENT_DIRECTIONS_BY_ENUM[direction]
    if not y and not x:
        raise ValueError("y or x must be passed!")
    if y and x:
        return False  # diagonal
    return bool(x)


def reverse_coordinates(coordinates: tuple[int, int]) -> tuple[int, int]:
    x, y = coordinates
    return x * -1, y * -1


def is_a_way_back(direction1: OrthogonalDirectionEnum, direction2: OrthogonalDirectionEnum) -> bool:
    coordinates1 = ADJACENT_DIRECTIONS_BY_ENUM[direction1]
    coordinates2 = ADJACENT_DIRECTIONS_BY_ENUM[direction2]

    reversed_coordinates2 = reverse_coordinates(coordinates2)
    return coordinates1 == reversed_coordinates2
