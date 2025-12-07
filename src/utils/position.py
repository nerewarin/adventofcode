import math
import typing
from collections.abc import Sequence
from typing import NamedTuple, TypeVar

from src.utils.numbers import get_sign

if typing.TYPE_CHECKING:
    from src.utils.directions import AbstractDirectionEnum

T = TypeVar("T")


class Position2D(NamedTuple):
    x: int | float
    y: int | float

    @classmethod
    def integer_point_from_tuple_of_strings(cls, t: tuple[str, str]) -> "Position2D":
        return cls(*(int(value) for value in t))

    # Vector operations
    # -------------------------
    def __add__(self, other: "Position2D") -> "Position2D":
        return Position2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position2D") -> "Position2D":
        return Position2D(self.x - other.x, self.y - other.y)

    def scale(self, k: int | float) -> "Position2D":
        return Position2D(self.x * k, self.y * k)

    # -------------------------
    # Distances
    # -------------------------
    def distance_to(self, other: "Position2D") -> float:
        """Euclidean distance."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def manhattan_to(self, other: "Position2D") -> int | float:
        """Manhattan (taxicab) distance."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def chebyshev_to(self, other: "Position2D") -> int | float:
        """Diagonal / Chebyshev distance."""
        return max(abs(self.x - other.x), abs(self.y - other.y))

    # -------------------------
    # Geometry: slope
    # -------------------------
    def slope(self) -> float:
        """
        Slope y/x.
        Vertical line (x = 0) → ±inf
        Origin (0,0) → nan
        """
        if self.x == 0:
            if self.y == 0:
                return math.nan
            return math.copysign(math.inf, self.y)
        return self.y / self.x

    def reversed(self) -> "Position2D":
        return Position2D(self.y, self.x)

    def get_actions_to(self, other: "Position2D", allowed_direction="orthogonal") -> list["AbstractDirectionEnum"]:
        if allowed_direction == "orthogonal":
            from src.utils.directions import ORTHOGONAL_DIRECTIONS

            x, y = other - self
            horizontal = [ORTHOGONAL_DIRECTIONS[(0, (get_sign(x)))]] * abs(x)
            vertical = [ORTHOGONAL_DIRECTIONS[(get_sign(y), 0)]] * abs(y)
            return horizontal + vertical
        else:
            raise NotImplementedError(f"{allowed_direction=} is not implemented.")


def get_value_by_position[T](
    pos: Position2D,
    grid: Sequence[Sequence[T]],
) -> T:
    x, y = pos
    return grid[y][x]


def set_value_by_position[T](
    pos: Position2D,
    value: T,
    grid: list[list[T]],
) -> None:
    x, y = pos
    grid[y][x] = value
