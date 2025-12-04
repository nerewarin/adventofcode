import math
from collections.abc import Sequence
from typing import NamedTuple, TypeVar

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


def get_value_by_position[T](
    pos: Position2D,
    grid: Sequence[Sequence[T]],
) -> T:
    x, y = pos
    return grid[y][x]
