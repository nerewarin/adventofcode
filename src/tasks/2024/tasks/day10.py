import logging
import os

from src.utils.test_and_run import test

# Configure logging based on environment variable
log_level = os.getenv("level", "INFO")
logging.basicConfig(level=log_level)
_logger = logging.getLogger(__name__)

# from math lessons:
#
# y
# ^
# |
# |
# |
# .----> x
#
# from computer history (first display)
# 0...-> v
# ---->
# ...... >    ..so
#
# .----> x
# |
# |
# |
# v
# y
DIRECTIONS = [
    # y, x
    (0, -1),  # left
    (-1, 0),  # down
    (0, 1),  # right
    (1, 0),  # top
]


def out_of_borders(y, x, grid):
    max_y = len(grid)
    if y < 0 or y >= max_y:
        return True

    max_x = len(grid[y])
    if x < 0 or x >= max_x:
        return True

    return False


def get_trailhead_score(
    y: int,
    x: int,
    grid: list[list[int]],
    path: list[tuple[int, int]] | None = None,
    reached_nines: list[tuple[int, int]] | None = None,
) -> int:
    if path is None:
        path = []

    if reached_nines is None:
        reached_nines = []

    pos = (y, x)

    val0 = grid[y][x]
    if val0 == 9 and pos not in reached_nines:
        reached_nines.append(pos)
        path.append(pos)
        _logger.debug(path)
        return 1

    score = 0

    for dy, dx in DIRECTIONS:
        y1 = y + dy
        x1 = x + dx

        if out_of_borders(y1, x1, grid):
            continue

        val1 = grid[y1][x1]
        if val1 != val0 + 1:
            continue

        score += get_trailhead_score(y1, x1, grid, path + [pos], reached_nines)

    return score


def task1(input: list[str]) -> int:
    grid = []
    for y, row in enumerate(input):
        new_row = []
        for x, v in enumerate(row):
            if v.isdigit():
                new_row.append(int(v))
            else:
                new_row.append(-1)
        grid.append(new_row)

    result = 0
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val == 0:
                score = get_trailhead_score(y, x, grid)
                result += score

    return result


if __name__ == "__main__":
    test(task1, 1)
    test(task1, 2, test_part=2)
    test(task1, 4, test_part=3)
    test(task1, 3, test_part=4)
    test(task1, 36, test_part=5)
