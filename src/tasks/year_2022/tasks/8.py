"""---- Day 8: Treetop Tree House ---
https://adventofcode.com/2022/day/8
"""

import math

from src.utils.input_formatters import cast_2d_list_elements
from src.utils.test_and_run import run, test


def count_visible_trees(inp):
    rows = len(inp)
    cols = len(inp[0])
    # create a 2d-map
    m = cast_2d_list_elements(inp)

    def _is_visible():
        # edge
        if not row_num or row_num == rows - 1 or not col_num or col_num == cols - 1:
            return True

        col = [m[r][col_num] for r in range(rows)]
        for direction in (
            # left
            row[:col_num],
            # right
            row[col_num + 1 :],
            # top
            col[:row_num],
            # bot
            col[row_num + 1 :],
        ):
            if max(direction) < size:
                return True

        return False

    is_visible = 0
    for row_num, row in enumerate(m):
        for col_num, size in enumerate(row):
            if _is_visible():
                is_visible += 1

    return is_visible


def choose_best_spot(inp):
    m = cast_2d_list_elements(inp)
    max_scenic_score = 0

    rows = len(inp)
    for row_num, row in enumerate(m):
        for col_num, size in enumerate(row):
            score_by_size = []
            col = [m[r][col_num] for r in range(rows)]
            for i, direction in enumerate(
                (
                    # left
                    reversed(row[:col_num]),
                    # right
                    row[col_num + 1 :],
                    # top
                    reversed(col[:row_num]),
                    # bot
                    col[row_num + 1 :],
                )
            ):
                size_score = 0
                for neighbor_size in direction:
                    size_score += 1
                    if neighbor_size >= size:
                        break
                score_by_size.append(size_score)

            score = math.prod(score_by_size)
            if score > max_scenic_score:
                max_scenic_score = score

    return max_scenic_score


if __name__ == "__main__":
    task = count_visible_trees

    # part 1
    test(task, expected=21)
    run(task)

    # part 2
    test(choose_best_spot, expected=8)
    run(choose_best_spot)
