"""--- Day 1: Calorie Counting ---
https://adventofcode.com/2022/day/1
"""

from src.utils.input_formatters import cast_2d_list_elements, group_by_blank
from src.utils.test_and_run import run, test


def count_calories(inp, top=1):
    lines = group_by_blank(inp)
    lines = cast_2d_list_elements(lines)

    sums = sorted(sum(line) for line in lines)

    return sum(sums[-top:])


if __name__ == "__main__":
    test(count_calories, top=1, expected=24000)
    run(count_calories, top=1)

    test(count_calories, top=3, expected=45000)
    run(count_calories, top=3)
