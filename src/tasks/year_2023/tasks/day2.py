"""
--- Day 2: Cube Conundrum---
https://adventofcode.com/2023/day/2
"""

import re
from functools import reduce
from operator import mul

from src.utils.test_and_run import run, test

_REXP = re.compile(r"(\d+)\s*(\w+)")


def cube_conundrum(inp, limit=None, **_):
    colors = ("red", "green", "blue")
    if limit:
        limit = dict(zip(colors, limit))

    valid = []
    for i, line in enumerate(inp):
        line = line[line.index(":") + 1 :].strip()

        # Extract color and number pairs using regular expression
        matches = _REXP.findall(line)

        # Create a dictionary to store the maximum number for each color
        max_numbers = {}

        # Iterate through matches and update the dictionary
        for number, color in matches:
            number = int(number)

            if not color.isdigit() and color not in max_numbers or number > max_numbers[color]:
                max_numbers[color] = number

        if limit:
            for color, max_v in limit.items():
                if max_numbers[color] > max_v:
                    break
            else:
                valid.append(i + 1)
        else:
            # part 2
            # Calculate the product of all values
            product = reduce(mul, max_numbers.values(), 1)
            valid.append(product)

            # print(line)
            # print(max_numbers)
            # print(product)
            # print()

    return sum(valid)


if __name__ == "__main__":
    test(cube_conundrum, limit=(12, 13, 14), expected=8)
    run(cube_conundrum, limit=(12, 13, 14))

    test(cube_conundrum, test_part=2, expected=400)
    test(cube_conundrum, expected=2286)
    assert run(cube_conundrum) < 249275
