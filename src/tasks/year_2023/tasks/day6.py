"""
--- Day 6: Wait For It ---
https://adventofcode.com/2023/day/6

y = (speed = wait * 1) * (t - wait)
y = (x * a) * (b - x)
y = abx - ax^2

find max y
y' = ab - 2ax = a(b - 2x)
x = b / 2
y(x) = ab^2 / 4
...

solve
(x⋅a)⋅(b−x)>c
d  = sqrt(b*b - 4*a*c)
(b + d) / 2*a < x < (b - d) / 2*a
"""

import math
from functools import reduce

from src.utils.test_and_run import run, test


def print(*_):
    pass


def wait_for_it(inp, part=1):
    _times, _min_dists = [[int(v) for v in line.split(":")[-1].strip().split()] for line in inp]
    if part == 2:
        _times, _min_dists = [[int(line.split(":")[-1].strip().replace(" ", ""))] for line in inp]
    options = []
    for time, min_dist in zip(_times, _min_dists):
        print(f"{time=}, {min_dist=}")
        a = 1  # speed increasing per time of wait
        b = time
        c = min_dist
        d_ = math.sqrt(b * b - 4 * a * c)

        x0_ = (b - d_) / 2 * a
        x0 = int(x0_) + 1
        print(f"{x0_=} -> {x0=}")

        x1_ = (b + d_) / 2 * a
        x1 = int(x1_)
        if x1_ == x1:
            x1 -= 1
        print(f"{x1_=} -> {x1=}")

        x = int(x1 - x0) + 1
        print(x)
        options.append(x)

    return reduce(lambda x, y: x * y, options, 1)


if __name__ == "__main__":
    test(wait_for_it, expected=288)
    run(wait_for_it)

    test(wait_for_it, part=2, expected=71503)
    run(wait_for_it, part=2)
