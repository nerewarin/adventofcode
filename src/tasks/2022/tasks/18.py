"""--- Day 17: Pyroclastic Flow ---
https://adventofcode.com/2022/day/17
"""
import datetime
import functools
from collections import defaultdict

from src.utils.test_and_run import test, run

DIRECTIONS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]
CACHE = {}


def _parse_puzzle(inp):
    res = []
    for l in inp:
        res.append([int(x) for x in l.split(",")])
    return res


def init(points):
    dd_bool = functools.partial(functools.partial(defaultdict, bool))
    matrix = defaultdict(functools.partial(defaultdict, dd_bool))

    def set(x, y=None, z=None):
        if y is None and z is None:
            try:
                x, y, z = x
            except:
                x, y, z = x
        if x not in matrix:
            matrix[x] = {}
        if y not in matrix[x]:
            matrix[x][y] = {}
        matrix[x][y][z] = 1

    for point in points:
        set(point)

    return matrix
def get(x, y=None, z=None, matrix=None):
    if y is None and z is None:
        x, y, z = x
    if x in matrix:
        if y in matrix[x]:
            if z in matrix[x][y]:
                return True

def vector_sum(p1, p2):
    assert len(p1) == len(p2)
    return tuple([p1[i] + p2[i] for i in range(len(p1))])


def empty_is_inner_hole(p, matrix, directions_to_check=None):
    global CACHE
    if p in CACHE:
        res = CACHE[p]
        print(p, res)
        return res

    res = False
    if get(p, matrix=matrix):
        res = True
        CACHE[p] = res
        print(p, res)
        return res

    locals = 0
    if directions_to_check is None:
        directions_to_check = DIRECTIONS
    for d, dir1 in enumerate(directions_to_check):
        p1 = vector_sum(p, dir1)
        sx, sy, sz = dir1
        if get(p1, matrix=matrix):
            locals += 1
        else:
            if any([not (0 < aa < 20) for aa in p1]):
                CACHE[p] = False
                return False
            all_directions_but_back = [d for d in directions_to_check if d != (-sx, -sy, -sz)]
            # dd = sorted(DIRECTIONS, key=lambda d: d != (sx, sy, sz))
            r = empty_is_inner_hole(p1, matrix, all_directions_but_back)
            print(p, r)
            if r:
            # if empty_is_inner_hole(p1, matrix):
                locals += 1
            else:
                CACHE[p] = False
                return False

    if locals == len(directions_to_check):
        res = True

    CACHE[p] = res
    print(p, res)
    return res

def task1(inp, p2=False):
    points = _parse_puzzle(inp)
    return _task(points, p2)

def _task(points, p2=False):
    # init 3d world
    global CACHE
    CACHE = {}
    matrix = init(points)

    res = 0
    for x, y, z in points:
        local = 6
        for sx, sy, sz in DIRECTIONS:
            p = x + sx, y + sy, z + sz
            if get(p, matrix=matrix):
                local -= 1

        res += local

    if p2:
        holes = 0
        for x, y, z in points:
            for sx, sy, sz in DIRECTIONS:
                p = x + sx, y + sy, z + sz

                if get(p, matrix=matrix):
                    continue

                all_directions_but_back = [d for d in DIRECTIONS if d != (-sx, -sy, -sz)]
                ishole = empty_is_inner_hole(p, matrix, all_directions_but_back)
                # dd = sorted(DIRECTIONS, key=lambda d: d != (sx, sy, sz))
                # ishole = empty_is_inner_hole(p, matrix, dd)

                if ishole:
                    holes += 1

        res -= holes
    return res


def part1(expected):
    start = datetime.datetime.now()

    try:
        test(task1, expected=expected)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    res = run(task1)  # 3346
    end = datetime.datetime.now()
    print(end - start)


def task2(inp):
    return task1(inp, p2=True)


def part2(expected):
    start = datetime.datetime.now()

    try:
        test(task2, expected=expected)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    res = run(task2)  # 3178 is too high
    # 3118 is too high
    # 1975 is too low
    end = datetime.datetime.now()
    print(end - start)


if __name__ == "__main__":
    # part1(expected=64)
    assert _task([(0,0,0), (0, 0,1)], p2=True) == 10
    assert _task([(0,0,0), (0, 0,2)], p2=True) == 12
    assert _task([
        (0,0,0),
        (0,0,1),
        (0,0,2),
        (0, 0,3)
    ], p2=True) == 12
    part2(expected=58)
