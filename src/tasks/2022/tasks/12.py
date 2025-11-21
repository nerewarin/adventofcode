"""--- Day 12: Hill Climbing Algorithm ---
https://adventofcode.com/2022/day/12
"""
from src.utils.pathfinding import astar
from src.utils.test_and_run import test, run


def _parse_puzzle(inp):
    maze = []
    start = None
    end = None
    for row_num, line in enumerate(inp):
        row = []
        for col_num, elm in enumerate(line):
            if elm == 'S':
                elm = 'a'
                start = (row_num, col_num)
            elif elm == 'E':
                elm = 'z'
                end = (row_num, col_num)

            signal = ord(elm) - 96  # a = 1

            row.append(signal)

        maze.append(row)

    return maze, start, end


def task(inp):
    """
    hill_climbing_algorithm
    """
    maze, start, end = _parse_puzzle(inp)

    path = astar(maze, start, end, allow_diagonal=False, signal_limit=1)

    # print(path)
    print(f"res: {len(path) - 1}")

    return len(path) - 1


if __name__ == "__main__":
    # part 1
    test(task, expected=31)
    res = run(task)
    assert res == 490, f"res {res} != {490}"

    # part 2
    # test(task, mode=1, expected=29)
    # run(task, mode=1)
    print(res - 2)  # solved with eyes
