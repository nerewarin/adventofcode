"""--- Day 6: Tuning Trouble ---
https://adventofcode.com/2022/day/6
"""
import collections

from src.utils.test_and_run import test, run


def tuning_trouble(inp, max_len=4):
    inp = inp[0]

    fringe = collections.deque(maxlen=max_len)
    for idx, char in enumerate(inp):
        fringe.append(char)
        if len(fringe) == max_len == len(set(fringe)):
            return idx + 1


if __name__ == "__main__":
    task = tuning_trouble

    # part 1
    test(task, expected=5)
    run(task)

    # part 2
    test(task, max_len=14, expected=23)
    run(task, max_len=14)
