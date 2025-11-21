"""
--- Day 19: Tractor Beam ---

https://adventofcode.com/2019/day/19

"""
import math
import collections

from _intcode_computer import IntcodeComputer


class TractorBeam:
    def __init__(self, *args, network_size=50, **kwargs):
        pass

    def count_ones(self, max_x, max_y):
        _map = collections.defaultdict(int)
        for y in range(max_y):
            for x in range(max_x):
                computer = IntcodeComputer()
                computer.feed(x)
                computer.feed(y)
                # print(f'feed ({x}, {y})')
                res = next(computer)
                # print(f'received {res}')
                _map[(x, y)] = res

        return sum(val for val in _map.values() if val)

    def find_closest_square_under_beam(self, expected_width, expected_high):
        for y in range(750, 1500):
            for x in range(1500, 2000):
                computer = IntcodeComputer()
                computer.feed(x)
                computer.feed(y)
                # print(f'feed ({x}, {y})')
                value = next(computer)
                if not value:
                    continue

                computer = IntcodeComputer()
                computer.feed(x + 99)
                computer.feed(y)
                value = next(computer)
                if not value:
                    continue

                computer = IntcodeComputer()
                computer.feed(x)
                computer.feed(y + 99)
                value = next(computer)
                if not value:
                    continue

                return _get_part2_result(x, y)

        raise RuntimeError()


def _get_part2_result(x, y):
    res = x * 10000 + y
    if res >= 15240780:
        raise ValueError('answer is too high!', res)
    if res <= 4950286:
        raise ValueError('answer is too low!', res)
    if res in (
        15250787,
        15240780,
        15230780,
        15240781,
        15130775,
    ):
        raise ValueError('entered before!', res)
    return res


def part1(*args, **kwargs):
    return TractorBeam(*args, **kwargs).count_ones(50, 50)


def part2(*args, **kwargs):
    return TractorBeam(*args, **kwargs).find_closest_square_under_beam(100, 100)


if __name__ == '__main__':
    # assert part1() == 121

    for res in (
        part2(),
    ):
        print(res)
        # 15240780 is too high!
