"""
--- Day 9: Sensor Boost ---

https://adventofcode.com/2019/day/3
"""
from _intcode_computer import IntcodeComputer


def part1(*args, **kwargs):
    return IntcodeComputer(*args, signals=[1], **kwargs).compute()


def part2(*args, **kwargs):
    return IntcodeComputer(*args, signals=[2], **kwargs).compute()


if __name__ == '__main__':
    tst_mem1 = '109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99'
    tst_mem2 = '1102,34915192,34915192,7,4,7,99,0'
    tst_mem3 = '104,1125899906842624,99'
    for res in (
        # part1(tst_mem1),
        # part1(tst_mem2),
        # part1(tst_mem3),
        part1(),
        part2(),
    ):
        # print(res)
        pass