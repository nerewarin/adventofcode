"""
--- Day 4: Secure Container ---

https://adventofcode.com/2019/day/4

"""

import os
import math
import re
import itertools
import collections

from dataclasses import dataclass
from typing import Tuple


class SecureContainer:
    def __init__(self, inp=None, mode=None):
        if inp is None:
            with open(os.path.join('inputs', '{}.txt'.format(__file__.split('/')[-1].split('.')[0]))) as f:
                inp = f.read()

        self.min, self.max = (int(digit) for digit in inp.split('-'))

    def count_different_passwords(self, mode=None):
        """
        How many different passwords within the range given in your puzzle input meet these criteria?

        - It is a six-digit number.
        - The value is within the range given in your puzzle input.
        - Two adjacent digits are the same (like 22 in 122345).
        - Going from left to right, the digits never decrease; they only ever increase or stay the same (like 111123 or 135679).

        Returns:
            int

        """
        res = 0
        for candidate in range(self.min, self.max):
            # Two adjacent digits are the same (like 22 in 122345).
            candidate_str = str(candidate)
            repeats = collections.defaultdict(int)
            decreases = False
            repeated = 0
            for idx in range(len(candidate_str) - 1):
                first = candidate_str[idx]
                second = candidate_str[idx + 1]
                # Going from left to right, the digits never decrease; they only ever increase or stay the same (like 111123 or 135679).
                if int(second) < int(first):
                    decreases = True
                    break

                if first == second:
                    repeated += 1
                    repeats[first] = repeated
                else:
                    repeated = 0

            if decreases:
                continue

            if not any(val == 1 for val in repeats.values()):
                continue

            res += 1

        return res


inp = '245318-765747'


def part1():
    return SecureContainer(inp).count_different_passwords()

def part2():
    # 569 too low
    return SecureContainer(inp).count_different_passwords(mode=2)


if __name__ == '__main__':
    for res in (
        # part1(),
        part2(),
    ):
        print(res)
