"""
--- Day 1: Trebuchet?! ---
https://adventofcode.com/2023/day/1
"""

import re

from src.utils.helpers import HUMAN_DIGITS
from src.utils.test_and_run import run, test

_pattern = "|".join(f"({word})" for word in HUMAN_DIGITS)
_DIGITS_REXP = re.compile(rf"({_pattern})")
WORD2DIGIT = {HUMAN_DIGITS[n]: n + 1 for n in range(len(HUMAN_DIGITS))}


def trebuchet2(inp, **_):
    res = 0
    for line in inp:
        v = None
        for is_reversed, it in enumerate(
            [line, reversed(line)],
        ):
            candidate = ""
            for s in it:
                if s.isdigit():
                    if v is None:
                        v = 10 * int(s)
                    else:
                        v += int(s)
                    break

                if is_reversed:
                    candidate = s + candidate
                else:
                    candidate += s

                digit = None
                for word in WORD2DIGIT:
                    if is_reversed:
                        flag = candidate.startswith(word)
                    else:
                        flag = candidate.endswith(word)
                    if flag:
                        digit = WORD2DIGIT[word]
                        break

                if digit is not None:
                    if v is None:
                        v = 10 * digit
                    else:
                        v += digit
                    break

        res += v

    return res


def trebuchet(inp):
    res = 0
    for line in inp:
        v = None
        for it in (
            line,
            reversed(line),
        ):
            for s in it:
                if s.isdigit():
                    if v is None:
                        v = 10 * int(s)
                    else:
                        v += int(s)
                    break

        res += v

    return res


if __name__ == "__main__":
    test(trebuchet, expected=142)
    run(trebuchet)

    test(trebuchet2, test_part=3, expected=91)

    test(trebuchet2, test_part=2, expected=281)

    res = run(trebuchet2)
    assert 54712 < res < 59556
    assert res != 54731
