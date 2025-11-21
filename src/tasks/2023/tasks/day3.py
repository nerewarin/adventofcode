"""
--- Day 3: Gear Ratios  ---
  """
import re
from collections import defaultdict

from src.utils.test_and_run import run, test

_NUMBERS = re.compile(r'(\d+)')
_SYMBOLS = re.compile(r'[^0-9.]')


def is_part(start, end, i, inp):
    row = inp[i]
    l = len(row)
    row_symbols = [(match.start(), match.end()) for match in _SYMBOLS.finditer(row)]
    if row_symbols:
        if start > 0:
            left = inp[i][start - 1]
            if _SYMBOLS.match(left):
                if left == "*":
                    return i, start - 1
                return True
        if end < l - 1:
            right = inp[i][end]
            if _SYMBOLS.match(right):
                if right == "*":
                    return i, end
                return True

    if i > 0:
        prev = inp[i - 1]
        row_symbols = [(match.start(), match.end()) for match in _SYMBOLS.finditer(prev)]
        if row_symbols:
            for top_i in range(start - 1, end + 1):
                if 0 < top_i < l - 1:
                    top = prev[top_i]
                    if _SYMBOLS.match(top):
                        if top == "*":
                            return i - 1, top_i
                        return True

    if i < l - 1:
        bot = inp[i + 1]
        row_symbols = [(match.start(), match.end()) for match in _SYMBOLS.finditer(bot)]
        if row_symbols:
            for bot_i in range(start - 1, end + 1):
                if 0 < bot_i < l - 1:
                    bot_el = bot[bot_i]
                    if _SYMBOLS.match(bot_el):
                        if bot_el == "*":
                            return i + 1, bot_i
                        return True
    return False


def gear_ratios(inp, **_):
    valid = []
    for i, line in enumerate(inp):
        numbers = [(match.start(), match.end()) for match in _NUMBERS.finditer(line)]

        for start, end in numbers:
            value = int(line[start: end])
            if is_part(start, end, i, inp):
                valid.append(value)
                # print(value)
            # else:
                # print(-value)

    return sum(valid)


def gear_ratios2(inp, **_):
    valid = []
    gear_to_numbers = defaultdict(list)
    for i, line in enumerate(inp):
        numbers = [(match.start(), match.end()) for match in _NUMBERS.finditer(line)]

        for start, end in numbers:
            value = int(line[start: end])
            is_a_part = is_part(start, end, i, inp)
            if isinstance(is_a_part, tuple):
                gear_pos = is_a_part
                gear_to_numbers[gear_pos].append(value)
                valid.append(value)
            elif is_a_part is True:
                valid.append(value)

    res = 0
    for gear, numbers in gear_to_numbers.items():
        if len(numbers) == 2:
            res += numbers[0] * numbers[1]

    return res


if __name__ == "__main__":
    test(gear_ratios, expected=4361)
    run(gear_ratios)

    test(gear_ratios2, expected=467835)
    run(gear_ratios2)
