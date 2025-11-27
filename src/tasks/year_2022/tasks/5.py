"""--- Day 5: Supply Stacks ---
https://adventofcode.com/2022/day/5
"""

import re
from collections import defaultdict

from src.utils.test_and_run import run, test

_COMMAND = re.compile(r"move (\d+) from (\d+) to (\d+)")


def supply_stacks(inp, settlement="reversed"):
    stacks = defaultdict(list)
    stack_lines = []
    for level, line in enumerate(inp):
        if line.startswith(" 1"):
            stack_levels = max([int(level) for level in re.sub(r"\s+", " ", line).split(" ") if level])
            inp = inp[level + 2 :]
            break
        stack_lines.append(line)
    else:
        raise ValueError("no stack_indexes found")

    for line in stack_lines:
        for level in range(stack_levels):
            sym = line[1 + level * 4].strip()
            if sym:
                stacks[level].append(sym)

    for cmd in inp:
        amount, src, dst = (int(s) for s in _COMMAND.match(cmd).groups())

        src -= 1
        dst -= 1
        items = stacks[src][:amount]

        stacks[src] = stacks[src][amount:]

        if settlement == "reversed":
            items.reverse()
        stacks[dst] = items + stacks[dst]

    return "".join(stacks[idx][0] if stacks[idx] else "" for idx in range(stack_levels))


if __name__ == "__main__":
    # part 1
    test(supply_stacks, expected="CMZ")
    run(supply_stacks)

    # part 2
    test(supply_stacks, settlement="straight", expected="MCD")
    run(supply_stacks, settlement="straight")
