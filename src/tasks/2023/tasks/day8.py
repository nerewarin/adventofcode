"""
--- Day 8: Haunted Wasteland ---
https://adventofcode.com/2023/day/8
"""
import re
import math

from src.utils.test_and_run import run, test

pattern = re.compile(r'(\w+)\s*=\s*\(([^)]+)\)')


def _parse_inp(inp):
    instructions = inp[0]

    nodes = {}
    for line in inp[2:]:
        match = pattern.match(line)
        if match:
            name = match.group(1)
            values = tuple(map(str.strip, match.group(2).split(',')))
            nodes[name] = values

    return instructions, nodes


def find_minimum_common_divisor(*lists):
    min_values = [min(lst) for lst in lists]

    # Find the minimum common multiple of the minimum values
    min_common_divisor = math.lcm(*min_values)

    # Check for common divisors
    for i in range(min_common_divisor, math.prod(min_values) + 1, min_values[0]):
        if all(i % value == 0 for value in min_values[1:]):
            min_common_divisor = i
            break

    return min_common_divisor


def haunted_wasteland(inp):
    instructions, nodes = _parse_inp(inp)

    current = "AAA"
    end = "ZZZ"
    lap = 0

    while current != end:
        for i in instructions:
            if i == 'R':
                ind = 1
            elif i == 'L':
                ind = 0
            else:
                raise ValueError

            current = nodes[current][ind]

        lap += 1

    return lap * len(instructions)

def haunted_wasteland2(inp, **kw):
    instructions, nodes = _parse_inp(inp)
    print(f"{instructions[:20]}...")

    is_end = lambda x: x[-1] == "Z"
    is_start = lambda x: x[-1] == "A"

    currents = [
        node for node in nodes if is_start(node)
    ]
    initial_currents = list(currents)
    ends = [
        node for node in nodes if is_end(node)
    ]
    lap = 0

    instructions_length = len(instructions)
    matches = {i: [] for i in range(len(currents))}

    while not all((current in ends for current in currents)):
        for c_ind in range(len(currents)):
            for i, instr in enumerate(instructions):
                if instr == 'R':
                    ind = 1
                elif instr == 'L':
                    ind = 0
                else:
                    raise ValueError

                current = nodes[currents[c_ind]][ind]
                currents[c_ind] = current

        lap += 1

        for i, current in enumerate(currents):
            if current in ends:
                total_steps = lap * instructions_length

                matches[i].append(total_steps)

                print(f"{initial_currents[i]} meets {current} at {lap=}, {total_steps=}")

        if all(matches.values()):
            res = find_minimum_common_divisor(*list(matches.values()))
            return res


if __name__ == "__main__":
    # test(haunted_wasteland, 2)
    # run(haunted_wasteland)

    test(haunted_wasteland2, test_part=2, expected=6)
    assert run(haunted_wasteland2) > 443
