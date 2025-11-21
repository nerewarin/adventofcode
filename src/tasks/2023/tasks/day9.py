"""
--- Day 8: Mirage Maintenance ---
https://adventofcode.com/2023/day/9
"""

from src.utils.test_and_run import run, test


def guess_next_number(line, part=1):
    inp = list(map(int, line.split()))
    layers = [(inp)]
    last_layer = inp
    while any([elm != 0 for elm in last_layer]):
        layer = []
        for i, a in enumerate(last_layer[:-1]):
            b = last_layer[i + 1]
            diff = b - a
            layer.append(diff)
        layers.append(layer)
        last_layer = layer

    # print(layers)
    bot = 0
    for i, layer in enumerate(reversed(layers)):
        if part == 2:
            next = layer[0] - bot
        else:
            next = bot + layer[-1]

        if i == len(layers) - 1:
            return next

        bot = next

    raise RuntimeError("couldnt find solution")


def mirage(inp, **kw):
    return sum([guess_next_number(line, **kw) for line in inp])


if __name__ == "__main__":
    test(mirage, 114)
    run(mirage)
    test(mirage, part=2, expected=2)
    run(mirage, part=2)
