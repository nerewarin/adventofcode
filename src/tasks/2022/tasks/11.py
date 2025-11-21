"""----Day 11: Monkey in the Middle ---
https://adventofcode.com/2022/day/11
"""
import dataclasses
import re
import math
from collections import deque, Counter

from src.utils.test_and_run import test, run


class Operation:
    """
    Examples:
        Operation: new = old + 4
        Operation: new = old * old
    """

    rexp = re.compile(r"\s*Operation: new = old ([+*]) ([\w]+)")

    def __init__(self, raw):
        res = self.rexp.match(raw.strip())
        self.sign, self.operand = res.groups()

    def __str__(self):
        return f"{self.sign} {self.operand}"

    def execute(self, old):
        if self.operand == "old":
            op = old
        else:
            op = int(self.operand)

        r = f"{old} {self.sign} {op} % {self.lcm_res}"
        res = eval(r)
        return res


@dataclasses.dataclass
class Monkey:
    items: deque[int] = None
    operation: Operation = None
    test_divider: int = None
    true_res: int = None
    false_res: int = None

    def add_items(self, items):
        self.items = deque(items)

    def test(self, value):
        return not value % self.test_divider


def _parse_puzzle(inp):
    monkeys = []
    block_size = 7
    monkey = Monkey()
    for line_num, line in enumerate(inp):
        mode = line_num % block_size
        if mode == 1:
            _, items = line.split("Starting items: ")
            monkey.add_items(tuple(int(x) for x in items.split(", ")))
        elif mode == 2:
            monkey.operation = Operation(line)
        elif mode == 3:
            _, tst = line.split("Test: divisible by ")
            monkey.test_divider = int(tst)
            monkey.operation.test_divider = monkey.test_divider
        elif mode == 4:
            _, _mn = line.split("throw to monkey ")
            monkey.true_res = int(_mn)
        elif mode == 5:
            _, _mn = line.split("throw to monkey ")
            monkey.false_res = int(_mn)

            monkeys.append(monkey)
            monkey = Monkey()

    return monkeys


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def task(inp, mode=None):
    """monkey business
    """
    monkeys = _parse_puzzle(inp)
    lcm_res = monkeys[0].test_divider
    for monkey in monkeys[1:]:
        lcm_res = lcm(lcm_res, monkey.test_divider)
    for monkey in monkeys:
        monkey.operation.lcm_res = lcm_res

    stats = Counter()
    rounds = 20
    if mode:
        rounds = 10000
    for round_idx in range(rounds):
        for monkey_idx, monkey in enumerate(monkeys):
            while monkey.items:
                worry_level0 = monkey.items.popleft()

                stats[monkey_idx] += 1

                worry_level1 = monkey.operation.execute(worry_level0)

                if mode:
                    worry_level2 = worry_level1
                else:
                    worry_level2 = worry_level1 // 3

                target = monkey.false_res
                tst_res = monkey.test(worry_level2)
                if tst_res:
                    target = monkey.true_res

                monkeys[target].items.append(worry_level2)

    return math.prod(dict(stats.most_common(2)).values())


if __name__ == "__main__":
    # part 1
    test(task, expected=10605)
    run(task)

    # part 2
    test(task, mode=1, expected=2713310158)
    run(task, mode=1)
