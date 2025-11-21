"""--- Day 9: Rope Bridge ---
https://adventofcode.com/2022/day/9
"""
from functools import partial

from src.utils.test_and_run import test, run


class Register:
    def __init__(self, mode):
        self.x = 1
        self.part2 = mode == 'draw'
        if self.part2:
            self.cycle = 0
        else:
            self.cycle = 1

        self.canvas = ""

    def _draw(self):
        prnt = False
        if self.part2:
            # update crt
            cx = self.cycle % 40
            if self.x - 1 <= cx <= self.x + 1:
                self.canvas += "#"
            else:
                self.canvas += "."

            if len(self.canvas) == 40:
                prnt = self.canvas
                self.canvas = ""
        return prnt

    def addx(self, v):
        """
        addx V takes two cycles to complete.
         After two cycles, the X register is increased by the value V. (V can be negative.)
        """
        prnt = self._draw()

        strength = 0
        self.cycle += 1
        if self.cycle == 20 or not (self.cycle - 20) % 40:
            strength = self.cycle * self.x
            # print(self.cycle, f"addx {v} (1)", self.x, strength)

        prnt2 = self._draw()
        prnt = prnt or prnt2

        self.cycle += 1
        self.x += int(v)
        if self.cycle == 20 or not (self.cycle - 20) % 40:
            strength = self.cycle * self.x
            # print(self.cycle, f"addx {v} (2)", self.x, strength)

        if self.part2:
            return prnt
        return strength

    def noop(self):
        prnt = self._draw()

        strength = 0
        self.cycle += 1
        if self.cycle == 20 or not (self.cycle - 20) % 40:
            strength = self.cycle * self.x
            # print(self.cycle, f"noop", self.x, strength)
        if self.part2:
            return prnt
        return strength


def task(inp, mode=None):
    r = Register(mode)
    res_ = 0
    s = []
    for cmd_num, cmd in enumerate(inp):
        # print(cmd)
        fn, value, *_ = (cmd + ' dummy').split(" ")

        f = {
            "noop": r.noop,
            "addx": partial(r.addx, value)
        }[fn]
        res = f()
        if isinstance(res, str):
            print(res)
            s.append(res)
        elif res and res != 0:
            # print("cmd_num", cmd_num, cmd)
            res_ += res

    if s:
        print()
        return s
    return res_


if __name__ == "__main__":
    # part 1
    test(task, expected=13140)
    run(task)

    # part 2
    test(task, mode="draw", expected=[line for line in """##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######.....""".split("\n")])
    run(task, mode="draw")
