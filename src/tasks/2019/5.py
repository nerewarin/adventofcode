"""
https://adventofcode.com/2019/day/5
"""


def calc():
    with open("inputs/5.txt") as f:
        inp = [int(x) for x in f.read().split(",")]
        # inp = [int(x) for x in '3,9,8,9,10,9,4,9,99,-1,8'.split(',')]

        def get_value(val_or_idx, mode):
            if mode == 0:
                idx = val_or_idx
                return inp[idx]

            val = val_or_idx
            return val

        curr_idx = 0
        while True:
            instruction = inp[curr_idx]

            op = instruction % 100

            _ins = instruction // 100
            mode1 = _ins % 10

            _ins = _ins // 10
            mode2 = _ins % 10

            _ins = _ins // 10
            mode3 = _ins % 10

            if op == 99:
                return inp[0]

            param1 = inp[curr_idx + 1]
            param2 = inp[curr_idx + 2]
            param3 = inp[curr_idx + 3]
            if op < 3:
                shift = 4
                if op == 1:
                    inp[param3] = get_value(param1, mode1) + get_value(param2, mode2)
                elif op == 2:
                    inp[param3] = get_value(param1, mode1) * get_value(param2, mode2)
            elif op < 5:
                shift = 2
                if op == 3:
                    # Opcode 3 takes a single integer as input and saves it to the position given by its only parameter
                    inp[param1] = int(input())
                elif op == 4:
                    print(inp[param1])
            elif op < 7:
                # jump-if-true
                if op == 5 and get_value(param1, mode1):
                    shift = get_value(param2, mode2) - curr_idx
                # jump-if-false
                elif op == 6 and not get_value(param1, mode1):
                    shift = get_value(param2, mode2) - curr_idx
                else:
                    shift = 3
            elif op < 9:
                # less than
                if op == 7 and get_value(param1, mode1) < get_value(param2, mode2):
                    _val = 1
                # equals
                elif op == 8 and get_value(param1, mode1) == get_value(param2, mode2):
                    _val = 1
                else:
                    _val = 0
                inp[param3] = _val
                shift = 4

            curr_idx += shift


def part1():
    return calc()


def part2():
    return calc()


for res in (
    # part1(),
    part2(),
):
    # print(res)
    pass
