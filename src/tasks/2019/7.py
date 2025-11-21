"""
https://adventofcode.com/2019/day/7
"""

import itertools


class Amplifier:
    def __init__(self, num, phase, inputs=None):
        self.num = num  # just a number of Amplifier
        self.phase = phase
        self.signals = []
        self.signals.append(phase)

        if inputs is None:
            with open("inputs/7.txt") as f:
                inputs = f.read()
        self.inputs = [int(x) for x in inputs.split(",")]
        self.gen = self._generator()

    def __repr__(self):
        return f"Amp{self.num}: phase{self.phase}, signals={self.signals}"

    def __next__(self):
        return next(self.gen)

    def feed(self, value):
        # feed the generator
        self.signals.append(value)

    def get_value(self, val_or_idx, mode):
        if mode == 0:
            idx = val_or_idx
            return self.inputs[idx]

        val = val_or_idx
        return val

    def _generator(self):
        curr_idx = 0
        while True:
            instruction = self.inputs[curr_idx]

            op = instruction % 100

            _ins = instruction // 100
            mode1 = _ins % 10

            _ins = _ins // 10
            mode2 = _ins % 10

            _ins = _ins // 10
            mode3 = _ins % 10

            if op == 99:
                return self.inputs[0]

            param1 = self.inputs[curr_idx + 1]
            param2 = self.inputs[curr_idx + 2]
            try:
                param3 = self.inputs[curr_idx + 3]
            except KeyError:
                param3 = None  # dirty but who cares
            if op < 3:
                shift = 4
                if op == 1:
                    self.inputs[param3] = self.get_value(param1, mode1) + self.get_value(param2, mode2)
                elif op == 2:
                    self.inputs[param3] = self.get_value(param1, mode1) * self.get_value(param2, mode2)
            elif op < 5:
                shift = 2
                if op == 3:
                    # Opcode 3 takes a single integer as input and saves it to the position given by its only parameter
                    self.inputs[param1] = int(self.signals.pop(0))
                elif op == 4:
                    out = self.inputs[param1]
                    yield out
                    a = 9
            elif op < 7:
                # jump-if-true
                if op == 5 and self.get_value(param1, mode1):
                    shift = self.get_value(param2, mode2) - curr_idx
                # jump-if-false
                elif op == 6 and not self.get_value(param1, mode1):
                    shift = self.get_value(param2, mode2) - curr_idx
                else:
                    shift = 3
            elif op < 9:
                # less than
                if op == 7 and self.get_value(param1, mode1) < self.get_value(param2, mode2):
                    _val = 1
                # equals
                elif op == 8 and self.get_value(param1, mode1) == self.get_value(param2, mode2):
                    _val = 1
                else:
                    _val = 0
                self.inputs[param3] = _val
                shift = 4
            else:
                raise NotImplementedError(f"op={op} is unknown")

            curr_idx += shift


def part2():
    phases_combinations = itertools.permutations(list(range(5, 10)))
    last_system_outputs = []
    inp = None
    # tests
    # phases_combinations, inp = [[[9, 8, 7, 6, 5]]], '3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5'
    # phases_combinations. inp = [[[9,7,8,5,6]]], '''3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10'''
    for phases in phases_combinations:
        amplifiers = [Amplifier(ind, phases[ind], inp) for ind in range(5)]
        amplifiers[0].feed(0)

        last_system_output = None
        outputs = []
        amp_idx = 0
        next_amp = amplifiers[amp_idx]
        lap = 0
        while True:
            amp = next_amp
            try:
                output = next(amp)
            except StopIteration:
                last_system_outputs.append(last_system_output)
                break

            amp_idx = amp_idx + 1
            if amp_idx == len(amplifiers):
                amp_idx = 0
                last_system_output = output
                outputs.append(output)
                lap += 1

            next_amp = amplifiers[amp_idx]
            next_amp.feed(output)

    return max(last_system_outputs)


for res in (
    # part1(),
    part2(),
):
    # print(res)
    print(res)
