"""
--- Day 16: Flawed Frequency Transmission ---

https://adventofcode.com/2019/day/16

"""

import os
import _tools


class FFT:
    def __init__(self, inp=None, mode=None):
        if inp is None:
            with open(os.path.join('inputs', '{}.txt'.format(__file__.split('/')[-1].split('.')[0]))) as f:
                inp = f.read()

        self.inp = [int(digit) for digit in inp.strip()]
        self.input_len = len(inp)
        self.pattern = [0, 1, 0, -1]
        self.pattern_len = len(self.pattern)
        self.mode = mode

    lcm = _tools.lcm

    def run(self, phases):
        inp = self.inp
        offset = 0

        if self.mode == 2:
            inp = inp * 10000
            _offset = inp[:7]
            offset = int(''.join(str(x) for x in _offset))

        inp = inp[offset:]
        inp_size = len(inp)

        for phase in range(phases):
            val = 0
            for idx in range(inp_size):
                _idx = -idx - 1
                val += inp[_idx]
                val = abs(val) % 10
                inp[_idx] = val

        return ''.join(str(digit) for digit in inp)


inp1 = '12345678'
inp2 = '03036732577212944063491565474664'


def test(test_num):
    if test_num == 1:
        res = FFT(inp1).run(4)
        assert res == '01029498', 'test{} failed!: {}'.format(test_num, res)
    elif test_num == 2:
        msg = FFT(inp2, mode=2).run(100)
        res = msg[:8]
        assert res == '84462026', 'test{} failed!: {}'.format(test_num, res)
    else:
        raise ValueError('test{} not implemented'.format(test_num))
    return 'test{} ok'.format(test_num)


def part1():
    return FFT().run(100)[:8]


def part2():
    return FFT(mode=2).run(100)[:8]


if __name__ == '__main__':
    for res in (
        # test(1),
        # part1(),
        # test(2),
        part2(),
    ):
        print(res)
