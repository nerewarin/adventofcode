"""
--- Day 21: Springdroid Adventure ---

https://adventofcode.com/2019/day/21

"""

import re

from _intcode_computer import IntcodeComputer

TILE2DRAW = {
    0: '#',
    1: '.',
    2: '~',
    -1: ' ',
}


class NoSignal(Exception):
    pass


class IntcodeComputer21(IntcodeComputer):
    def _get_op3_input(self):
        if self.signals:
            return self.signals.pop(0)
        # return input('Enter instruction:\n')
        self.gen = self._compute()
        raise NoSignal()


class SpringScript:
    _jump_size = 4
    _rexp = re.compile(r'(\w+)\s+(\w)\s(\w)')

    def __init__(self, to_draw=False, extended_mode=False):
        self.to_draw = to_draw
        # Two registers are available: T, the temporary value register, and J, the jump register

        # Your springdroid can detect ground at four distances:
        # one tile away (A), two tiles away (B), three tiles away (C), and four tiles away (D).
        # If there is ground at the given distance, the register will be true; if there is a hole,
        # the register will be false
        self.registers = {
            'T': False,  # temporary value register
            'J': False,  # jump register
            'A': False,  # is there ground 1 tile away
            'B': False,  # is there ground 2 tiles away
            'C': False,  # is there ground 3 tiles away
            'D': False,  # is there ground 4 tiles away
            'E': False,
            'F': False,
            'G': False,
            'H': False,
            'I': False,
        }

        self.computer = IntcodeComputer21()

        self.extended_mode = extended_mode

    @staticmethod
    def _check_is_writable(y):
        return y in ('T', 'J')

    def _and(self, x, y):
        self._check_is_writable(y)

        val = self.registers[x] and self.registers[y]
        self.registers[y] = val
        return val

    def _or(self, x, y):
        self._check_is_writable(y)

        val = self.registers[x] or self.registers[y]
        self.registers[y] = val
        return val

    def _not(self, x, y):
        self._check_is_writable(y)

        val = not self.registers[x]
        self.registers[y] = val
        return val

    @staticmethod
    def _get_string_in_ascii(s):
        return [ord(c) for c in s] + [10]

    def _test(self, program, test_map):
        for idx, reg in enumerate(
            ['A', 'B', 'C', 'D'] if not self.extended_mode
            else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        ):
            v = test_map[idx]
            self.registers[reg] = v

        for line in program:
            if line in ('WALK', 'RUN'):
                break
            res = self._rexp.match(line)
            if not res:
                raise ValueError(line)
            instr, x, y = res.groups()
            method = {
                'AND': self._and,
                'OR': self._or,
                'NOT': self._not,
            }[instr]
            _reg = dict(self.registers)
            print(_reg)

            method(x, y)

            print(f'{instr} {x} {y}')
            print('{}'.format({k: v for k, v in self.registers.items() if v != _reg[k]}))

        res = False
        if self.registers['J']:
            print('jump!')
            res = True

        print()
        return res

    def _get_walk_program(self):
        """
        There are only three instructions available in springscript:

        AND X Y sets Y to true if both X and Y are true; otherwise, it sets Y to false.
        OR X Y sets Y to true if at least one of X or Y is true; otherwise, it sets Y to false.
        NOT X Y sets Y to true if X is false; otherwise, it sets Y to false.

        Returns:
            list of int: ASCII code

        """
        # _jump_if_ground4 = '''
        #     OR {} J   # jump if ground in 4 steps
        # _jump_if_ground4 = 'OR E J'
        _jump_if_ground4 = '''
            OR D J
            {}
        '''.format(
            '' if not self.extended_mode
            else '''
                OR H T
                OR E T
                AND T J
            '''
        )

        _but_dont_jump_if_no_hole = '''
            NOT A T  # 1 if need jump
            NOT T T  # 0 if need jump
            AND B T  # 0 if need jump
            AND C T  # 0 if need jump (one of three tiles has hole)
            {}
            NOT T T  # 1 if need jump
            AND T J
        '''.format(
            # '\n'.join(('AND {} T'.format(s) for s in 'DEF')) if self.extended_mode
            '' if self.extended_mode
            else ''
        )

        do = 'RUN' if self.extended_mode else 'WALK'
        _program = f'''
            {_jump_if_ground4}
            
            {_but_dont_jump_if_no_hole}
            
            {do}
        '''
        program = []
        for _line in _program.split('\n'):
            line = _line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            comment_idx = line.find('#')
            if comment_idx > -1:
                line = line[:comment_idx]
                if not line:
                    continue

                for idx, symbol in enumerate(reversed(line)):
                    if symbol.isalpha():
                       break

                line = line[:-idx]

            program.append(line)

        if self.extended_mode:
            # self._test(program, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, ])

            # bad = '#####..#@########'
            # bad = '#####...#########'
            # bad = '#####..#.########'
            # bad = '#####.#.#...#.###'
            bad = '##.#.#...#.###'
            map = [{'#': 1}.get(x, 0) for x in bad]
            assert not self._test(program, map)

            bad = '##...#########'
            map = [{'#': 1}.get(x, 0) for x in bad]
            assert not self._test(program, map)
        else:
            self._test(program, [1, 1, 1, 1, 1, 0, 1])

        codes = []
        for instruction in program:
            codes.extend(self._get_string_in_ascii(instruction))

        return codes

    def get_amout_of_hull_damage(self):
        try:
            for out in self.computer.gen:
                print(f'{chr(out)}', end='')
        except NoSignal:
            # now enter the program
            for instruction_code in self._get_walk_program():
                print(f'{chr(instruction_code)}', end='')
                self.computer.feed(instruction_code)

        try:
            for out in self.computer.gen:
                print(f'{chr(out)}', end='')
        except ValueError:
            return out


def part1(*args, **kwargs):
    return SpringScript(*args, **kwargs).get_amout_of_hull_damage()


def part2(*args, **kwargs):
    return SpringScript(extended_mode=True).get_amout_of_hull_damage()


if __name__ == '__main__':
    for res in (
        part1(),
        part2(),
    ):
        print(res)
