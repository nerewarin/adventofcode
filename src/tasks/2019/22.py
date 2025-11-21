"""
--- Day 22: Slam Shuffle ---

https://adventofcode.com/2019/day/22

"""

import re
import collections

import _tools


class SlamShuffle:
    _increment_rexp = re.compile(r'deal with increment ([-]*\d+)')
    _cut_rexp = re.compile(r'cut ([-]*\d+)')

    def __init__(self, inp=None, deck_size=10007, deck=None):
        if inp is None:
            inp = _tools.get_puzzle_input(scalar_type=str, delimeter='\n', multiline=True)
        else:
            inp = [val.strip() for val in inp.split('\n') if val.strip()]
        self.inp = inp
        self._deck_size = deck_size
        self.deck = deck or list(range(self._deck_size))

    def _deal_with_increment(self, period):
        temp = collections.defaultdict(int)
        temp_idx = 0
        deck_idx = 0
        while deck_idx < self._deck_size:
            temp[temp_idx] = self.deck[deck_idx]
            temp_idx = (temp_idx + period) % self._deck_size
            deck_idx += 1
        self.deck = [temp[idx] for idx in range(self._deck_size)]

    def _shuffle(self):
        _last_instruction = None
        for instr_idx, raw_instruction in enumerate(self.inp):
            if len(set(self.deck)) != self._deck_size:
                raise ValueError(f'error dealing {_last_instruction}')
            if raw_instruction == 'deal into new stack':
                self.deck = self.deck[::-1]
                _last_instruction = raw_instruction
                continue

            _incr_instruction = self._increment_rexp.match(raw_instruction)
            if _incr_instruction:
                period = int(_incr_instruction.group(1))
                self._deal_with_increment(period)
                _last_instruction = raw_instruction
                continue

            _cut_instruction = self._cut_rexp.match(raw_instruction)
            if _cut_instruction:
                idx = int(_cut_instruction.group(1))
                self.deck = self.deck[idx:] + self.deck[:idx]
                _last_instruction = raw_instruction
                continue

            raise ValueError(f'unknown {raw_instruction}')
        return

    def get_position(self, value):
        f"""
        
        Args:
            value (int): value in deck (from 0 to 10006) 

        Returns:
            int: position of {value} in deck
            
        """
        self._shuffle()
        return self.deck.index(value)

    @staticmethod
    def _get_inct_pos(period, position, deck_size):
        if position == 0:
            return 0
        return (period * position - 1) % deck_size

    def get_a_b(self, position, deck_size, cycles):
        n = deck_size
        a, b = 1, 0
        for instr_idx, raw_instruction in enumerate(self.inp):
            print(f'a={a} b={b}')
            print(f'{position}: {raw_instruction}')
            if raw_instruction == 'deal into new stack':
                a = -a % n
                b = (-b + n - 1) % n
                continue

            _incr_instruction = self._increment_rexp.match(raw_instruction)
            if _incr_instruction:
                period = int(_incr_instruction.group(1))
                k = period
                a = a * k % n
                b = b * k % n
                continue

            _cut_instruction = self._cut_rexp.match(raw_instruction)
            if _cut_instruction:
                idx = int(_cut_instruction.group(1))
                b = (b - idx) % n
                continue
            raise ValueError(f'unknown {raw_instruction}')

        pow_ = pow(a, cycles, deck_size)
        a, b = pow_, b * (pow_ - 1) * pow(a - 1, n - 2, n) % n

        result = pow(a, n-2, n) * (position - b) % n
        return result


def _str_to_deck(s):
    return [int(x) for x in s.split(' ')]


def test1():
    ss1 = SlamShuffle('''
        deal with increment 7
        deal into new stack
        deal into new stack
    ''', deck_size=10)
    ss1._shuffle()
    assert ss1.deck == _str_to_deck('0 3 6 9 2 5 8 1 4 7')

    ss2 = SlamShuffle('''
        cut 6
        deal with increment 7
        deal into new stack
    ''', deck_size=10)
    ss2._shuffle()
    assert ss2.deck == _str_to_deck('3 0 7 4 1 8 5 2 9 6')

    ss3 = SlamShuffle('''
deal with increment 7
deal with increment 9
cut -2
    ''', deck_size=10)
    ss3._shuffle()
    assert ss3.deck == _str_to_deck('6 3 0 7 4 1 8 5 2 9')

    ss4 = SlamShuffle('''
deal into new stack
cut -2
deal with increment 7
cut 8
cut -4
deal with increment 7
cut 3
deal with increment 9
deal with increment 3
cut -1
    ''', deck_size=10)
    ss4._shuffle()
    assert ss4.deck == _str_to_deck('9 2 5 8 1 4 7 0 3 6')


def part1(*args, **kwargs):
    pos = 2019
    result = SlamShuffle(*args, **kwargs).get_position(pos)
    # result = SlamShuffle(*args, **kwargs).get_a_b(pos, 1007, 1)

    if result != 5472:
        raise ValueError(f'{result} is wrong or do u use another input? then skip this message')
    return result


def part2(*args, **kwargs):
    deck_size = 119315717514047
    cycles = 101741582076661
    ss = SlamShuffle(*args, **kwargs)
    pos2020 = 2020
    pos2020 = ss.get_a_b(pos2020, deck_size, cycles)
    if pos2020 <= 54854748320449:
        raise ValueError(f'{pos2020} is too low!')
    return pos2020


if __name__ == '__main__':
    for res in (
        test1(),
        part1(),
        part2(),
    ):
        print(res)
