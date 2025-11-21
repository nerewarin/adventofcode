"""
--- Day 25: Cryostasis ---

https://adventofcode.com/2019/day/25

"""

import re
import collections
import itertools
import warnings
import random
import math

import _tools
from _intcode_computer import ASCIICapableComputer


_regexp = re.compile(r'- ([\w ]+)')


class Cryostasis(ASCIICapableComputer):
    directions = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_msg = None
        # self._doors = []
        # self._visited_doors = []
        self._map = []

    def _get_msg(self, *args, **kwargs):
        self.last_msg = super(Cryostasis, self)._get_msg(*args, **kwargs)
        return self.last_msg

    def _extract_info(self, s, msg=None):
        if msg is None:
            msg = self.last_msg

        if s not in msg:
            return []

        info = []
        for line in msg[msg.rfind(s) + len(s):].split('\n')[1:]:
            _res = _regexp.match(line)
            if not _res:
                break

            res = _res.group(1)
            info.append(res)

        return info

    def run(self):
        def _get_door():
            if any(door not in directions for door in doors):
                raise ValueError(f'wrong door parsed: {doors}')

            _sorted_doors = list(sorted(doors, key=lambda door: place2moves[place][door]))
            door = _sorted_doors[0]
            place2moves[place][door] += 1
            return door

        def get_state(msg, to_print, autoloot=True):
            place = msg.split('==')[1].strip()
            if place == 'Engineering':
                a= 9
            _d = msg[msg.rfind(place) + len(place) + 4:]
            description = _d[:_d.find('\n')]
            if 'Command?' not in msg:
                raise NotImplementedError(msg)

            items = self._extract_info('Items here:', msg)
            if any(item in directions for item in items):
                raise ValueError(f'wrong item parsed: {items}')
            if items and autoloot:
                for item in items:
                    bad_items = (
                        'giant electromagnet',
                        'escape pod',
                        'infinite loop',
                        'molten lava',
                        'photons',
                    )
                    if item in bad_items:
                        warnings.warn(f'found {item!r} in {place!r} but skip it!')
                        continue

                    self.feed(f'take {item}', to_print=to_print)

            self.feed('inv')
            inv_msg = self._get_msg(to_print=to_print)
            inv = self._extract_info('Items in your inventory:', inv_msg)

            doors = self._extract_info('Doors here lead:', msg[msg.index(place):])
            return place, description, items, inv, doors

        directions = self.directions
        step = 1
        place = None
        place2moves = collections.defaultdict(collections.Counter)
        msg = self._get_msg(to_print=True)
        to_print = False
        last_move = None
        while True:
            place, description, items, inv, doors = get_state(msg, to_print)
            # print(f'{step}. {place}: last_move {last_move!r}, doors ({doors}), items: {items}')
            if len(inv) >= 8 and place == 'Security Checkpoint':
                break

            door = _get_door()

            self.feed(door, to_print=to_print)

            self._map.append({
                'place': place,
                'doors': doors,
                'move': door,
                'items': items,
                'inv': inv,
                'description': description,
            })

            step += 1

            last_move = door

            msg = self._get_msg(to_print=to_print)
            if not msg:
                raise ValueError()

        '''
            Security Checkpoint
        '''
        default_msg = self._get_msg(to_print=True)
        prev_nav_doors = None
        nav_doors = None
        all_inv = list(inv)
        probes = math.factorial(len(all_inv)) * 2
        # to_print = True
        while msg == default_msg or nav_doors == prev_nav_doors:
            if not probes:
                raise ValueError('no probes left')

            random.shuffle(inv)
            for i, item in enumerate(inv):
                self.feed(f'drop {item}', to_print=to_print)
                inv.pop(i)
                break

            place, description, items, inv, doors = get_state(msg, to_print, autoloot=False)

            for door in doors:
                if directions[door] != last_move:
                    break

            self.feed(door, to_print=to_print)
            msg = self._get_msg(to_print=to_print)
            if 'You can\'t go that way' not in msg and  'Alert' not in msg:
                return re.match(r'(\d+)', msg).group(1)

            if not inv:
                for item in all_inv:
                    self.feed(f'take {item}')
                    inv.append(item)

                # print('picked all items')
                # print(f'probes left {probes}')
                continue

            probes -= 1

            # print(f'step {step}')
            step += 1

    def feed(self, value, to_print=None):
        _to_print = self._get_to_print(to_print)

        if _to_print:
            print('>>>', end='')

        for x in str(value):
            super().feed(ord(x), _to_print)

        super().feed(self._new_line)


def part1(*args, **kwargs):
    return Cryostasis(*args, **kwargs).run()


def test(test_num):
    _inp = '''
        ....#
        #..#.
        #..##
        ..#..
        #....
    '''
    test_inp = [val.strip() for val in _inp.split('\n') if val.strip()]
    if test_num == 1:
        res = Cryostasis(test_inp).get_biodiversity_rating_of_repeating_layout()
        assert res == 2129920, 'test{} failed!: {}'.format(test_num, res)
    else:
        raise NotImplementedError(f'unknown test_num = {test_num}')

    return 'test{} ok'.format(test_num)


if __name__ == '__main__':
    for res in (
        # test(1),
        part1(), # 536904736
    ):
        print(res)
