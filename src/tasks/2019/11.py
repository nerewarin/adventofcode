"""
--- Day 11: Space Police ---

https://adventofcode.com/2019/day/11

"""

import os
import math
import collections
import operator

from _intcode_computer import IntcodeComputer

BLACK, WHITE = 0, 1
LEFT, RIGHT = 0, 1


class SpacePolice:
    def __init__(self, inp=None, mode=None):
        # key (tuple) : coordinates
        self.map = collections.defaultdict(
            tuple  # value: current_color (int), painted (bool)
        )

        self.computer = IntcodeComputer(inp, mode)

        self.direction = 0  # up

    def _update_direction(self, turn):
        if turn not in (LEFT, RIGHT):
            raise ValueError()
        self.direction = (self.direction + (-1) ** (turn + 1)) % 4
        return self.direction

    def _get_step(self):
        if self.direction == 0:
            return (0, -1)
        if self.direction == 1:
            return (-1, 0)
        if self.direction == 2:
            return (0, 1)
        if self.direction == 3:
            return (1, 0)

    def paint(self):
        """
        The Intcode program will serve as the brain of the robot

        Returns:
            int: number of panels it paints at least once, regardless of color

        """
        pos = (0, 0)
        poses = [pos]
        dirs = [self.direction]
        while True:
            # print(list(self.computer.memory.values()))
            try:
                # provide 0 if the robot is over a black panel or 1 if the robot is over a white panel.
                state = self.map.get(pos)
                # print(pos)


                curr_color = 1 if state and state[0] else 0
                if not state:
                    inp_msg = f'default ({curr_color})'
                else:
                    inp_msg = curr_color

                self.computer.feed(curr_color)
                print(f'input {inp_msg}')
                print()

                # First, it will output a value indicating the color to paint the panel the robot is over:
                # 0 means to paint the panel black, and 1 means to paint the panel white
                color = next(self.computer)
                # print('WHITE' if color else 'BLACK')

                # Second, it will output a value indicating the direction the robot should turn:
                # 0 means it should turn left 90 degrees, and 1 means it should turn right 90 degrees.
                turn = next(self.computer)
                # print('RIGHT' if turn else 'LEFT')

                print(f'on position {pos} got paint {color} turn {turn}')

                self.map[pos] = color, True

                self._update_direction(turn)

                step = self._get_step()
                pos = tuple(map(operator.add, pos, step))
                print(f'move to  {pos}')

                poses += [pos]
                dirs += [self.direction]

            except StopIteration as e:
                return sum(val[1] for val in self.map.values())


def test1():
    inp = '''
        .#..#
        .....
        #####
        ....#
        ...##
    '''
    res = SpacePolice(inp).paint()
    assert res == 6, 'test1 failed!: {}'.format(res)
    return 'test1 ok'


def test7():
    test_num = 7
    res = MonitoringStation(inp6).vaporize(200)
    assert res == (8, 2), 'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def part1(*args, **kwargs):
    return SpacePolice(*args, **kwargs).paint()
#
#
# def part2(*args, **kwargs):
#     x, y = MonitoringStation(*args).vaporize(200)
#     return x * 100 + y


if __name__ == '__main__':
    for res in (
        # test1(),
        # test2(),
        # test3(),
        # test4(),
        # test5(),
        # test6(),
        part1(),
        # test7(),
        # part2(),
    ):
        print(res)
