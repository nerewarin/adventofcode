"""
--- Day 3: Crossed Wires ---

https://adventofcode.com/2019/day/3
"""

import os
import math


class CrossedWires:
    def __init__(self, _map=None):
        if _map is None:
            with open(os.path.join('inputs', '{}.txt'.format(__file__.split('/')[-1].split('.')[0]))) as f:
                line1, line2 = [
                    [x for x in line.strip().split(',')]
                    for line in f.readlines()
                ]
        else:
            line1, line2 = [
                [x for x in line.strip().split(',')]
                for line in _map.split('\n') if line.strip()
            ]

        lines_xy = []
        for line_idx, line in enumerate((line1, line2)):
            elm = (0, 0)
            line_coords = [elm]
            for move in line:
                direction, size = move[0], int(move[1:])
                x, y = elm
                if direction == 'R':
                    x = x + size
                elif direction == 'L':
                    x = x - size
                elif direction == 'U':
                    y = y + size
                elif direction == 'D':
                    y = y - size
                else:
                    raise ValueError('direction', direction)

                elm = x, y
                line_coords.append(elm)
            lines_xy.append(line_coords)

        self.line1, self.line2 = lines_xy

    @staticmethod
    def line_intersection(line1, line2, default=None):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return default

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        # we have not lines but segments so check its on two sectors additionaly
        if x < max(
            min(point[0] for point in line)
            for line in (line1, line2)
        ):
            return default
        if x > min(
            max(point[0] for point in line)
            for line in (line1, line2)
        ):
            return default
        if y > min(
            max(point[1] for point in line)
            for line in (line1, line2)
        ):
            return default
        if y < max(
            min(point[1] for point in line)
            for line in (line1, line2)
        ):
            return default

        return x, y

    @staticmethod
    def manh_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) + abs(y1 - y2)

    def get_cross_nearest_to_centre(self):
        centre = (0.0, 0.0)
        min_dist = float('inf')
        for idx1, point1 in enumerate(self.line1[:-1]):
            section1 = (point1, self.line1[idx1 + 1])

            for idx2, point2 in enumerate(self.line2[:-1]):
                section2 = (point2, self.line2[idx2 + 1])

                intersection = self.line_intersection(section1, section2, default=centre)
                if intersection != centre:
                   distance = self.manh_distance(centre, intersection)

                   if distance < min_dist:
                       min_dist = distance

        return int(min_dist)

    def get_minimum_steps_to_cross(self):
        centre = (0.0, 0.0)
        prev_dist1 = 0
        min_dist = float('inf')
        for idx1, point1 in enumerate(self.line1[:-1]):
            section1 = (point1, self.line1[idx1 + 1])

            prev_dist2 = 0
            for idx2, point2 in enumerate(self.line2[:-1]):
                section2 = (point2, self.line2[idx2 + 1])

                intersection = self.line_intersection(section1, section2, default=centre)
                if intersection != centre:
                    distance1 = self.manh_distance(point1, intersection)
                    distance2 = self.manh_distance(point2, intersection)

                    distance = sum((
                        prev_dist1,
                        prev_dist2,
                        distance1,
                        distance2,
                    ))

                    if distance < min_dist:
                        min_dist = distance

                prev_dist2 += self.manh_distance(*section2)

            prev_dist1 += self.manh_distance(*section1)

        return int(min_dist)


inp = '''
    R75,D30,R83,U83,L12,D49,R71,U7,L72
    U62,R66,U55,R34,D71,R55,D58,R83
'''

inp2 = '''
    R8,U5,L5,D3
    U7,R6,D4,L4
'''


def test1():
    test_num = 1
    res = CrossedWires(inp).get_cross_nearest_to_centre()
    assert res == 159, \
        'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def test2():
    test_num = 2
    res = CrossedWires(inp2).get_minimum_steps_to_cross()
    assert res == 30, \
        'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def test3():
    test_num = 3
    res = CrossedWires(inp).get_minimum_steps_to_cross()
    assert res == 610, \
        'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def part1(*args, **kwargs):
    return CrossedWires(*args, **kwargs).get_cross_nearest_to_centre()


def part2(*args, **kwargs):
    return CrossedWires(*args, **kwargs).get_minimum_steps_to_cross()


if __name__ == '__main__':
    for res in (
        test1(),
        part1(),

        test2(),
        test3(),
        part2(),
    ):
        print(res)
