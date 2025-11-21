"""
--- Day 10: Monitoring Station ---

https://adventofcode.com/2019/day/10

"""

import os
import math


class MonitoringStation:
    BUSY = 1
    EMPTY = 0
    place2int = {
        '.': EMPTY,
        '#': BUSY,
    }

    def __init__(self, _map=None):
        if _map is None:
            with open(os.path.join('inputs', '{}.txt'.format(__file__.split('/')[-1].split('.')[0]))) as f:
                _map = f.read()
        else:
            _map = '\n'.join((
                line.strip() for line in _map.split('\n')
            ))
        _map = [
            [self.place2int.get(place, self.BUSY) for place in line]
            for line in _map.split('\n') if line
        ]

        self.map = _map

    @staticmethod
    def get_angle(x1, y1, x2, y2):
        # let angle 0 be to the up
        return (math.atan2(y2 - y1, x2 - x1) + math.pi / 2) % (math.pi * 2)

    def get_visible_asteroids_count(self, x1, y1):
        checked_angles = {}
        counter = 0
        for y2, line2 in enumerate(self.map):
            for x2, place2 in enumerate(line2):
                # skip self
                if x1 == x2 and y1 == y2:
                    continue

                if not place2:
                    continue

                angle = self.get_angle(x1, y1, x2, y2)
                if angle in checked_angles:
                    continue

                counter += 1
                checked_angles[angle] = (x2, y2)

        return counter, checked_angles

    def get_visibility_map(self, return_angle2asteroid_map=False):
        visibility_map = []
        angle2asteroid_map = {}
        for y1, line1 in enumerate(self.map):
            visibility_line = []
            for x1, place1 in enumerate(line1):
                # place is empty - save 0 as a num of visible asteroids
                if not place1:
                    visibility_line.append(place1)
                    continue

                # count visible asteroids by finding angles to every visible point
                visible_asteroids, checked_angles = self.get_visible_asteroids_count(x1, y1)
                visibility_line.append(visible_asteroids)
                angle2asteroid_map[x1, y1] = checked_angles

            visibility_map.append(visibility_line)

        if return_angle2asteroid_map:
            return visibility_map, angle2asteroid_map

        return visibility_map

    def find_best_place_and_count(self):
        visibility_map = self.get_visibility_map()
        x, y = (0, 0)
        max_vision = float('-inf')
        for y1, line1 in enumerate(visibility_map):
            for x1, value in enumerate(line1):
                if value > max_vision:
                    x, y = (x1, y1)
                    max_vision = value

        return (x, y), max_vision

    def find_best_place_and_angle2asteroid_map(self):
        visibility_map, angle2asteroid_map = self.get_visibility_map(return_angle2asteroid_map=True)
        x, y = (0, 0)
        max_vision = float('-inf')
        for y1, line1 in enumerate(visibility_map):
            for x1, value in enumerate(line1):
                if value > max_vision:
                    x, y = (x1, y1)
                    max_vision = value

        return (x, y), max_vision, angle2asteroid_map

    def vaporize(self, desirable_vaporized_idx):
        station_pos, *max_vision, angle2asteroid_map = self.find_best_place_and_angle2asteroid_map()
        x, y = None, None
        # for idx in range(desirable_vaporized_idx):
        #     x, y, angle2asteroid_map = self._vaporize(station_pos, angle2asteroid_map)

        # The laser starts by pointing up and always rotates clockwise, vaporizing any asteroid it hits
        angle2asteroids = angle2asteroid_map[station_pos]
        if not angle2asteroids:
            dummy, angle2asteroid_map = self.get_visibility_map(return_angle2asteroid_map=True)
            angle2asteroids = angle2asteroid_map[station_pos]

        sorted_angles = sorted(angle2asteroids)

        idx = 0
        while True:
            for angle in sorted_angles:
                vaporized_x, vaporized_y = angle2asteroids[angle]
                self.map[vaporized_y][vaporized_x] = self.EMPTY
                idx += 1
                if idx == desirable_vaporized_idx:
                    return vaporized_x, vaporized_y

            dummy, angle2asteroid_map = self.get_visibility_map(return_angle2asteroid_map=True)
            angle2asteroids = angle2asteroid_map[station_pos]

            sorted_angles = sorted(angle2asteroids)

            if not sorted_angles:
                raise RuntimeError()


def test1():
    inp = '''
        .#..#
        .....
        #####
        ....#
        ...##
    '''
    res = MonitoringStation(inp).get_visibility_map()
    assert res == [[0, 7, 0, 0, 7], [0, 0, 0, 0, 0], [6, 7, 7, 7, 5], [0, 0, 0, 0, 7], [0, 0, 0, 8, 7]], \
        'test1 failed!: {}'.format(res)
    return 'test1 ok'


def test2():
    inp = '''
        #.........
        ...A......
        ...B..a...
        .EDCG....a
        ..F.c.b...
        .....c....
        ..efd.c.gb
        .......c..
        ....f...c.
        ...e..d..c
    '''
    return 'test2 is not implemented'


def test3():
    inp = '''
        ......#.#.
        #..#.#....
        ..#######.
        .#.#.###..
        .#..#.....
        ..#....#.#
        #..#....#.
        .##.#..###
        ##...#..#.
        .#....####
    '''
    ms = MonitoringStation(inp)
    res = ms.find_best_place_and_count()
    assert res == ((5, 8), 33), 'test3 failed!: {}'.format(res)
    return 'test3 ok'


def test4():
    inp = '''
        #.#...#.#.
        .###....#.
        .#....#...
        ##.#.#.#.#
        ....#.#.#.
        .##..###.#
        ..#...##..
        ..##....##
        ......#...
        .####.###.
    '''
    res = MonitoringStation(inp).find_best_place_and_count()
    assert res == ((1, 2), 35), 'test4 failed!: {}'.format(res)
    return 'test4 ok'


def test5():
    test_num = 5
    inp = '''
        .#..#..###
        ####.###.#
        ....###.#.
        ..###.##.#
        ##.##.#.#.
        ....###..#
        ..#.#..#.#
        #..#.#.###
        .##...##.#
        .....#.#..
    '''
    res = MonitoringStation(inp).find_best_place_and_count()
    assert res == ((6, 3), 41), 'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


inp6 = '''
    .#..##.###...#######
    ##.############..##.
    .#.######.########.#
    .###.#######.####.#.
    #####.##.#.##.###.##
    ..#####..#.#########
    ####################
    #.####....###.#.#.##
    ##.#################
    #####.##.###..####..
    ..######..##.#######
    ####.##.####...##..#
    .#####..#.######.###
    ##...#.##########...
    #.##########.#######
    .####.#.###.###.#.##
    ....##.##.###..#####
    .#.#.###########.###
    #.#.#.#####.####.###
    ###.##.####.##.#..##
'''


def test6():
    test_num = 6
    res = MonitoringStation(inp6).find_best_place_and_count()
    assert res == ((11, 13), 210), 'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def test7():
    test_num = 7
    res = MonitoringStation(inp6).vaporize(200)
    assert res == (8, 2), 'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def part1(*args, **kwargs):
    return MonitoringStation(*args).find_best_place_and_count()


def part2(*args, **kwargs):
    x, y = MonitoringStation(*args).vaporize(200)
    return x * 100 + y


if __name__ == '__main__':
    for res in (
        test1(),
        test2(),
        test3(),
        test4(),
        test5(),
        test6(),
        part1(),
        test7(),
        part2(),
    ):
        print(res)
