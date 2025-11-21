
"""
--- Day 24: Planet of Discord ---

https://adventofcode.com/2019/day/24

"""

import collections
import itertools

import _tools


class PlanetOfDiscord:
    _size = 5

    def __init__(self, inp=None):
        _inp = inp or _tools.get_puzzle_input(scalar_type=str, multiline=True)
        self._layouts_history = set()
        self._lvl2inp = {
            0: tuple(tuple(1 if val == '#' else 0 for val in row) for row in _inp)
        }
        self.nulls_layout = tuple([tuple([0] * self._size) for x in range(self._size)])
        self.minute = 0

    def _get_layout(self, level=0):
        return self._lvl2inp[level]

    @staticmethod
    def _get_value(layout, x, y):
        if y < 0 or y >= len(layout):
            return 0
        row = layout[y]
        if x < 0 or x >= len(row):
            return 0
        return row[x]

    def count_adjacent_bugs(self, layout, x, y, level=0):
        res = sum((
            self._get_value(layout, x + 1, y),
            self._get_value(layout, x - 1, y),
            self._get_value(layout, x, y + 1),
            self._get_value(layout, x, y - 1),
        ))

        if len(self._lvl2inp) == 1:
            # part 1 - no levels
            return res

        adj_sum = 0
        centre = self._size // 2
        max_idx = self._size - 1

        if x == centre and y == centre:
            return 0

        # adjacent by border
        layout_level_away = self._lvl2inp.get(level - 1, self.nulls_layout)
        if x in (0, max_idx):
            adjacent_x = centre + (1 if x else -1)
            adjacent_y = centre
            adj_sum += self._get_value(layout_level_away, adjacent_x, adjacent_y)
            """
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |X| | | | |     |
                 |     |-+-+-+-+-|     |
                 |     |X| | | | |     |
                 |     |-+-+-+-+-|     |
                 |  +  |X| |?| | |     |
                 |     |-+-+-+-+-|     |
                 |     |X| | | | |     |
                 |     |-+-+-+-+-|     |
                 |     |X| | | | |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            """

        if y in (0, max_idx):
            breakpoint = 0
            adjacent_x = centre
            adjacent_y = centre + (1 if y else -1)
            adj_sum += self._get_value(layout_level_away, adjacent_x, adjacent_y)
            """
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |    +    |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |y|y|y|y|y|     |
                 |     |-+-+-+-+-|     |
                 |     | | | | | |     |
                 |     |-+-+-+-+-|     |
                 |     | | |?| | |     |
                 |     |-+-+-+-+-|     |
                 |     | | | | | |     |
                 |     |-+-+-+-+-|     |
                 |     | | | | | |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            -----+-----+---------+-----+-----
                 |     |         |     |
                 |     |         |     |
                 |     |         |     |
            """


        # adjacent by centre
        layout_level_into = self._lvl2inp.get(level + 1, self.nulls_layout)
        if x == centre and y in (centre - 1, centre + 1):
            for adjacent_x in range(self._size):
                adjacent_y = max_idx if y == centre + 1 else 0
                value = self._get_value(layout_level_into, adjacent_x, adjacent_y)
                adj_sum += value
                """
                     |     |         |     |
                     |     |    +    |     |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |  +  |    X    |  +  |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |+|+|+|+|+|     |
                     |     |-+-+-+-+-|     |
                     |     | | | | | |     |
                     |     |-+-+-+-+-|     |
                     |     | | | | | |     |
                     |     |-+-+-+-+-|     |
                     |     | | | | | |     |
                     |     |-+-+-+-+-|     |
                     |     |+|+|+|+|+|     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |  +  |    X    |  +  |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |     |    +    |     |
                     |     |         |     |
                """

        if y == centre and x in (centre - 1, centre + 1):
            breakpoint = 0
            for adjacent_y in range(self._size):
                max_x = (self._size - 1)
                adjacent_x = max_x if x == centre + 1 else 0
                value = self._get_value(layout_level_into, adjacent_x, adjacent_y)
                adj_sum += value
                """
                     |     |         |     |
                     |     |         |     |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |     |         |     |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |+| | | |+|     |
                     |     |-+-+-+-+-|     |
                     |     |+| | | |+|     |
                     |     |-+-+-+-+-|     |
                     |  y  |+| | | |+|  y  |
                     |     |-+-+-+-+-|     |
                     |     |+| | | |+|     |
                     |     |-+-+-+-+-|     |
                     |     |+| | | |+|     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |     |         |     |
                     |     |         |     |
                -----+-----+---------+-----+-----
                     |     |         |     |
                     |     |         |     |
                     |     |         |     |
                """

        # print('res', res, 'adj_sum', adj_sum)
        return res + adj_sum

    def get_first_repeated_layout(self, level=0):
        layout = self._get_layout(level)
        while layout not in self._layouts_history:
            self._layouts_history.add(layout)
            # print(len(self.layouts))

            new_list = []
            for y in range(self._size):
                new_row = []
                for x in range(self._size):
                    adjacent_bugs = self.count_adjacent_bugs(layout, x, y)
                    current_val = layout[y][x]

                    if current_val:
                        new_val = adjacent_bugs == 1
                    else:
                        new_val = adjacent_bugs in (1, 2)

                    new_row.append(int(new_val))

                new_list.append(tuple(new_row))

            layout = tuple(new_list)
            self._lvl2inp[level] = layout

        return self._lvl2inp[level]

    def get_biodiversity_rating(self, layout):
        rating = 0
        worth = 1
        for i, val in enumerate(itertools.chain.from_iterable(layout)):
            if val:
                rating += worth
            worth *= 2
        return rating

    def get_biodiversity_rating_of_repeating_layout(self):
        layout = self.get_first_repeated_layout()
        return self.get_biodiversity_rating(layout)

    def _draw(self):
        print(f'\n=== MINUTE {self.minute} ===\n')
        for level, layout in self._lvl2inp.items():
            # DRAW
            print('level', level)
            for row in layout:
                print(row)
            # END DRAW

    def simulate_recursively(self, minutes):
        min_l, max_l = 0, 0
        for minute in range(minutes):
            self.minute = minute
            # consider we add two new layers every odd step (first, 3th, ...)
            if not minute % 2:
                min_l -= 1
                max_l += 1
                self._lvl2inp[min_l] = self.nulls_layout
                self._lvl2inp[max_l] = self.nulls_layout

            self._draw()

            new_lvl2layout = {}
            for level, layout in self._lvl2inp.items():
                new_list = []
                for y in range(self._size):
                    new_row = []
                    for x in range(self._size):
                        adjacent_bugs = self.count_adjacent_bugs(layout, x, y, level)
                        current_val = layout[y][x]

                        if current_val:
                            new_val = adjacent_bugs == 1
                        else:
                            new_val = adjacent_bugs in (1, 2)

                        new_row.append(int(new_val))

                    new_list.append(tuple(new_row))

                new_lvl2layout[level] = tuple(new_list)

            self._lvl2inp = new_lvl2layout

        self.minute += 1
        self._draw()

        return None

    def count_all_bugs(self):
        bugs = 0
        for level, layout in self._lvl2inp.items():
            bugs_on_lvl = sum(itertools.chain.from_iterable(layout))
            bugs += bugs_on_lvl

        return bugs


def part1(*args, **kwargs):
    return PlanetOfDiscord(*args, **kwargs).get_biodiversity_rating_of_repeating_layout()


def part2():
    solver = PlanetOfDiscord()
    solver.simulate_recursively(200)
    return solver.count_all_bugs()


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
        res = PlanetOfDiscord(test_inp).get_biodiversity_rating_of_repeating_layout()
        assert res == 2129920, 'test{} failed!: {}'.format(test_num, res)
    elif test_num == 2:
        res = part1()
        assert res == 25719471, 'test{} failed!: {}'.format(test_num, res)
    elif test_num == 3:
        solver = PlanetOfDiscord(test_inp)
        solver.simulate_recursively(10)
        res = solver.count_all_bugs()
        assert res == 99, 'test{} failed!: {}'.format(test_num, res)
    else:
        raise NotImplementedError(f'unknown test_num = {test_num}')

    return 'test{} ok'.format(test_num)


if __name__ == '__main__':
    for res in (
        # test(1),
        # test(2),
        # test(3),  # 108, 91
        # part1(),
        part2(),  # 101 is low, 1748 and 2042 are wrong
    ):
        print(res)
