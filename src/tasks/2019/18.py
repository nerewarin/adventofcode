"""
--- Day 18: Many-Worlds Interpretation ---

https://adventofcode.com/2019/day/18

"""

import re
import collections

import _tools


_regexp = re.compile(r'- ([\w ]+)')


class Tile:
    free = '.'
    wall = '#'
    agent = '@'


class NoSolution(Exception):
    pass


class MazeWithKeysAndDoors:
    def __init__(self, inp=None, idx=None, to_print=False):
        self._maze, self._start_pos, self._doors, self._keys = self._parse_input_to_maze(inp)
        self._collected_keys = tuple()
        self._to_print = to_print
        self.idx = idx

    def _parse_input_to_maze(self, _inp):
        inp = _inp or _tools.get_puzzle_input(scalar_type=str, delimeter='', multiline=True)
        maze = []
        start_pos = None
        doors = {}
        keys = {}
        for row_num, line in enumerate(inp):
            row = []
            for col_num, symbol in enumerate(line):
                pos = (col_num, row_num)
                if self._is_door(symbol):
                    doors[symbol] = pos
                elif self._is_key(symbol):
                    keys[symbol] = pos
                elif self._is_agent(symbol):
                    start_pos = pos

                row.append(symbol)

            maze.append(row)

        return maze, start_pos, doors, keys

    @staticmethod
    def _is_wall(symbol):
        return symbol == Tile.wall

    @staticmethod
    def _is_free(symbol):
        return symbol == Tile.free

    @staticmethod
    def _is_key(symbol):
        return symbol.isalpha() and symbol.islower()

    @staticmethod
    def _is_door(symbol):
        if not isinstance(symbol, str):  return False
        return symbol.isalpha() and symbol.isupper()

    @staticmethod
    def _is_agent(symbol):
        return symbol == Tile.agent

    def _draw_maze(self, vertex=None, keys=tuple()):
        if not self._to_print:
            return

        if vertex is None:
            vertex = self._start_pos

        print()

        for row_num, line in enumerate(self._maze):
            for col_num, symbol in enumerate(line):
                if self._is_door(symbol) and self._door_is_open(symbol, keys):
                    symbol = Tile.free

                elif self._is_key(symbol) and symbol in keys:
                    symbol = Tile.free

                elif self._is_agent(symbol) and vertex != self._start_pos:
                    symbol = Tile.free

                elif (col_num, row_num) == vertex:
                    symbol = Tile.agent

                print(symbol, end='')

            print()

    def _get_symbol(self, x, y):
        if y < 0 or y >= len(self._maze):
            return None
        row = self._maze[y]
        if x < 0 or x >= len(row):
            return None
        return row[x]

    @staticmethod
    def _door_is_open(door, keys_collected):
        return door.lower() in keys_collected

    def update_collected_keys(self, another_keys):
        self._collected_keys = tuple(set(self._collected_keys).union(set(another_keys)))
        return self._collected_keys

    def _get_adjacent_nodes(self, vertex, keys):
        for x, y in (
            (vertex[0], vertex[1] - 1),
            (vertex[0], vertex[1] + 1),
            (vertex[0] - 1, vertex[1]),
            (vertex[0] + 1, vertex[1]),
        ):
            symbol = self._get_symbol(x, y)
            if symbol is None:
                continue

            elif self._is_wall(symbol):
                continue

            elif self._is_free(symbol) or self._is_agent(symbol):
                yield x, y, keys

            elif self._is_door(symbol):
                if self._door_is_open(symbol, keys):
                    yield x, y, keys
                else:
                    yield x, y, symbol

            elif self._is_key(symbol):
                _keys = set(keys)
                yield x, y, tuple(_keys.union({symbol}))

    def is_complete(self, keys_collected):
        return not set(self._keys) - set(keys_collected)

    def get_shortest_path_of_collecting_all_keys(self):
        self._draw_maze()

        start_level = 0
        queue = collections.deque([(self._start_pos, start_level, self._collected_keys)])
        seen = {
            (self._start_pos, tuple(self._collected_keys)): start_level
        }
        visit_order = []
        doors_locked = {}

        while queue:
            vertex, level, keys_collected = queue.popleft()
            visit_order.append((vertex, level, keys_collected))

            self._draw_maze(vertex, keys_collected)

            if self.is_complete(keys_collected):
                # all keys collected
                return vertex, level, keys_collected, doors_locked

            for x, y, door_or_keys in self._get_adjacent_nodes(vertex, keys_collected):
                if self._is_door(door_or_keys):
                    doors_locked[door_or_keys] = (x, y)
                    continue

                new_keys_collected = door_or_keys
                node = (x, y)
                if (node, new_keys_collected) in seen:
                    continue

                new_level = level + 1

                seen[(node, new_keys_collected)] = new_level

                queue.append(
                    (node, new_level, tuple(new_keys_collected))
                )

        # raise NoSolution()

        # now, instead of raise NoSolution, we should return maximum keys and a minimun level where they were reached
        s = []
        for state in sorted(seen, key=lambda key_val: (-len(key_val[1]), key_val[1])):
            node, keys_collected = state
            level = seen[state]
            s.append({state: level})
            return node, level, keys_collected, doors_locked
        return


class MazeWithKeysAndDoorsFourAgents(MazeWithKeysAndDoors):
    def __init__(self, inp=None, **kwargs):
        super().__init__(inp, **kwargs)

        center_x, center_y = len(self._maze[0]) // 2, len(self._maze) // 2

        if not inp:
            self._maze[center_y][center_x] = Tile.wall
            self._maze[center_y + 1][center_x] = Tile.wall
            self._maze[center_y - 1][center_x] = Tile.wall
            self._maze[center_y][center_x + 1] = Tile.wall
            self._maze[center_y][center_x - 1] = Tile.wall
            self._maze[center_y + 1][center_x + 1] = Tile.agent
            self._maze[center_y + 1][center_x - 1] = Tile.agent
            self._maze[center_y - 1][center_x + 1] = Tile.agent
            self._maze[center_y - 1][center_x - 1] = Tile.agent

        north_part = self._maze[:center_y]
        south_part = self._maze[center_y + 1:]
        assert len(south_part) == len(north_part), f'len(north_part) ({len(north_part)}) != len(south_path) ({south_path})'
        self._solvers = [
            MazeWithKeysAndDoors(maze, idx)
            for idx, maze in enumerate([
                [row[:center_x] for row in north_part],
                [row[center_x + 1:] for row in north_part],
                [row[:center_x] for row in south_part],
                [row[center_x + 1:] for row in south_part],
            ])
        ]
        max_y = None
        max_x = None
        for s in self._solvers:
            if max_y and len(s._maze) != max_y:
                raise ValueError('bad map parsing')
            max_y = len(s._maze)

            row_wight = len(s._maze[0])
            if max_x and row_wight != max_x:
                raise ValueError(f'bad row_wight {row_wight}')
            max_x = row_wight

    def get_shortest_path_of_collecting_all_keys(self):
        steps = []
        solved = []
        while len(self._collected_keys) != len(self._keys):
            for idx, solver in enumerate(self._solvers):
                if idx in solved:
                    continue

                solver._collected_keys = tuple(set(solver._collected_keys) - set(solver._keys))
                try:
                    vertex, level, keys_collected, doors_locked = solver.get_shortest_path_of_collecting_all_keys()
                    self.update_collected_keys(keys_collected)
                    if solver.is_complete(keys_collected):
                        steps.append(level)
                        solved.append(idx)
                except NoSolution as e:
                    print(idx, e)
                    continue

        return sum(steps)

    def update_collected_keys(self, another_keys):
        super().update_collected_keys(another_keys)

        for solver in self._solvers:
            solver.update_collected_keys(self._collected_keys)

        return self._collected_keys


def part1(*args, **kwargs):
    vertex, level, keys_collected, doors_locked = MazeWithKeysAndDoors(*args, **kwargs).get_shortest_path_of_collecting_all_keys()
    return level


def part2(*args, **kwargs):
    res = MazeWithKeysAndDoorsFourAgents(*args, **kwargs).get_shortest_path_of_collecting_all_keys()
    if res <= 1086:
        raise ValueError(f'answer {res} is too low!')

    return res


def test(test_num):
    if test_num == 1:
        _inp = '''
            #########
            #b.A.@.a#
            #########
        '''
        expected = 8
    elif test_num == 2:
        _inp = '''
            ########################
            #f.D.E.e.C.b.A.@.a.B.c.#
            ######################.#
            #d.....................#
            ########################
        '''
        expected = 86
    elif test_num == 3:
        _inp = '''
            ########################
            #...............b.C.D.f#
            #.######################
            #.....@.a.B.c.d.A.e.F.g#
            ########################
        '''
        expected = 132
    else:
        raise NotImplementedError(f'unknown test_num = {test_num}')

    inp = [val.strip() for val in _inp.split('\n') if val.strip()]
    # inp = None
    vertex, level, keys_collected, doors_locked = MazeWithKeysAndDoors(inp, to_print=True).get_shortest_path_of_collecting_all_keys()
    assert level == expected, 'test{} failed!: {}'.format(test_num, res)
    return 'test{} ok'.format(test_num)


def test2(test_num):
    if test_num == 1:
        _inp = '''
            #######
            #a.#Cd#
            ##@#@##
            #######
            ##@#@##
            #cB#Ab#
            #######
        '''
        expected = 8
    elif test_num == 2:
        _inp = '''
            ###############
            #d.ABC.#.....a#
            ######@#@######
            ###############
            ######@#@######
            #b.....#.....c#
            ###############
        '''
        expected = 24
    elif test_num == 3:
        _inp = '''
            #############
            #DcBa.#.GhKl#
            #.###@#@#I###
            #e#######j###
            #############
            ###d#######k#
            ###C#@#@###J#
            #fEbA.#.FgHi#
            #############
        '''
        expected = 32
    else:
        raise NotImplementedError(f'unknown test_num = {test_num}')

    inp = [val.strip() for val in _inp.split('\n') if val.strip()]
    res = MazeWithKeysAndDoorsFourAgents(inp, to_print=True).get_shortest_path_of_collecting_all_keys()
    assert res == expected, 'test2{} failed!: {}'.format(test_num, res)
    return 'test2{} ok'.format(test_num)


if __name__ == '__main__':
    for res in (
        # test(1),
        # test(2),
        # test(3),
        # part1(),
        # test2(1),
        # test2(2),
        # test2(3),
        part2(),
    ):
        print(res)
