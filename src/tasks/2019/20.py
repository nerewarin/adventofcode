"""
--- Day 20: Donut Maze ---

https://adventofcode.com/2019/day/20

"""

import collections
import itertools

import _tools


class Tile:
    free = '.'
    wall = '#'
    entry_portal = 'A'
    exit_portal = 'Z'
    end = object()


class NoSolution(Exception):
    pass


class DonutMaze:
    def __init__(self, part_num, inp=None, is_test=False, to_print=False, recursive=False):
        if inp is None:
            with open(_tools.get_initiator_fname()) as f:
                inp = _parse_input(f.read())
        self._part_num = part_num
        self._is_test = is_test
        self.inp = inp
        self.to_print = to_print
        self._portal2exit_pos = {}
        self._maze, self._portal_parts = self._parse_maze_and_portals()
        self._start = self._get_start()
        self._recursive = recursive
        self._maze_width = max(len(line) for line in self.inp)  # maze width is a max width of any line
        self._maze_height = len(self.inp)

    def _parse_maze_and_portals(self):
        pos2symbol = {}
        portal_parts = collections.defaultdict(list)
        for y, line in enumerate(self.inp):
            for x, symbol in enumerate(line):
                pos = x, y
                if self._is_portal(symbol):
                    portal_parts[symbol].append(pos)

                pos2symbol[pos] = symbol

        return pos2symbol, portal_parts

    def _get_start(self):
        entry_portal_poses = self._portal_parts[Tile.entry_portal]
        for first, second in itertools.permutations(entry_portal_poses, 2):
            if first[0] == second[0]:
                common_x = first[0]
                assert abs(first[1] - second[1]) == 1
                centre_y = (first[1] + second[1]) / 2
                candidates = (
                    (common_x, centre_y - 1.5),
                    (common_x, centre_y + 1.5),
                )
                break
            elif first[1] == second[1]:
                assert first[1] == second[1]
                common_y = first[1]
                assert abs(first[0] - second[0]) == 1
                centre_x = (first[0] + second[0]) / 2
                candidates = (
                    (centre_x - 1.5, common_y),
                    (centre_x + 1.5, common_y),
                )
                break
            else:
                continue
        else:
            raise ValueError(f'no adjacent entry_portal_poses! {entry_portal_poses}')

        for candidate in candidates:
            candidate = tuple(int(part) for part in candidate)
            if self._is_free(self._maze.get(candidate)):
                return candidate
        else:
            raise ValueError()

    @staticmethod
    def _is_portal(symbol):
        return symbol.isalpha() and symbol.isupper()

    @staticmethod
    def _is_wall(symbol):
        return symbol == Tile.wall

    @staticmethod
    def _is_free(symbol):
        return symbol == Tile.free

    def _get_adjacent_coordinates(self, vertex):
        for coords in (
            (vertex[0], vertex[1] - 1),
            (vertex[0], vertex[1] + 1),
            (vertex[0] - 1, vertex[1]),
            (vertex[0] + 1, vertex[1]),
        ):
            if coords in self._maze:
                yield coords

    @staticmethod
    def _get_portal_key(portal_title, first_portal_part_pos):
        return portal_title, first_portal_part_pos

    @staticmethod
    def _get_portal_title(first_portal_symbol, first_portal_part_pos, second_portal_symbol, second_portal_part_pos):
        # sort portal symbols in a readable order
        if first_portal_part_pos[0] < second_portal_part_pos[0]:
            return first_portal_symbol + second_portal_symbol
        elif first_portal_part_pos[1] < second_portal_part_pos[1]:
            return first_portal_symbol + second_portal_symbol
        return second_portal_symbol + first_portal_symbol

    def _get_portal_exit(self, first_portal_symbol, first_portal_part_pos, second_portal_symbol, second_portal_part_pos, floor):
        portal_title = self._get_portal_title(first_portal_symbol, first_portal_part_pos, second_portal_symbol, second_portal_part_pos)

        if portal_title == Tile.entry_portal * 2:
            return None
        if portal_title == Tile.exit_portal * 2:
            if floor:
                return None
            # raise RuntimeError('win? add one level and commit answer?')
            return Tile.end, floor, Tile.exit_portal * 2

        next_floor = self._get_next_floor(first_portal_part_pos, floor, portal_title)
        if next_floor < 0:
            return None

        portal_key = self._get_portal_key(portal_title, first_portal_part_pos)
        if portal_key in self._portal2exit_pos:
            exit_pos = self._portal2exit_pos[portal_key]
            return self._return_portal_exit(exit_pos, next_floor, portal_title, portal_key)

        # find pos of second portal part symbol
        symbol_poses = []
        _symbol_poses = [pos for pos in self._portal_parts[second_portal_symbol]]
        for pos in _symbol_poses:
            for exit_pos_candidate in self._get_adjacent_coordinates(pos):
                if exit_pos_candidate in (first_portal_part_pos, second_portal_part_pos):
                    continue
                if self._maze[exit_pos_candidate] == first_portal_symbol:
                    symbol_poses.append(exit_pos_candidate)
        if len(symbol_poses) > 2:
            raise RuntimeError('prog error')

        # find exit
        for exit_portal_part1 in symbol_poses:
            exit_portal_part2 = None
            for exit_pos_candidate in self._get_adjacent_coordinates(exit_portal_part1):
                symbol = self._maze[exit_pos_candidate]

                if self._is_free(symbol):
                    return self._return_portal_exit(exit_pos_candidate, next_floor, portal_title, portal_key)

                elif self._is_portal(symbol):
                    exit_portal_part2 = exit_pos_candidate

            for exit_pos_candidate in self._get_adjacent_coordinates(exit_portal_part2):
                symbol = self._maze.get(exit_pos_candidate)
                if self._is_free(symbol):
                    return self._return_portal_exit(exit_pos_candidate, next_floor, portal_title, portal_key)
            else:
                raise ValueError()
        else:
            raise NoSolution()

    def _return_portal_exit(self, exit_pos, next_floor, portal_title, portal_key):
        if portal_key not in self._portal2exit_pos:
            self._portal2exit_pos[portal_key] = exit_pos
        elif self._portal2exit_pos[portal_key] != exit_pos:
            raise ValueError('bad caching!')

        return exit_pos, next_floor, portal_title

    def _get_next_floor(self, first_portal_part_pos, floor, portal_title):
        x, y = first_portal_part_pos
        if not self._recursive:
            shift = 0
        elif 2 < x < self._maze_width - 3 and 2 < y < self._maze_height - 3:
            shift = 1
        else:
            shift = -1
        # shift = 0
        return floor + shift

    def _get_free_adjacent_nodes(self, vertex, floor):
        x0, y0 = vertex
        for adjacent_node in self._get_adjacent_coordinates(vertex):
            x, y = adjacent_node
            symbol = self._maze.get(adjacent_node)
            if not symbol:
                continue

            elif self._is_wall(symbol):
                continue

            elif self._is_free(symbol):
                yield adjacent_node, floor, None

            elif self._is_portal(symbol):
                second_portal_part_pos = (x0 + (x - x0) * 2, (y0 + (y - y0) * 2))
                second_portal_symbol = self._maze[second_portal_part_pos]
                portal_exit = self._get_portal_exit(symbol, adjacent_node, second_portal_symbol, second_portal_part_pos, floor)
                if portal_exit:
                    yield portal_exit

    def _is_complete(self, vertex):
        return vertex == Tile.end

    @staticmethod
    def _is_portal_in_path(portal_title, expected_floor, expected_level, path):
        for perm in itertools.permutations(portal_title, 2):
            title = ''.join(perm)
            for step in path:
                vertex, floor, portal, level = step
                if portal == title and floor == expected_floor and (level == expected_level if expected_level else True):
                    return True
        return False

    @staticmethod
    def _get_path_through_portals(path):
        return [step for step in path if step[2]]

    def _custom_check(self, floor, path):
        # if abs(floor) > 10:
        #     return False

        if self._is_test and self._part_num == 2:
            tst_steps = (
                ('XF', 1, 16),
                ('CK', 2, 27),
                ('ZH', 3, 42),
                ('WB', 4, 53),
                ('IC', 5, 64),
                ('RF', 6, 75),
                ('NM', 7, 84),
                ('LP', 8, 97),
                ('FD', 9, 122),
                ('XQ', 10, 131),
                ('WB', 9, 136),
                ('ZH', 8, 147),
                ('CK', 7, 162),
                ('XF', 6, 173),
                ('OA', 5, 188),
                ('CJ', 4, 197),
                ('RE', 3, 206),
                ('IC', 4, 211),
                ('RF', 5, 222),
                ('NM', 6, 231),
                ('LP', 7, 244),
                ('FD', 8, 269),
                ('XQ', 9, 278),
                ('WB', 8, 283),
                ('ZH', 7, 294),
                ('CK', 6, 309),
                ('XF', 5, 320),
                ('OA', 4, 335),
                ('CJ', 3, 344),
            )
        else:
            return True

        if not path:
            # nth to check yet
            return True

        for i, (expected_portal, expected_floor, expected_level) in enumerate(tst_steps):
            if len(path) < i + 1:
                # nth to check yet
                return True

            step = [path[i]]
            if isinstance(expected_floor, str):
                expected_floor = expected_level
                expected_level = None
            if not self._is_portal_in_path(expected_portal, expected_floor, expected_level, step):
                return False

        if len(path) > len(tst_steps):
            breakpoint = None
        return True

    def get_shortest_path(self):
        start_level = 0
        start_floor = 0
        path = []
        queue = collections.deque([(self._start, start_level, start_floor, path)])
        seen = {
            (start_floor, self._start): start_level
        }
        visit_order = [(start_level, start_floor, self._start)]

        top_paths = []

        while queue:
            vertex, level, floor, path = queue.popleft()

            if self._is_complete(vertex):
                # # lopotkov-style
                _floor = 0
                for step in path:
                    vertex, next_floor, portal_title, level = step
                    if not portal_title:
                        continue
                    print((portal_title, 'inner' if next_floor < _floor else 'outer', next_floor))
                    _floor = next_floor
                print('***'*10)

                # for analyze in sublime
                prev_was_portal = False
                for step in path:
                    vertex, next_floor, portal_title, level = step
                    _sublime_cursor = f':{vertex[1] + 5}:{vertex[0] + 1}'
                    if not portal_title:
                        if prev_was_portal:
                            print(_sublime_cursor)
                            prev_was_portal = False
                            print()
                        continue
                    prev_was_portal = True
                    print(_sublime_cursor)
                    print(', '.join((str(x) for x in (portal_title, next_floor, level))))
                print('==='*10)
                return level

            # if not self._custom_check(floor, self._get_path_through_portals(path)):
            #     continue

            for xy, next_floor, portal_title in self._get_free_adjacent_nodes(vertex, floor):
                _path = list(path)
                if (next_floor, xy) in seen:
                    continue

                new_level = level + 1

                seen[(next_floor, xy)] = new_level

                visit_order.append((level, next_floor, vertex))

                _path.append((vertex, next_floor, portal_title, level))

                queue.append(
                    (xy, new_level, next_floor, _path)
                )

        raise NoSolution()


def _parse_input(inp):
    min_col_idx = float('inf')
    lines = [val for val in inp.split('\n') if val.strip()]
    for line in lines:
        for col_idx, symbol in enumerate(line):
            if symbol != ' ' and col_idx < min_col_idx:
                min_col_idx = col_idx

    return [[symbol for symbol in line[min_col_idx:]] for line in lines]


def test(test_num, recursive=False):
    if test_num == 1:
        _inp = '''
                     A           
                     A           
              #######.#########  
              #######.........#  
              #######.#######.#  
              #######.#######.#  
              #######.#######.#  
              #####  B    ###.#  
            BC...##  C    ###.#  
              ##.##       ###.#  
              ##...DE  F  ###.#  
              #####    G  ###.#  
              #########.#####.#  
            DE..#######...###.#  
              #.#########.###.#  
            FG..#########.....#  
              ###########.#####  
                         Z       
                         Z       
            '''
        if recursive:
            expected = 26
        else:
            expected = 23
    elif test_num == 2:
        _inp = '''
             Z L X W       C                 
             Z P Q B       K                 
  ###########.#.#.#.#######.###############  
  #...#.......#.#.......#.#.......#.#.#...#  
  ###.#.#.#.#.#.#.#.###.#.#.#######.#.#.###  
  #.#...#.#.#...#.#.#...#...#...#.#.......#  
  #.###.#######.###.###.#.###.###.#.#######  
  #...#.......#.#...#...#.............#...#  
  #.#########.#######.#.#######.#######.###  
  #...#.#    F       R I       Z    #.#.#.#  
  #.###.#    D       E C       H    #.#.#.#  
  #.#...#                           #...#.#  
  #.###.#                           #.###.#  
  #.#....OA                       WB..#.#..ZH
  #.###.#                           #.#.#.#  
CJ......#                           #.....#  
  #######                           #######  
  #.#....CK                         #......IC
  #.###.#                           #.###.#  
  #.....#                           #...#.#  
  ###.###                           #.#.#.#  
XF....#.#                         RF..#.#.#  
  #####.#                           #######  
  #......CJ                       NM..#...#  
  ###.#.#                           #.###.#  
RE....#.#                           #......RF
  ###.###        X   X       L      #.#.#.#  
  #.....#        F   Q       P      #.#.#.#  
  ###.###########.###.#######.#########.###  
  #.....#...#.....#.......#...#.....#.#...#  
  #####.#.###.#######.#######.###.###.#.#.#  
  #.......#.......#.#.#.#.#...#...#...#.#.#  
  #####.###.#####.#.#.#.#.###.###.#.###.###  
  #.......#.....#.#...#...............#...#  
  #############.#.#.###.###################  
               A O F   N                     
               A A D   M                     
               '''
        if recursive:
            expected = 396
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(f'unknown test_num = {test_num}')

    inp = _parse_input(_inp)
    res = DonutMaze(test_num, inp, is_test=True, to_print=True, recursive=recursive).get_shortest_path()
    assert res == expected, 'test{} failed!: {}'.format(test_num, res)
    return 'test{} {}ok'.format(test_num, '(recursice) ' if recursive else '')


def run(part_num, *args, **kwargs):
    if part_num == 1:
        res = DonutMaze(part_num, *args, **kwargs).get_shortest_path()
        expected = 498
    elif part_num == 2:
        res = DonutMaze(part_num, *args, recursive=True, **kwargs).get_shortest_path()
        expected = 5564
    else:
        raise NotImplementedError(f'unknown part_num = {part_num}')

    if expected != 'unknown':
        assert res == expected,  f'part{part_num} failed!: res = {res} but {expected} expected!'

    return 'run{} = {}'.format(part_num, res)


if __name__ == '__main__':
    for _res in (
        # test(1),
        # run(1),
        # test(1, recursive=True),
        # test(2, recursive=True),
        run(2),
    ):
        print(_res)
