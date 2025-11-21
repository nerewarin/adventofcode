"""
--- Day 10: Pipe Maze ---
https://adventofcode.com/2023/day/10
"""
import collections

from src.utils.test_and_run import run, test


def sum_2d(tuples_2d):
    return tuple(sum(components) for components in zip(*tuples_2d))


def multiply_2d(pos, coef):
    return tuple([pos[i] * coef for i in range(len(pos))])


class State:
    def __init__(self, inp, pos, step=0, path=None):
        self.inp = inp
        self.x, self.y = self.pos = pos
        self.symbol = self.inp[self.y][self.x]

        self.step = step or 0
        self.path = path or []

        self.height = len(self.inp)
        self.width = len(self.inp[0])

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}(pos={self.pos}, symbol={self.symbol}, step={self.step})"
        )

    def _is_connected(self, pos, start=None):
        x2, y2 = pos
        if not 0 <= x2 < self.width:
            return False
        if not 0 <= y2 < self.height:
            return False

        end_sym = self.inp[y2][x2]
        if end_sym == ".":
            # return self. == 2
            # TODO return self.part == 2
            return False

        if start is None:
            start = self.pos
        x1, y1 = start

        start_sym = self.inp[y1][x1]
        if start_sym == ".":
            return False

        pos_diff = x2 - x1, y2 - y1
        return self._are_pipes_connected(start_sym, end_sym, pos_diff)

    def _are_pipes_connected(self, sym1, sym2, pos_diff):
        left, right, top, bot = self._left, self._right, self._top, self._bot

        possible_connections = {
            "-": {
                "-": [left, right],
                "|": [],
                "F": [left],
                "7": [right],
                "J": [right],
                "L": [left],
            },
            "|": {
                "-": [],
                "|": [top, bot],
                "F": [top],
                "7": [top],
                "J": [bot],
                "L": [bot],
            },
            "F": {
                "-": [right],
                "|": [bot],
                "F": [],
                "7": [right],
                "J": [bot, right],
                "L": [bot],
            },
            "7": {
                "-": [left],
                "|": [bot],
                "F": [left],
                "7": [],
                "J": [bot],
                "L": [bot, left],
            },
            "J": {
                "-": [left],
                "|": [top],
                "F": [left, top],
                "7": [top],
                "J": [],
                "L": [left],
            },
            "L": {
                "-": [right],
                "|": [top],
                "F": [top],
                "7": [top, right],
                "J": [right],
                "L": [],
            }
        }
        if sym1 == "S":
            for sym1 in possible_connections:
                if sym2 in possible_connections[sym1] and pos_diff in possible_connections[sym1][sym2]:
                    return True
            return False

        if sym2 == "S":
            for sym2 in possible_connections[sym1]:
                if pos_diff in possible_connections[sym1][sym2]:
                    return True
            return False

        return sym2 in possible_connections[sym1] and pos_diff in possible_connections[sym1][sym2]

    @property
    def _left(self):
        return -1, 0

    @property
    def _right(self):
        return 1, 0

    @property
    def _top(self):
        return 0, -1

    @property
    def _bot(self):
        return 0, 1

    @property
    def _directions(self):
        return [
            self._left,
            self._right,
            self._top,
            self._bot,
        ]

    @staticmethod
    def _get_directions():
        return [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

    def repr_last_step(self, last_step=None):
        if last_step is None:
            last_step = self.path[-1]

        succ = self
        for k, v in {
            "LEFT": succ._left,
            "RIGHT": succ._right,
            "TOP": succ._top,
            "BOT": succ._bot,
        }.items():
            if last_step == v:
                return k

    def get_successors(self):
        for dx, dy in self._get_directions():
            x = self.x + dx
            y = self.y + dy
            pos = (x, y)

            if self.path:
                last_step = self.path[-1]
                if (last_step[0] * -1, last_step[1] * -1) == (dx, dy):
                    # print(f"dont want go back to {x, y}")
                    continue

            if self._is_connected(pos):
                yield State(self.inp, pos, self.step + 1, self.path + [(dx, dy)])


class PipeMaze:
    def __init__(self, inp, part):
        self._inp = inp
        self._part = part

    def __repr__(self):
        return "\n".join(line for line in self._inp)

    def is_dot(self, xy):
        x, y = xy
        return self._inp[y][x] == "."

    def _get_start_pos(self):
        for y, line in enumerate(self._inp):
            for x, elm in enumerate(line):
                if elm == "S":
                    return x, y
                    # if self._part == 1:
                    #     return x, y
                    # if self._part == 2:
                    #     return x - 1, y
        raise ValueError(f"no start position in {self._inp}")

    def get_steps_to_farthest_position(self, start_pos=None, step=None, path_coordinates=None):
        if not start_pos:
            start_pos = self._get_start_pos()

        all_pipes = {start_pos}
        state = State(self._inp, start_pos, step, path_coordinates)
        fringe = collections.deque([state])

        while fringe:
            state = fringe.popleft()

            for succ in state.get_successors():
                all_pipes.add(succ.pos)
                # print(f"{state} do {succ.repr_last_step()} -> {succ}")
                if succ.symbol == "S":
                    maze = list([["."] * len(row) for row in self._inp])
                    for y, line in enumerate(self._inp):
                        for x, elm in enumerate(line):
                            if (x, y) in all_pipes:
                                v = elm
                                match elm:
                                    case "|":
                                        v = "┃"
                                    case "-":
                                        v = "━"
                                    case "F":
                                        v = "┏"
                                    case "L":
                                        v = "┗"
                                    case "J":
                                        v = "┛"
                                    case "7":
                                        v = "┓"
                                maze[y][x] = v
                            else:
                                assert maze[y][x] == "."
                                maze[y][x] = "x"
                        print("".join(maze[y]))

                    if self._part == 2:
                        x, y = self._get_start_pos()
                        path_coordinates = []
                        res = 0
                        for dx, dy in succ.path:
                            x += dx
                            y += dy
                            path_coordinates.append((x, y))

                        # looks like S is always an angle so we can define inner area as an area inside this angle
                        first_and_last_steps = [succ.path[0], succ.path[-1]]
                        # revert last step to show directions from start point as a vector
                        start_vectors = first_and_last_steps[0], multiply_2d(first_and_last_steps[-1], -1)

                        start_vector = sum_2d(start_vectors)

                        visited = set([])
                        if self.is_dot(start_vector):
                            a = 0
                            # TODO search for dots in all directions until you reach border

                        return res

                    if self._part == 2:
                        res = 0
                        for i, (x, y) in enumerate(path_coordinates):
                            action = succ.path[i]
                            if self._inp[y][x] == ".":
                               res += 1
                        assert x, y == self._get_start_pos()  # make sure we end just where we start
                        return res

                    return succ.step // 2
                else:
                    fringe.append(succ)

        raise ValueError("no solution to get_steps_to_farthest_position")


def pipe_maze(inp, part=1, **kw):
    return PipeMaze(inp, part).get_steps_to_farthest_position()


if __name__ == "__main__":
    # test(pipe_maze, 4)
    # test(pipe_maze, test_part=2, expected=8)
    # run(pipe_maze)
    #
    # # not implemented - visual solving
    # test(pipe_maze, part=2, test_part=3, expected=4)
    # test(pipe_maze, part=2, test_part=4, expected=8)
    # test(pipe_maze, part=2, test_part=5, expected=10)
    assert run(pipe_maze, part=2) == 451


