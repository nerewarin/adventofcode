"""
--- Day 17: Clumsy Crucible ---
https://adventofcode.com/2023/day/17
"""

from tasks.day10 import State as BaseState
from src.utils.input_formatters import cast_2d_list_elements
from src.utils.pathfinding import manhattan_distance, aStarSearch
from src.utils.position_search_problem import PositionSearchProblem
from src.utils.test_and_run import run, test

# def print(*_):
#     pass

class State(BaseState):
    directions = {
        (0, 1): "v",
        (0, -1): "^",
        (1, 0): ">",
        (-1, 0): "<",
    }

    def __init__(self, *args, visited=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited = visited or set([])

    def __hash__(self):
        return self.height * self.x + self.y

    def __repr__(self):
        return (
            # f"{self.__class__.__qualname__}(pos={self.pos}, symbol={self.symbol}, step={self.step})"
            f"{self.__class__.__qualname__}(pos={self.pos}, step={self.step}, path={self.path_str})"
        )

    @property
    def path_str(self):
        return "".join(self.directions[action] for action in self.path)

    def show_path(self):
        x, y = (0, 0)

        # copy
        inp = [list(row) for row in self.inp]
        for i, (dx, dy) in enumerate(self.path):
            x, y = x + dx, y + dy

            # loss_ = self._inp[y][x]
            # loss += loss_
            # print(f"{i+1}. ({x}, {y}) = {loss_}, {loss=}")

            inp[y][x] = self.directions[(dx, dy)]

        print(f"Showing path to {self}")
        for line in inp:
            print("".join(str(x) for x in line))
        print()

    def get_successors(self):
        for dx, dy in self._get_directions():
            if self.path and (-dx, -dy) == self.path[-1]:
                # skip going back
                continue

            x = self.x + dx
            y = self.y + dy
            pos = (x, y)

            if pos in self.visited:
                continue

            if not 0 <= x <= self.width - 1:
                continue

            if not 0 <= y <= self.height - 1:
                continue

            if self.path and len(self.path) >= 3:
                last_3_steps = set(self.path[-3:])
                last_3_steps.add((dx, dy))

                if len(last_3_steps) == 1:
                    continue

            visited = set(self.visited)
            visited.add(pos)
            yield (
                State(self.inp, pos, self.step + 1, self.path + [(dx, dy)], visited=visited),
                (dx, dy),
                # TODO check x y in right order
                self.inp[y][x]
            )


class Day17PositionSearchProblem(PositionSearchProblem):
    def getSuccessors(self, state):
        yield from state.get_successors()


class ClumsyCrucible:
    def __init__(self, inp, path=None):
        self._inp = cast_2d_list_elements(inp)

        self.height = len(self._inp)
        self.width = len(self._inp[0])

        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)
        self.path = path or []

        state = State(self._inp, self.start, 0, self.path)

        self.problem = Day17PositionSearchProblem(state=state, goal=self.end, inp=self._inp)

    def _print_test_path(self):
        # test path
        r = (1, 0)
        l = (-1, 0)
        t = (0, -1)
        b = (0, 1)

        best_path = [
            r,
            r,
            b,
            r,
            r,
            r,
            t,
            r,
            r,
            r,
            b,
            b,
            r,
            r,
            b,
            b,
            r,
            b,
            b,
            b,
            r,
            b,
            b,
            b,
            l,
            b,
            b,
            r,
        ]
        print("correct path")
        loss = 0
        x, y = self.start
        for i, (dx, dy) in enumerate(best_path):
            x, y = x + dx, y + dy

            loss_ = self._inp[y][x]
            loss += loss_
            print(f"{i+1}. ({x}, {y}) = {loss_}, {loss=}")

    def minimize_loss(self):
        def heuristic(state, problem):
            # return manhattan_distance(state.pos, problem.goal) #  * 10
            child_path = state.path
            repeats = 0
            last_move = None
            for i in reversed(child_path):
                if last_move is None:
                    last_move = i
                    continue

                if i != last_move:
                    break
                else:
                    last_move = i
                    repeats += 1

            return manhattan_distance(state.pos, problem.goal) + repeats * 2

        # return astar(world, self.start, end=self.end, max_blocks_in_a_single_direction=3)
        best_path = aStarSearch(self.problem, heuristic)

        loss = 0
        x, y = self.start
        for i, (dx, dy) in enumerate(best_path):
            x, y = x + dx, y + dy

            loss_ = self._inp[y][x]
            loss += loss_
            print(f"{i+1}. ({x}, {y}) = {loss_}, {loss=}")

        # self._print_test_path()

        return loss


def minimize_loss(inp, **kw):
    return ClumsyCrucible(inp).minimize_loss()


if __name__ == "__main__":
    test(minimize_loss, 102)
    res = run(minimize_loss)
    # not works but was close :)
    assert res == 956
