"""
--- Day 16: Reindeer Maze ---
https://adventofcode.com/2024/day/16

The idea is to use same A-star algorythm as for src/tasks/year_2023/tasks/day17.py

"""

from collections.abc import Iterable

from src.utils.directions_orthogonal import DIRECTIONS, DirectionEnum, go, out_of_borders
from src.utils.logger import get_logger
from src.utils.pathfinding import aStarSearch, manhattan_distance
from src.utils.position import Position2D
from src.utils.position_search_problem import BaseState, PositionSearchProblem
from src.utils.test_and_run import run, test

_logger = get_logger()

START = "S"
END = "E"
WALL = "#"
SPACE = "."


class State(BaseState):
    def __init__(self, inp: list[list[str]], pos, step=None, path=None, actions=None, visited=None):
        self.inp = inp
        self.x, self.y = pos
        self.symbol = self.inp[self.y][self.x]

        self.step = step or 0
        self.path = path or []
        self.actions = actions or []
        self.visited = visited or (set(path) if path is not None else set())

    @property
    def pos(self) -> Position2D:
        return Position2D(self.y, self.x)

    def _get(self, yx) -> str:
        y, x = yx
        return self.inp[y][x]

    def get_successors(self) -> Iterable["State"]:
        for yx, direction in DIRECTIONS.items():
            pos = go(direction, self.pos)

            if out_of_borders(*yx, self.inp):
                continue

            new_path = self.path + [pos]
            new_actions = self.actions + [direction]

            yield self.__class__(self.inp, pos, self.step + 1, new_path, new_actions)

    def copy(self): ...

    def get_last_action(self) -> DirectionEnum:
        return self.actions[-1]


class ReindeerMazeSearchProblem(PositionSearchProblem):
    def get_successors(self, state: State):
        yield from state.get_successors()


class ReindeerMazeTask:
    def __init__(self, inp, task_num, path=None):
        self.maze, self.start, self.end = self._parse_input(inp)
        self._task_num = task_num

        self.height = len(self.maze)
        self.width = len(self.maze[0])

        self.path = path or []

        state = State(self.maze, self.start, 0, self.path)

        self.problem = ReindeerMazeSearchProblem(state=state, goal=self.end, inp=self.maze)

    @staticmethod
    def _parse_input(inp):
        start = None
        end = None
        maze = [[] for _ in inp]
        for row, line in enumerate(inp):
            for col, symbol in enumerate(line):
                maze[row].append(symbol)
                if symbol == START:
                    assert start is None
                    start = Position2D(col, row)
                elif symbol == END:
                    assert end is None
                    end = Position2D(col, row)
        assert start and end
        return maze, start, end

    def solve(self):
        def heuristic(state, problem):
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
        return best_path  # TODO get score


def task(inp: list[str], task_num: int = 1) -> int:
    return ReindeerMazeTask(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 7036)
    test(task1, 11048, test_part=2)
    run(task1)
