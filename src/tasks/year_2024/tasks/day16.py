"""
--- Day 16: Reindeer Maze ---
https://adventofcode.com/2024/day/16

The idea is to use same A-star algorythm as for src/tasks/year_2023/tasks/day17.py

"""

from collections.abc import Generator

from src.utils.directions_orthogonal import DIRECTIONS, DirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.logger import get_logger
from src.utils.pathfinding import a_star_search, manhattan_heuristic
from src.utils.position import Position2D
from src.utils.position_search_problem import BaseState, PositionSearchProblem
from src.utils.test_and_run import run

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

    def __str__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"(pos={self.pos}, step={self.step}, symbol={self.symbol}, last_action={self.get_last_action()})"
        )

    @property
    def pos(self) -> Position2D:
        return Position2D(self.x, self.y)

    def _get(self, pos: Position2D) -> str:
        x, y = pos
        return self.inp[y][x]

    def _is_wall(self, yx: Position2D) -> bool:
        return self._get(yx) == WALL

    def get_successors(self) -> Generator[tuple[BaseState, DirectionEnum, int]]:
        prior_action = self.get_last_action()
        for yx, direction in DIRECTIONS.items():
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue
            if self._is_wall(pos):
                continue

            new_path = self.path + [pos]
            new_actions = self.actions + [direction]

            state = self.__class__(self.inp, pos, self.step + 1, new_path, new_actions)
            action = state.get_last_action()

            cost = 1
            if prior_action is not None and action != prior_action:
                cost += 1000

            yield state, action, cost

    def get_last_action(self) -> DirectionEnum | None:
        if not self.actions:
            return None
        return self.actions[-1]


class ReindeerMazeSearchProblem(PositionSearchProblem): ...


class ReindeerMazeTask:
    def __init__(self, inp, task_num, path=None):
        self.maze, self.start, self.end = self._parse_input(inp)
        self._task_num = task_num

        self.height = len(self.maze)
        self.width = len(self.maze[0])

        self.path = path or []

        # The Reindeer start on the Start Tile (marked S) facing East and need to reach the End Tile (marked E)
        starting_rotation = DirectionEnum.right
        state = State(self.maze, self.start, 0, self.path, actions=[starting_rotation])

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
        *_, score = a_star_search(self.problem, manhattan_heuristic)
        return score


def task(inp: list[str], task_num: int = 1) -> int:
    return ReindeerMazeTask(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2)


if __name__ == "__main__":
    # test(task1, 7036)
    # test(task1, 11048, test_part=2)
    run(task1)
