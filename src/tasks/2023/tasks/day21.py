"""
--- Day 21: Step Counter ---
https://adventofcode.com/2023/day/21
"""
import collections
import dataclasses

from src.utils.input_formatters import cast_2d_list_elements
from src.utils.position_search_problem import PositionSearchProblem
from src.utils.test_and_run import run, test

STARTING_POSITION = "S"
GARDEN_PLOTS = "."
ROCKS = "#"


@dataclasses.dataclass
class State:
    maze: list[list[int]]
    pos: tuple[int, int]
    path: list[tuple[int, int]]
    infinite: bool
    step: int = 0

    def __post_init__(self):
        self.x, self.y = self.pos
        self.height = len(self.maze)
        self.width = len(self.maze[0])

    def serialize(self):
        return self.pos, self.step

    def __repr__(self):
        return f"{self.__class__.__qualname__}(start={self.pos}, step={self.step}, path={self.path})"

    @staticmethod
    def _get_directions():
        return [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

    def _fit(self, x, y):
        x = x % self.width
        y = y % self.height
        return x, y

    def _is_valid_move(self, xy):
        x, y = xy
        if not self.infinite:
            if not 0 <= x < self.width:
                return False
            if not 0 <= y < self.height:
                return False
        else:
            x, y = self._fit(x, y)

        return self.maze[x][y] == 0

    def get_successors(self):
        for dx, dy in self._get_directions():
            x = self.x + dx
            y = self.y + dy
            pos = (x, y)

            if self._is_valid_move(pos):
                yield State(self.maze, pos, self.path + [(dx, dy)], infinite=self.infinite, step=self.step + 1)


class Problem(PositionSearchProblem):
    def isGoalState(self, state: State):
        return state.step == self.goal

    def getSuccessors(self, state):
        yield from state.get_successors()


class StepCounter:
    def __init__(self, inp, steps, infinite=False):
        self.maze, self.start = self._parse_input(inp)
        self.steps = steps
        self.infinite = infinite

        self.start_state = State(
            self.maze, self.start, path=[], step=0, infinite=self.infinite
        )

    @staticmethod
    def _parse_input(inp):
        maze = cast_2d_list_elements(inp, type_=str)
        start = None
        for i, line in enumerate(maze):
            for j, elm in enumerate(line):
                if elm == ROCKS:
                    maze[i][j] = 1
                else:
                    maze[i][j] = 0
                    if elm == STARTING_POSITION:
                        start = (i, j)
        return maze, start

    def is_garden(self, xy):
        x, y = self.start_state._fit(*xy)
        return self.maze[x][y] == 0

    def _bfs(self):
        start_state = self.start_state
        problem = Problem(start_state, goal=self.steps)

        start_state = problem.getStartState()
        start_successors = problem.getSuccessors(start_state)
        fringe = collections.deque()
        for state in start_successors:
            fringe.appendleft(state)

        _current_steps = 0

        gardens = set([])
        visited = set([])
        min_step_for_visit = {}
        step2gardens = collections.defaultdict(set)
        while fringe:
            state = fringe.popleft()
            pos, step = state.serialize()
            # if pos not in visited:
            #     visited.add(pos)
            #     min_step_for_visit[pos] = step
            visited.add(state.serialize())

            if state.step > _current_steps:
                _current_steps = state.step
                print(f"step {_current_steps}")

            if self.is_garden(state.pos):
                step2gardens[state.step].add(state.serialize())

            if problem.isGoalState(state):
                if self.is_garden(state.pos):
                    gardens.add(state.pos)
                continue

            successors = problem.getSuccessors(state)
            for state in successors:
                if state.serialize() in visited:
                    continue
                # pos, step = state.serialize()
                # if pos in visited:
                #     if step == min_step_for_visit[pos]:
                #
                #
                #     if step < min_step_for_visit[pos]:
                #         min_step_for_visit[pos] = step
                #     # else:
                #     #     continue
                #     # min_step_for_visit[pos] = min(step, min_step_for_visit[pos])
                #     # continue
                fringe.appendleft(state)

        print([len(step2gardens[i]) for i in range(self.steps)])
        return len(gardens)

    def get_number_of_reachable_garden_plots(self):
        return self._bfs()


def step_counter(inp, steps):
    return StepCounter(inp, steps).get_number_of_reachable_garden_plots()


def step_counter2(*args, **kwargs):
    return StepCounter(*args, infinite=True, **kwargs).get_number_of_reachable_garden_plots()


if __name__ == "__main__":
    # test(step_counter, steps=1, expected=2)
    # test(step_counter, steps=2, expected=4)
    # test(step_counter, steps=3, expected=6)
    # test(step_counter, steps=6, expected=16)
    # assert run(step_counter, steps=64) == 3578

    # test(step_counter2, steps=50, expected=1594)
    test(step_counter2, steps=100, expected=6536)
    test(step_counter2, steps=500, expected=167004)
    # test(step_counter2, steps=1000, expected=668697)
    # test(step_counter2, steps=5000, expected=16733044)

    # approximation to step=500 is
    # 16735007.8625024
    # compared to
    # 16733044
    # run(step_counter, steps=26501365)
