"""
--- Day 16: Reindeer Maze ---
https://adventofcode.com/2024/day/16

The idea is to use same A-star algorythm as for src/tasks/year_2023/tasks/day17.py

"""

from collections.abc import Generator

from src.utils.directions import ORTHOGONAL_DIRECTIONS, OrthogonalDirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.logger import get_logger
from src.utils.pathfinding import PriorityQueue, astar, manhattan_heuristic
from src.utils.position import Position2D
from src.utils.position_search_problem import BaseState, PositionSearchProblem
from src.utils.test_and_run import run, test

_logger = get_logger()

START = "S"
END = "E"
WALL = "#"
SPACE = "."


class State1(BaseState):
    def __init__(self, inp: list[list[str]], pos, step=None, path=None, actions=None, cost=0):
        self.inp = inp
        self.x, self.y = pos
        self.symbol = self.inp[self.y][self.x]

        self.step = step or 0
        self.path = path or [pos]
        self.actions = actions or []
        self.cost = cost

    def __str__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"(pos={self.pos}, step={self.step}, symbol={self.symbol}, last_action={self.get_last_action()},"
            f"cost={self.cost})"
        )

    @property
    def pos(self) -> Position2D:
        return Position2D(self.x, self.y)

    def _get(self, pos: Position2D) -> str:
        x, y = pos
        return self.inp[y][x]

    def _is_wall(self, yx: Position2D) -> bool:
        return self._get(yx) == WALL

    @property
    def _directions(self) -> Generator[OrthogonalDirectionEnum]:
        yield from ORTHOGONAL_DIRECTIONS.items()

    def get_successors(self) -> Generator[tuple[BaseState, OrthogonalDirectionEnum, int]]:
        prior_action = self.get_last_action()
        for yx, direction in self._directions:
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue
            if self._is_wall(pos):
                continue

            new_path = self.path + [pos]
            actions = [direction]
            new_actions = self.actions + actions
            cost = self.get_cost_of_actions(actions)

            state = self.__class__(self.inp, pos, self.step + 1, new_path, new_actions, cost=self.cost + cost)
            action = direction

            yield state, action, cost

    def get_cost_of_actions(self, actions: list[OrthogonalDirectionEnum]) -> int:
        """
        Returns the cost of a particular sequence of actions.
        """
        if len(actions) != 1:
            raise NotImplementedError

        action = actions[0]
        prior_action = self.get_last_action()
        cost = 1
        if prior_action is not None and action != prior_action:
            cost += 1000
        return cost

    def get_last_action(self) -> OrthogonalDirectionEnum | None:
        if not self.actions:
            return None
        return self.actions[-1]


class State2(State1):
    def __hash__(self):
        # hash only by position
        return hash((self.pos, self.get_last_action()))

    def __eq__(self, other: "State2") -> bool:
        if not isinstance(other, State2):
            return NotImplemented
        # equality only by position
        return self.pos == other.pos and self.get_last_action() == other.get_last_action()


class ReindeerMazeTask:
    # The Reindeer start on the Start Tile (marked S) facing East and need to reach the End Tile (marked E)
    starting_rotation = OrthogonalDirectionEnum.right

    def __init__(self, inp, task_num, path=None):
        self.maze, self.start, self.end = self._parse_input(inp)
        self._task_num = task_num

        self.height = len(self.maze)
        self.width = len(self.maze[0])

        self.path = path or []
        self._heuristic = manhattan_heuristic

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

    def _get_problem(self, state_cls: type[State1] | type[State2]) -> PositionSearchProblem:
        state = state_cls(self.maze, self.start, 0, self.path, actions=[self.starting_rotation])
        return PositionSearchProblem(state=state, goal=self.end, inp=self.maze)

    def solve(self, best_score=None):
        if self._task_num == 1:
            problem = self._get_problem(State1)
            *_, score = astar(problem, self._heuristic)
        elif self._task_num == 2:
            problem = self._get_problem(State2)
            best_final_states = self._find_all_best_paths_with_a_star(problem, best_score)

            visited = set()
            for state in best_final_states:
                visited.update(set(state.path))
            score = len(visited)
        else:
            raise NotImplementedError
        return score

    def _find_all_best_paths_with_a_star(self, problem: PositionSearchProblem, best_cost=None):
        if best_cost is None:
            best_cost = float("inf")
        fringe = PriorityQueue()

        def add_to_fringe_fn(fringe: PriorityQueue, state, cost):
            est_cost = cost + self._heuristic(state[0], problem)
            fringe.push(state, est_cost)
            return est_cost

        closed = set()
        start = (problem.get_start_state(), 0, [])  # (state, cost, path)
        add_to_fringe_fn(fringe, start, 0)

        best_final_states = []

        while not fringe.isEmpty():
            (state, cost, path) = fringe.pop()

            est_cost = cost + self._heuristic(state, problem)
            if est_cost > best_cost:
                continue

            if problem.is_goal_state(state):
                print(f"Found path of {cost=}: {path}")
                if cost < best_cost:
                    best_cost = cost
                    best_final_states = []

                best_final_states.append(state)
                continue

            # STATE MUST BE HASHABLE BY POSITION!
            # if state in closed:
            #     continue
            # elif state not in closed or cost <= min((state.cost for state in closed if hash(state) == state)):
            closed.add(state)

            for child_node, child_action, child_cost in problem.get_successors(state):
                new_cost = cost + child_cost
                new_path = path + [child_action]
                # est_cost = new_path + self._heuristic(child_node, problem)

                new_state = (child_node, new_cost, new_path)
                add_to_fringe_fn(fringe, new_state, new_cost)

        return best_final_states


def task(inp: list[str], task_num: int = 1, best_score=None) -> int:
    return ReindeerMazeTask(inp, task_num).solve(best_score)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, best_score=None):
    return task(inp, task_num=2, best_score=best_score)


if __name__ == "__main__":
    # test(task1, 7036)
    # test(task1, 11048, test_part=2)
    # run(task1)  # 65436

    test(task2, 45)
    test(task2, 64, test_part=2)
    run(task2, best_score=65436)
