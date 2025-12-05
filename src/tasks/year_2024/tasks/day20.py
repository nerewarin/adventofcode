"""
--- Day 20: Race Condition ---
https://adventofcode.com/2024/day/20
"""

import logging
from collections.abc import Generator
from typing import Any, TypeVar, cast

from src.utils.directions import OrthogonalDirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.logger import get_logger, get_message_only_logger
from src.utils.maze import draw_maze, parse_maze
from src.utils.pathfinding import PriorityQueue, astar, manhattan_heuristic
from src.utils.position import Position2D, set_value_by_position
from src.utils.position_search_problem import OrthogonalPositionState, PositionSearchProblem
from src.utils.test_and_run import test

T = TypeVar("T")

_logger = get_logger()
_grid_logger = get_message_only_logger()

WALL = "#"
SPACE = "."
START = "S"
END = "E"
PATH_MEMBER = "O"
CHEAT_START = "1"
CHEAT_END = "2"


class RaceConditionState(OrthogonalPositionState):
    def __init__(self, *args, cheat_made_at: Position2D | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cheat_made_at = cheat_made_at
        self.fast_forwarded_from = None

    def get_successors(self) -> Generator[tuple["RaceConditionState", OrthogonalDirectionEnum, int]]:
        """
        Extends base functionality allowing one cheat
        """
        prior_action = self.get_last_action()
        for yx, direction in self._directions:
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue

            cheat_made_at = self.cheat_made_at
            if self._is_wall(pos):
                if cheat_made_at is None:
                    cheat_made_at = pos
                else:
                    continue

            new_path = self.path + [pos]
            actions = [direction]
            new_actions = self.actions + actions
            cost = self.get_cost_of_actions(actions)

            state = self.__class__(
                self.inp,
                pos,
                self.step + 1,
                new_path,
                new_actions,
                cost=self.cost + cost,
                wall_symbol=self.wall_symbol,
                cheat_made_at=cheat_made_at,
            )
            action = direction

            yield state, action, cost


class LimitedPositionSearchProblem(PositionSearchProblem):
    """
    Extends position search with cost limit
    """

    def __init__(
        self,
        *args,
        path_costs_cache=None,
        cost_threshold=None,
        initial_final_state: RaceConditionState | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert cost_threshold, (
            "branch with no limit is not implemented. maybe just use simple PositionSearchProblem for that?"
        )
        self.cost_threshold = cost_threshold
        assert initial_final_state
        self.initial_final_state = cast(RaceConditionState, initial_final_state)
        self.path_costs_cache = path_costs_cache or {}
        self.considered_cheats: set[Position2D] = set()

    def is_goal_state(self, state):
        return state.pos == self.goal

    def get_successors(self, state: RaceConditionState) -> Generator[tuple[RaceConditionState, Any, Any]]:
        """Must return new state, last action and its cost"""
        res = []
        for new_state, action, cost in state.get_successors():
            pos = new_state.pos
            _logger.debug(
                f"Considering move from {state.pos} ({state._get(pos)}) {action} to {new_state.pos} ({new_state._get(pos)}). cheat_at={new_state.cheat_made_at}. actions={[str(a) for a in new_state.actions]}"
            )
            # TODO idk what to choose. if I choose 1st - algo cant see jump to 11, 2 as cheat down since it already considered 11, 2 cheating right
            #  but if I do 2nd line, algo is super slow
            # if new_state.cheat_made_at and pos not in self.considered_successors_post_cheat:

            # TODO !!!!!!!!!!!!!!!!!! FOR REAL WE NEED JUST TO TRACK CHEATS WE ALREADY MADE! JUST NOT TO REPEAT THEM!
            if new_state.cheat_made_at:
                if new_state.cheat_made_at == pos:
                    # but what if we considered this tile from another cheat?
                    if pos not in self.considered_cheats:
                        self.considered_cheats.add(pos)
                    else:
                        continue
                        # todo = "do something not to duplicate search from this point again"

                if pos in self.path_costs_cache:
                    new_state.fast_forwarded_from = new_state.cost

                    # TODO move to fast_forward method
                    cached_cost = self.path_costs_cache[pos]
                    new_state.cost += cached_cost
                    # start_common_path_index = new_final_state.cost - len(new_final_state.path) + 1
                    common_initial_path = self.initial_final_state.path[-cached_cost:]
                    new_state.path += common_initial_path
                    new_state.actions += self.initial_final_state.actions[-cached_cost:]
                    new_state.step += cached_cost
                    new_state.symbol = self.initial_final_state.symbol
                    new_state.x = self.initial_final_state.x
                    new_state.y = self.initial_final_state.y

                    _logger.debug(
                        f"Cache hit at at {pos} ({new_state.cheat_made_at=}), add cost {cached_cost}, final cost {new_state.cost}"
                    )
                    if new_state.cost > self.cost_threshold:
                        continue

                res.append((new_state, action, cost))
            elif new_state.cheat_made_at is None:
                res.append((new_state, action, cost))

        # if not res:
        # _logger.debug(f"No succ at {state.pos} ({state.cheat_made_at=})")
        # state.draw()

        yield from res


def limited_search(problem: LimitedPositionSearchProblem, fringe, add_to_fringe_fn) -> RaceConditionState | None:
    """generic search rework"""
    closed = set()
    start = (problem.get_start_state(), 0, [])  # (state, cost, actions)
    add_to_fringe_fn(fringe, start, 0)

    while not fringe.isEmpty():
        (state, cost, actions) = fringe.pop()

        if problem.is_goal_state(state):
            return state

        # STATE MUST BE HASHABLE BY POSITION!
        if state not in closed:
            closed.add(state)

            for child_node, child_action, child_cost in problem.get_successors(state):
                new_cost = cost + child_cost
                new_actions = actions + [child_action]
                new_element = (child_node, new_cost, new_actions)
                #
                # if child_node.is_final:
                #     # TODO not sure..
                #     # and now update cache tp optimize?
                #     # cheat_made_at = child_node.cheat_made_at
                #     return child_node

                add_to_fringe_fn(fringe, new_element, new_cost)

    return None


def count_cheap_paths_with_astar(problem, heuristic):
    """Like astar but
    1. does not add state to fringe if cost reaches threshold
    2. consider
    """
    fringe = PriorityQueue()
    res = 0

    def add_to_fringe_fn(fringe: PriorityQueue, element, cost):
        (state, cost_, actions) = element
        new_cost = cost + heuristic(state, problem)
        assert cost_ == cost, f"{cost_=} != {cost=}"
        if new_cost > problem.cost_threshold:
            return None
        fringe.push(element, new_cost)

    solutions = []
    while new_final_state := limited_search(problem, fringe, add_to_fringe_fn):
        res += 1
        _logger.debug(
            f"New solution #{res} found with cost={new_final_state.cost}, is_final={new_final_state.fast_forwarded_from}"
        )

        if new_final_state.fast_forwarded_from:
            fast_forward_index = new_final_state.fast_forwarded_from
            unique_path = new_final_state.path[:fast_forward_index]
        else:
            fast_forward_index = 0
            unique_path = new_final_state.path
            # # fast_forward from problem.initial_final_state
            # start_common_path_index = new_final_state.cost - len(new_final_state.path) + 1
            # common_initial_path = problem.initial_final_state
            # continue  # since we do not extend its real path - we cant rely on it

        # TODO add all pos-->cost_to_goal after cheat to cache!
        for steps_to_goal_left, pos in enumerate(reversed(unique_path)):
            if pos == new_final_state.cheat_made_at:
                break

            if pos in problem.path_costs_cache:
                assert steps_to_goal_left == problem.path_costs_cache[pos]
            else:
                problem.path_costs_cache[pos] = steps_to_goal_left

        solutions.append(new_final_state)

    # TODO del
    edge_solutions = [s for s in solutions if s.cost == problem.cost_threshold]
    grid = problem.inp
    for i, sol in enumerate(edge_solutions):
        sym = chr(ord("a") + i)
        set_value_by_position(sol.cheat_made_at, sym, grid)
    _logger.debug(f"showing {len(edge_solutions)} solutions with cost = max allowed cost = {problem.cost_threshold})")
    draw_maze(grid)
    # ok now we see the problem is in not cutting (11, 2) and (15, 2). inspect them with debugger...
    # ...ok now I found that the problem is filtering by "considered_successors_post_cheat" (jump to cheat end was made by another cheat priorly but didnt finished maybe). wrong and slow with it
    # and INFINITELY slow without it on run. need to fix

    return res


class RaceCondition:
    def __init__(self, maze: list[list[str]], start: Position2D, goal: Position2D, task_num: int, target_savings: int):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.task_num = task_num
        self.target_savings = target_savings

        self._width = len(self.maze[0])
        self._height = len(self.maze)
        self._heuristic = manhattan_heuristic

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data={self.maze}. task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num, target_savings):
        return cls(*parse_maze(inp), task_num, target_savings)

    @staticmethod
    def _get(pos: Position2D, grid: list[list[int]]) -> int:
        x, y = pos
        return grid[y][x]

    def _show_grid(self, grid: list[list[str]] | None = None, level=logging.DEBUG) -> None:
        if grid is None:
            grid = self.maze
        draw_maze(grid)

    def _get_problem[T](self, state_cls, problem_cls: type[T], grid, **kw) -> T:
        state = state_cls(grid, self.start, 0)
        return problem_cls(state=state, goal=self.goal, inp=grid, **kw)

    def _copy_grid(self) -> list[list[str]]:
        return [lst.copy() for lst in self.maze]

    def _initial_run(self):
        problem = self._get_problem(OrthogonalPositionState, PositionSearchProblem, self.maze)
        initial_res = astar(problem, self._heuristic)
        if not initial_res:
            _logger.error("Path finding failed")
            return None

        _initial_final_state, *_ = initial_res
        return _initial_final_state

    def solve(self) -> int | None:
        _logger.debug("Initial grid:")
        self._show_grid()

        _initial_final_state = self._initial_run()

        min_actions_with_no_cheat = len(_initial_final_state.actions)
        grid_copy = self._copy_grid()
        for pos in _initial_final_state.path[1:-1]:  # leave start and end as is
            set_value_by_position(pos, PATH_MEMBER, grid_copy)

        _logger.debug(f"Best path for initial problem of cost {min_actions_with_no_cheat}:")
        _initial_final_state.draw()

        if self.task_num != 1:
            raise NotImplementedError(f"task_num = {self.task_num} is not implemented")

        # from initial run:
        # 1. use length of the best path with no cheats to compute cost_threshold
        target_savings = self.target_savings
        if min_actions_with_no_cheat <= target_savings:
            raise ValueError(f"{min_actions_with_no_cheat=} <= {target_savings=}!")
        cost_threshold = min_actions_with_no_cheat - target_savings
        # 2. cache path length from every initial path to goal not to compute it again
        # we can then update this cache fot new points found AFTER the cheat
        path_costs_cache = {}
        for i, pos in enumerate(_initial_final_state.path):  # leave start and end as is
            path_costs_cache[pos] = min_actions_with_no_cheat - i

        problem = self._get_problem(
            RaceConditionState,
            LimitedPositionSearchProblem,
            self.maze,
            path_costs_cache=path_costs_cache,
            cost_threshold=cost_threshold,
            initial_final_state=_initial_final_state,
        )
        res = count_cheap_paths_with_astar(problem, self._heuristic)
        return res


def task(inp: list[str], task_num: int | None = 1, target_savings: int | None = 100) -> int:
    return RaceCondition.from_multiline_input(inp, task_num, target_savings).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


class Tests:
    @classmethod
    def test1(cls):
        final_state_for_initial_run = RaceCondition.from_multiline_input(
            [
                [x for x in line.strip()]
                for line in """
                    ###############
                    #...#...#.....#
                    #.#.#.#.#.###.#
                    #S#...#.#.#...#
                    #######.#.#.###
                    #######.#.#...#
                    #######.#.###.#
                    ###..E#...#...#
                    ###.#######.###
                    #...###...#...#
                    #.#####.#.###.#
                    #.#...#.#.#...#
                    #.#.#.#.#.#.###
                    #...#...#...###
                    ###############
                """.split("\n")
                if line
            ],
            1,
            1,
        )._initial_run()

        assert len(final_state_for_initial_run.actions) == 84


if __name__ == "__main__":
    Tests.test1()
    test(task1, 1, target_savings=64)
    test(task1, 2, target_savings=40)  # +1
    test(task1, 3, target_savings=38)  # +1
    test(task1, 4, target_savings=36)  # +1
    test(task1, 5, target_savings=20)  # +1
    test(task1, 8, target_savings=12)  # +3
    test(task1, 10, target_savings=10)  # +2
    test(task1, 14, target_savings=8)  # +4
    test(task1, 16, target_savings=6)  # +2
    test(task1, 30, target_savings=4)  # +14
    test(task1, 44, target_savings=2)  # +14
    # run(task1)
