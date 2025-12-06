"""
--- Day 20: Race Condition ---
https://adventofcode.com/2024/day/20
"""

import logging
import re
from collections.abc import Generator
from typing import Any, NamedTuple, TypeVar, cast

from tqdm import tqdm

from src.utils.directions import OrthogonalDirectionEnum, go, is_a_way_back, out_of_borders
from src.utils.logger import get_logger, get_message_only_logger
from src.utils.maze import draw_maze, parse_maze
from src.utils.pathfinding import PriorityQueue, astar, manhattan_heuristic
from src.utils.position import Position2D, set_value_by_position
from src.utils.position_search_problem import OrthogonalPositionState, PositionSearchProblem
from src.utils.test_and_run import run, test

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
        for yx, direction in self.directions:
            if prior_action is not None and is_a_way_back(direction, prior_action):
                continue

            pos = go(direction, self.pos)

            if out_of_borders(*pos, self.inp, is_reversed=False):
                continue

            cheat_made_at = self.cheat_made_at
            if self.is_wall(pos):
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
        for new_state, new_action, step_cost in state.get_successors():
            pos = new_state.pos
            _logger.debug(
                f"Considering move from {state.pos} ({state._get(pos)}) {new_action} to {new_state.pos} ({new_state._get(pos)}). cheat_at={new_state.cheat_made_at}. actions={[str(a) for a in new_state.actions]}"
            )
            if new_state.cheat_made_at:
                if new_state.cheat_made_at == pos:
                    if pos not in self.considered_cheats:
                        self.considered_cheats.add(pos)
                    else:
                        continue

                if pos in self.path_costs_cache:
                    new_state.fast_forwarded_from = new_state.cost

                    # TODO move to fast_forward method
                    cached_cost = self.path_costs_cache[pos]
                    new_state.cost += cached_cost
                    common_initial_path = self.initial_final_state.path[-cached_cost:]
                    new_state.path += common_initial_path
                    new_state.actions += self.initial_final_state.actions[-cached_cost:]
                    new_state.step += cached_cost
                    new_state.symbol = self.initial_final_state.symbol
                    new_state.x = self.initial_final_state.x
                    new_state.y = self.initial_final_state.y
                    step_cost += cached_cost

                    _logger.debug(
                        f"Cache hit at at {pos} ({new_state.cheat_made_at=}), add cost {cached_cost}, final cost {new_state.cost}"
                    )
                    if new_state.cost > self.cost_threshold:
                        continue

                res.append((new_state, new_action, step_cost))
            elif new_state.cheat_made_at is None:
                res.append((new_state, new_action, step_cost))

        yield from res


class Cheat(NamedTuple):
    start: Position2D
    end: Position2D

    @property
    def length(self) -> int:
        return self.start.manhattan_to(self.end)


def limited_search(problem: LimitedPositionSearchProblem, fringe, add_to_fringe_fn) -> RaceConditionState | None:
    """generic search rework"""
    closed = set()
    start_state = problem.get_start_state()
    add_to_fringe_fn(fringe, start_state)

    while not fringe.isEmpty():
        state = fringe.pop()

        if problem.is_goal_state(state):
            return state

        # STATE MUST BE HASHABLE BY POSITION!
        if state not in closed:
            closed.add(state)

            for child_node, child_action, child_step_cost in problem.get_successors(state):
                new_cost = state.cost + child_step_cost

                assert child_node.cost == new_cost

                add_to_fringe_fn(fringe, child_node)

    return None


def count_cheap_paths_with_astar(problem: LimitedPositionSearchProblem, heuristic):
    """Like astar but
    1. does not add state to fringe if cost reaches threshold
    2. consider
    """
    fringe = PriorityQueue()
    res = 0

    def add_to_fringe_fn(fringe: PriorityQueue, state: RaceConditionState):
        estimated_cost = state.cost + heuristic(state, problem)
        if estimated_cost > problem.cost_threshold:
            return None
        fringe.push(state, estimated_cost)

    solutions = []
    while new_final_state := limited_search(problem, fringe, add_to_fringe_fn):
        res += 1
        _logger.info(
            f"New solution #{res} found with cost={new_final_state.cost}, fast_forwarded_from={new_final_state.fast_forwarded_from}"
        )

        if new_final_state.fast_forwarded_from:
            fast_forward_index = new_final_state.fast_forwarded_from
            unique_path = new_final_state.path[:fast_forward_index]
        else:
            fast_forward_index = 0
            unique_path = new_final_state.path

        tail_cost = len(new_final_state.path[fast_forward_index:])

        for steps_to_goal_left_, pos in enumerate(reversed(unique_path)):
            if pos == new_final_state.cheat_made_at:
                break

            steps_to_goal_left = steps_to_goal_left_ + tail_cost
            if pos in problem.path_costs_cache:
                assert steps_to_goal_left == problem.path_costs_cache[pos], (
                    f"steps_to_goal_left != problem.path_costs_cache[pos]: "
                    f"({steps_to_goal_left} != {problem.path_costs_cache[pos]}, {pos=})"
                )
            else:
                problem.path_costs_cache[pos] = steps_to_goal_left

        solutions.append(new_final_state)

        # optimization:
        # start new search from the last point of diverge from original path
        step = 0
        for step, pos in enumerate(new_final_state.path):
            if pos != problem.initial_final_state.path[step]:
                break
        last_common_step = step - 1
        if last_common_step < 0:
            raise RuntimeError("broken search from the last point of diverge from original path")
        start_pos = problem.initial_final_state.path[last_common_step]
        start_actions = list(problem.initial_final_state.actions[:last_common_step])
        start_path = list(problem.initial_final_state.path[:last_common_step])
        assert new_final_state.cheat_made_at not in start_path, f"{new_final_state.cheat_made_at=} in {start_path=}!"
        assert new_final_state.path[:last_common_step] == start_path
        assert new_final_state.actions[:last_common_step] == start_actions

        if start_pos != problem.startState.pos:
            # now fast-forward our problem from start position to this point of diverge
            # TODO further optimization includes iterative fast-forwarding start pos not from initial start but last fast-forward made
            _logger.info(f"Fast forwarding starting position {last_common_step} steps from start pos to {start_pos}")
            problem.startState = RaceConditionState(
                problem.inp,
                start_pos,
                step=last_common_step,
                path=start_path,
                actions=start_actions,
                cost=new_final_state.get_cost_of_actions(start_actions),
                wall_symbol=new_final_state.wall_symbol,
            )

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

    def draw_maze(self, grid: list[list[str]] | None = None, level=logging.DEBUG) -> None:
        if grid is None:
            grid = self.maze
        draw_maze(grid, level=level)

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

    def _check_every_cell_is_in_initial_path(self, initial_final_state: RaceConditionState) -> None:
        path = set(initial_final_state.path)
        for row_ids, row in enumerate(self.maze):
            for col_idx, value in enumerate(row):
                pos = Position2D(col_idx, row_ids)
                if initial_final_state.is_wall(pos):
                    continue
                if pos not in path:
                    raise RuntimeError(f"pos {pos} not in optimal path of initial run!")

    def _count_distinct_cheats(self, initial_final_state: RaceConditionState, strict: bool | None = False) -> int:
        """
        This problem is harder, but now I know:
        1. All spaces in our input exist in initial path (check it with 1st line of implementation below)
        2. Cheat is defined with start and end - implementing Cheat class

        strict mode:
            count only cheats which save EXACT target_savings
        """
        self._check_every_cell_is_in_initial_path(initial_final_state)

        target_savings = self.target_savings

        # for every step of initial path, we can try every cheat up to 20 length.
        # we get to some point. this point is either wall or part of initial path.
        # we know that cost to reach this cell by initial path is just its index in this path, but still let's
        # cache it for O(1) pick explicitly.
        # so we found good cheat if:
        # 1. this cheat is not covered yet
        # cheated = set()
        # def is_new_cheat(cheat: Cheat) -> bool:
        #     return cheat not in cheated
        #
        # # 2. cheat end is a space
        # def check_cheat_end(cheat: Cheat) -> bool:
        #     return not initial_final_state.is_wall(cheat.end)
        #
        # # 3. cheat makes needed amount of savings
        # def is_valuable_cheat(cheat: Cheat) -> bool:
        #     return pos_to_step[cheat.start] + cheat.length <= pos_to_step[cheat.end] + target_savings

        # ...but well, since we know that "All spaces in our input exist in initial path" - we can just consider
        # cheat from every path to some far point on the path
        initial_path = initial_final_state.path
        res = 0
        min_cheat_len = 2
        max_cheat_len = 20
        cheat_starts_to_consider = initial_path[:-target_savings]
        for step, cheat_start in tqdm(
            enumerate(cheat_starts_to_consider),
            desc="considering cheats from every perspective position of original path",
            total=len(cheat_starts_to_consider),
        ):
            # try to reach some far points
            target_shift = min_cheat_len + step + target_savings
            cheat_ends_to_consider = initial_path[target_shift:]
            for i, cheat_end in enumerate(cheat_ends_to_consider):
                cheat = Cheat(cheat_start, cheat_end)
                cheat_length = cheat.length
                if cheat_length > max_cheat_len:
                    continue

                initial_cost = target_shift + i
                cost_with_cheat = step + cheat_length
                savings = initial_cost - cost_with_cheat
                if savings == target_savings or strict is False and savings > target_savings:
                    res += 1

                    if _logger.level <= logging.DEBUG:
                        grid = self._copy_grid()
                        set_value_by_position(cheat.start, "1", grid)
                        set_value_by_position(cheat.end, "2", grid)

                        _logger.debug(
                            f"Found cheat #{res}: {cheat} of len {cheat.length} forking path at step {step} and saving {savings} steps (from {initial_cost} to {cost_with_cheat} to end point) meeting {target_savings=}"
                        )
                        draw_maze(grid)

        return res

    def solve(self, strict: bool | None = False) -> int | None:
        _logger.info("Initial grid:")
        self.draw_maze(level=logging.INFO)

        _initial_final_state = self._initial_run()

        min_actions_with_no_cheat = len(_initial_final_state.actions)
        grid_copy = self._copy_grid()
        for pos in _initial_final_state.path[1:-1]:  # leave start and end as is
            set_value_by_position(pos, PATH_MEMBER, grid_copy)

        _logger.info(f"Best path for initial problem of cost {min_actions_with_no_cheat}:")
        _initial_final_state.draw(level=logging.INFO)

        # from initial run:
        # 1. use length of the best path with no cheats to compute cost_threshold
        target_savings = self.target_savings
        if min_actions_with_no_cheat <= target_savings:
            raise ValueError(f"{min_actions_with_no_cheat=} <= {target_savings=}!")
        cost_threshold = min_actions_with_no_cheat - target_savings
        _logger.info(
            f"Counting paths with unique cheats with {cost_threshold=} (at maximum) following {target_savings=} (at minimum)"
        )

        # 2. cache path length from every initial path to goal not to compute it again
        # we can then update this cache fot new points found AFTER the cheat
        path_costs_cache = {}
        for i, pos in enumerate(_initial_final_state.path):  # leave start and end as is
            path_costs_cache[pos] = min_actions_with_no_cheat - i

        if self.task_num == 2:
            return self._count_distinct_cheats(_initial_final_state, strict)

        assert self.task_num == 1
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


def task(inp: list[str], task_num: int | None = 1, target_savings: int | None = 100, strict=False) -> int:
    return RaceCondition.from_multiline_input(inp, task_num, target_savings).solve(strict)


def task1(inp, **kw):
    # TODO rewrite using task2 approach
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


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
    # Tests.test1()
    # test(task1, 1, target_savings=64)
    # test(task1, 2, target_savings=40)  # +1
    # test(task1, 3, target_savings=38)  # +1
    # test(task1, 4, target_savings=36)  # +1
    # test(task1, 5, target_savings=20)  # +1
    # test(task1, 8, target_savings=12)  # +3
    # test(task1, 10, target_savings=10)  # +2
    # test(task1, 14, target_savings=8)  # +4
    # test(task1, 16, target_savings=6)  # +2
    # test(task1, 30, target_savings=4)  # +14
    # test(task1, 44, target_savings=2)  # +14
    # run(task1)  # 1490

    # test task2
    _rexp = re.compile(r"(\d+).*?(\d+)")
    for test_statement in """
        There are 32 cheats that save 50 picoseconds.
        There are 31 cheats that save 52 picoseconds.
        There are 29 cheats that save 54 picoseconds.
        There are 39 cheats that save 56 picoseconds.
        There are 25 cheats that save 58 picoseconds.
        There are 23 cheats that save 60 picoseconds.
        There are 20 cheats that save 62 picoseconds.
        There are 19 cheats that save 64 picoseconds.
        There are 12 cheats that save 66 picoseconds.
        There are 14 cheats that save 68 picoseconds.
        There are 12 cheats that save 70 picoseconds.
        There are 22 cheats that save 72 picoseconds.
        """.split("\n"):
        if not test_statement:
            continue
        _res = _rexp.search(test_statement)
        if _res:
            _expected_path_with_cheats, _savings = map(int, _res.groups())
            test(task2, _expected_path_with_cheats, target_savings=_savings, strict=True)

    run(task2)  # 1011325
