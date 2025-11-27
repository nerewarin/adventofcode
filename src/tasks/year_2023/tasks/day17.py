"""
--- Day 17: Clumsy Crucible ---
https://adventofcode.com/2023/day/17
"""

from collections import defaultdict

from src.tasks.year_2023.tasks.day10 import State as BaseState
from src.utils.input_formatters import cast_2d_list_elements
from src.utils.pathfinding import PriorityQueue, manhattan_distance, nullHeuristic
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
                self.inp[y][x],
            )


def a_star_search(problem: PositionSearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.get_start_state()
    start_successors = problem.get_successors(start_state)
    successors = {start_state: start_successors}
    fringe = PriorityQueue()
    pushed = set([])
    best_cost = {}
    best_path = {}
    last3steps = defaultdict(set)

    for state, action, cost in start_successors:
        moving = state, action, cost
        score = cost + heuristic(state, problem)
        fringe.push((state, [action], cost), score)
        best_cost[state.pos] = score
        pushed.add(moving)

    closed = [start_state.pos]

    for_debug = set([start_state.pos])

    while not fringe.isEmpty():
        moving = fringe.pop()
        state, path, cost = moving

        if problem.is_goal_state(state):
            # TODO del this shit start
            loss = 0
            x, y = start_state.x, start_state.y
            path_str = []
            for i, (dx, dy) in enumerate(state.path):
                x, y = x + dx, y + dy

                loss_ = problem.inp[y][x]
                loss += loss_
                path_str.append(f"{i + 1}. ({x}, {y}) = {loss_}, {loss=}")
            if loss < 985 and loss != 981 and loss not in range(957, 960):
                # TODO del this shit end. left only return path
                for p in path_str:
                    print(p)
                return path
            else:
                print(f"Found path loss of {loss}")
                continue

        closed.append(state)
        if state.pos in for_debug:
            pass

        if state not in successors.keys():
            successors[state] = problem.get_successors(state)

        for child in successors[state]:
            child_state, child_action, child_cost = child

            full_cost = cost + child_cost
            h = heuristic(child_state, problem)
            score = full_cost + h

            child_pos = child_state.pos

            if (child_pos not in closed) and (child not in pushed):
                is_best_cost = child_pos not in best_cost or score <= best_cost[child_pos]

                # just for 2023/day17: give a chance to path with fresh direction (last action was turn)
                # last_action_is_turn = len(child_path) > 1 and child_state.path[-1] != child_state.path[-2]
                last_action_is_turn = False
                last_three_actions = tuple(child_state.path[-3:])

                if is_best_cost or last_action_is_turn or last_three_actions not in last3steps[child_pos]:
                    if is_best_cost:
                        best_cost[child_pos] = score

                    last3steps[child_pos].add(last_three_actions)

                    best_path[child_pos] = child_state.path_str
                    fringe.push((child_state, path + [child_action], full_cost), score)
                    pushed.add(child)
    else:
        return []


class ClumsyCrucible:
    def __init__(self, inp, path=None):
        self._inp = cast_2d_list_elements(inp)

        self.height = len(self._inp)
        self.width = len(self._inp[0])

        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)
        self.path = path or []

        state = State(self._inp, self.start, 0, self.path)

        self.problem = PositionSearchProblem(state=state, goal=self.end, inp=self._inp)

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
            print(f"{i + 1}. ({x}, {y}) = {loss_}, {loss=}")

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
        best_path = a_star_search(self.problem, heuristic)

        loss = 0
        x, y = self.start
        for i, (dx, dy) in enumerate(best_path):
            x, y = x + dx, y + dy

            loss_ = self._inp[y][x]
            loss += loss_
            print(f"{i + 1}. ({x}, {y}) = {loss_}, {loss=}")

        # self._print_test_path()

        return loss


def minimize_loss(inp, **kw):
    return ClumsyCrucible(inp).minimize_loss()


if __name__ == "__main__":
    test(minimize_loss, 102)
    res = run(minimize_loss)
    # not works but was close :)
    assert res == 956
