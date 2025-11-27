"""--- Day 16: Proboscidea Volcanium ---
https://adventofcode.com/2022/day/16
"""

import datetime
import itertools
import logging
import re
from copy import copy
from functools import lru_cache

from src.utils.pathfinding import uniformCostSearch
from src.utils.position_search_problem import PositionSearchProblem
from src.utils.test_and_run import run, test

_REXP = re.compile(r"Valve (\w+) has flow rate=(\d+); tunnels? leads? to valves? (.+)")
PRINTED_STATES = set([])
MODE = None
CACHE = {}


class Valve:
    def __init__(self, name, flow_rate, connected_to):
        self.name = name
        self.flow_rate = flow_rate
        self.connections = {k: connected_to[k] for k in sorted(connected_to)}

    def __repr__(self):
        return f"{self.__class__.__qualname__}(name={self.name}, flow={self.flow_rate}, connected={self.connections})"


@lru_cache
def _parse_puzzle(inp):
    valve_by_name = {}
    for line in inp:
        name, _flow_rate, raw_path = _REXP.match(line).groups()
        valve = Valve(name, int(_flow_rate), {x: 1 for x in raw_path.split(", ")})
        valve_by_name[name] = valve

    for name, valve in valve_by_name.items():
        for c in valve.connections:
            valve_by_name[c].connections[name] = 1

    valves_copy = dict(valve_by_name)
    for valve in valves_copy.values():
        for connected_name in dict(valve.connections):
            cv = valve_by_name.get(connected_name)
            if cv:
                if valve.flow_rate > 0:
                    pass
                else:
                    for name in valve.connections:
                        if name != connected_name:
                            if name in valve_by_name:
                                v = valve.connections[name] + valve.connections[connected_name]
                                if name in cv.connections:
                                    v = min(v, cv.connections[name])
                                cv.connections[name] = v
                            else:
                                excluded_valve = valves_copy[name]
                                for name, val in excluded_valve.connections.items():
                                    _ = 0
                                    cv.connections[name] = val + 1
                    if valve.name == "AA":
                        continue
                    cv.connections.pop(valve.name)
            else:
                continue

        if valve.flow_rate <= 0:
            if valve.name == "AA":
                continue
            del valve_by_name[valve.name]

    # need a connection from every to every
    lst = sorted(valve_by_name)
    for i in range(len(lst)):
        for j in range(len(lst)):
            start_name = lst[i]
            goal_name = lst[j]
            if start_name == goal_name:
                continue
            start_vale = valve_by_name[start_name]
            if goal_name in start_vale.connections:
                continue
            if goal_name == "AA":
                continue
            state = State(valve_by_name, pos=start_name)

            problem = PositionSearchProblem(state, goal_name)
            actions = uniformCostSearch(problem)

            steps = 0
            s_ = start_name
            for action in actions:
                steps += valve_by_name[s_].connections[action]
                s_ = action
            if goal_name in start_vale.connections:
                steps = min(steps, start_vale.connections[goal_name])
            start_vale.connections[goal_name] = steps

    global MODE
    MODE = "prod"
    for name, v in valve_by_name.items():
        v.connections = {k: v for k, v in v.connections.items() if k != "AA"}

    return valve_by_name


TIME_TO_OPEN = 1
TIME_TO_MOVE = 1


class State:
    def __init__(
        self,
        valve_by_name,
        step=0,
        pos="AA",
        fuel=0,
        opened_valves: set[str] = None,
        fps=0,
        path=None,
        step2action=None,
        step2openings=None,
    ):
        self.valve_by_name = valve_by_name
        self.step = step
        self.pos = pos
        self.valve = valve_by_name[pos]
        self.fuel = fuel
        self.opened_valves = opened_valves or set([])
        self.fps = fps
        # jsut for debug
        self.path = path or []
        self.step2action = step2action or {}
        self.step2openings = step2openings or {}

    @property
    def closed_valves(self):
        return set(self.valve_by_name) - set(self.opened_valves) - {"AA"}

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}(step2openings={self.step2openings}, pos={self.pos}, step={self.step}, "
            f"fuel={self.fuel}, fps={self.fps}), opened_valves={self.opened_valves}"
        )

    def get_successors(self):
        res = {}
        for k, dist in self.valve.connections.items():
            if k in self.opened_valves:
                continue
            if k == self.pos:
                logging.warning(k)
                continue
            if 30 - self.step < dist + 1:
                logging.debug(k)
                continue

            res[k] = dist

        # def get_KPD(k):
        #     if k in self.opened_valves:
        #         return 0
        #     dist = self.valve.connections[k]
        #     profit = max(0, 30 - self .step - dist - TIME_TO_OPEN) * self.valve_by_name[k].flow_rate
        #     # return self.valve_by_name[k].flow_rate * (k not in self.opened_valves )/ self.valve.connections[k]
        #
        #     # return  (k not in self.opened_valves ) / self.valve.connections[k]
        #     return profit
        #
        # sorted_res = sorted(res, key=lambda k: get_KPD(k), reverse=True)
        # r = {k: res[k] for k in sorted_res }
        # return r
        return res

    def move_to_valve(self, name):
        if name == self.pos:
            pass
        dist = self.valve.connections[name]

        cost = dist * TIME_TO_MOVE

        add = cost * self.fps
        new_fuel = self.fuel + add
        self.step2action[self.step] = f"Move {name} in {cost} steps. fuel+={add}"

        self.step += cost
        for c in range(cost):
            self.path += [f"{name}_{c + 1}/{cost}"]

        # order is important here
        self.fuel = new_fuel

        self.pos = name
        self.valve = self.valve_by_name[name]

        return self.fuel

    def open_value(self, name):
        step = self.step
        fuel = self.fuel
        if name == "AA" and self.step == 0:
            assert False
        valve = self.valve_by_name[name]

        cost = TIME_TO_OPEN
        self.step += cost

        # order is important here
        add = cost * self.fps
        self.fuel = fuel + add
        # valve.open()
        self.opened_valves.add(name)

        new_fps = self.fps + valve.flow_rate

        self.step2action[step] = f"Open {name} fps={self.fps}->{new_fps}, fuel+={add}"
        self.step2openings[step] = name  # valve.flow_rate

        self.fps = new_fps

        self.path += [f"+{name}"]

        return self.fuel

    def move_and_open_valve(self, name):
        self.move_to_valve(name)

        if self.valve.flow_rate <= 0:
            return

        if name in self.opened_valves:
            return

        return self.open_value(name)

    def is_goal_state(self):
        if self.step == 30:
            return True

        min_dist = None
        for name, v in self.valve_by_name.items():
            if name == self.pos:
                continue
            if name in self.opened_valves:
                continue
            if v.flow_rate == 0:
                continue
            else:
                dist = self.valve.connections[name]
                if min_dist is None:
                    min_dist = dist
                else:
                    min_dist = min(min_dist, dist)

        if min_dist is None:
            return True

        if 30 - self.step < min_dist:  # TODO not sure!
            return True

        return False

    def copy(self):
        return State(
            self.valve_by_name,
            self.step,
            self.pos,
            self.fuel,
            copy(set(self.opened_valves)),
            self.fps,
            list(self.path),
            copy(dict(self.step2action)),
            copy(dict(self.step2openings)),
        )

    def finalize(self):
        assert self.step <= 30
        add = (30 - self.step) * self.fps
        if add < 0:
            raise

        action = f"Finalize: fps={self.fps}, fuel+={add}"
        self.step2action[self.step] = action

        self.fuel += add
        self.step = 30

    def diff(self, other: "State"):
        return State(
            valve_by_name=self.valve_by_name,
            step=self.step - other.step,
            pos=self.pos,
            fuel=self.fuel - other.fuel,
            opened_valves=self.opened_valves - other.opened_valves,
            fps=self.fps - other.fps,
            path=self.path[len(other.path) :],
            step2action={step: action for step, action in self.step2action.items() if step not in other.step2action},
            step2openings={
                step: action for step, action in self.step2openings.items() if step not in other.step2openings
            },
        )

    def add(self, other: "State"):
        return State(
            valve_by_name=other.valve_by_name,
            step=self.step + other.step,
            pos=other.pos,
            fuel=self.fuel + other.fuel,
            opened_valves=self.opened_valves.union(other.opened_valves),
            fps=self.fps + other.fps,
            path=self.path + other.path,
            step2action={**self.step2action, **other.step2action},
            step2openings={**self.step2openings, **other.step2openings},
        )

    def get_diffs_to_final_state(self):
        global CACHE
        cache_key = (self.pos, self.step, tuple(sorted(self.opened_valves)))
        if cache_key in CACHE:
            return CACHE[cache_key]

        start_state = self
        diff_candidates = []
        if start_state.is_goal_state():
            state = start_state.copy()
            state.finalize()
            diff = state.diff(start_state)
            diff_candidates.append(diff)
            CACHE[cache_key] = diff_candidates
            return diff_candidates

        successors = start_state.get_successors()
        for name in successors:
            to_inspect(start_state)

            state = start_state.copy()

            to_open = True
            if to_open is True:
                if name in state.opened_valves:
                    continue
                change_state = state.move_and_open_valve
            else:
                change_state = state.move_to_valve

            # do job
            change_state(name)

            candidates_from_changed_state = state.get_diffs_to_final_state()
            candidates_from_start_state = [
                candidate.add(state.diff(start_state)) for candidate in candidates_from_changed_state
            ]
            diff_candidates.extend(candidates_from_start_state)

        diff_candidates.sort(key=lambda x: -x.fuel)
        if not diff_candidates:
            start_copy = start_state.copy()
            finalized_copy = start_state.copy()
            finalized_copy.finalize()
            diff_candidates = [finalized_copy.diff(start_copy)]

        CACHE[cache_key] = diff_candidates

        return diff_candidates


def to_inspect(state):
    if state is None:
        return
    toInspect = [
        "DD_1/1",
        "+DD",
        # "CC_1/1",
        "BB_1/2",
        "BB_2/2",
        "+BB",
        "JJ_1/3",
        # "JJ_2/3",
        # "JJ_3/3",
        # "+JJ",
    ]
    if state.path[: len(toInspect)] == toInspect:
        pass


def _get_diff_to_final(start_state):
    state = start_state.copy()
    state.finalize()
    return state.diff(start_state)


def get_max_fuel_for_pair_agents(start_states: tuple[State]) -> tuple[State]:
    # max2
    global CACHE
    positions = tuple(s.pos for s in start_states)
    steps = tuple(s.step for s in start_states)
    opened_valves = set.union(*(set(s.opened_valves) for s in start_states))

    cache_key = (positions, steps, tuple(sorted(opened_valves)))
    if cache_key in CACHE:
        return CACHE[cache_key]

    diff_candidates = []

    successors_by_state = tuple((sorted(state.get_successors())) for state in start_states)
    successors = tuple(set.union(*(set(s.get_successors()) for s in start_states)))
    if set(successors_by_state[0]) != set(successors_by_state[1]):
        pass
    if not successors:
        states = tuple(map(State.copy, start_states))

        best_diff = []
        for i, state in enumerate(states):
            state.finalize()
            diff = state.diff(start_states[i])
            best_diff.append(diff)
        best_diff = tuple(best_diff)
        CACHE[cache_key] = best_diff
        return best_diff

    succ_permutations = list(itertools.product(*successors_by_state))
    for names in succ_permutations:
        if len(set(names)) == 1:
            continue
        states = tuple(map(State.copy, start_states))

        for name, state in zip(names, states):
            state.move_and_open_valve(name)
            state.opened_valves.update(names)

        candidates_pair = get_max_fuel_for_pair_agents(states)
        candidates_from_start_state = []
        for start_state, state, final_candidate in zip(
            start_states,
            states,
            candidates_pair,
        ):
            candidate = final_candidate.add(state.diff(start_state))
            candidates_from_start_state.append(candidate)

        diff_candidates.append(tuple(candidates_from_start_state))

    diff_candidates.sort(key=lambda x: -sum(s.fuel for s in x))

    if not diff_candidates:  # THERE US A PROBLEM IF ONLY 1 LEFT FOR BOTH
        final_diffs0 = start_states[0].copy().get_diffs_to_final_state()
        final_diffs1 = start_states[1].copy().get_diffs_to_final_state()

        final_diffs0.sort(key=lambda x: -x.fuel)
        final_diff0 = final_diffs0[0]
        assert final_diff0.fuel == max(s.fuel for s in final_diffs0)

        final_diffs1.sort(key=lambda x: -x.fuel)
        final_diff1 = final_diffs1[0]
        assert final_diff1.fuel == max(s.fuel for s in final_diffs1)

        # that means one of the state has no options
        if len(successors) == 1:
            if final_diff1.fuel > final_diff0.fuel:
                final0 = start_states[0].copy()
                final0.finalize()
                final_diff0 = final0.diff(start_states[0])
            else:
                final1 = start_states[1].copy()
                final1.finalize()
                final_diff1 = final1.diff(start_states[1])
        diff_candidates = [
            tuple((final_diff0, final_diff1))
            # tuple([
            #     final_diff0.diff(start_states[0]),
            #     final_diff1.diff(start_states[1]),
            # ])
        ]

    best_states_pair = diff_candidates[0]
    CACHE[cache_key] = best_states_pair

    return best_states_pair


def get_max_fuel(start_state: State):
    to_inspect(start_state)

    diff_candidates = start_state.get_diffs_to_final_state()

    return max(s.fuel for s in diff_candidates)


def task1(inp):
    valve_by_name = _parse_puzzle(tuple(inp))
    print("paths initialized")

    state = State(valve_by_name)

    # # # TODO debug
    # state.move_and_open_valve("DD")
    # # state.move_to_valve("CC")
    # state.move_and_open_valve("BB")
    # state.move_and_open_valve("JJ")
    # # state.move_to_valve("EE")
    # state.move_and_open_valve("HH")
    # state.move_and_open_valve("EE")
    # state.move_and_open_valve("CC")
    # state.finalize()
    # # # state.open_value('AA')

    global CACHE
    CACHE = {}
    final_state = get_max_fuel(state)
    return final_state


def task2(inp):
    valve_by_name = _parse_puzzle(tuple(inp))
    state1 = State(valve_by_name, step=4)
    state2 = State(valve_by_name, step=4)

    global CACHE
    CACHE = {}
    final_state = get_max_fuel_for_pair_agents((state1, state2))
    res = sum(s.fuel for s in final_state)
    return res


def part1():
    start = datetime.datetime.now()

    try:
        test(task1, expected=1651)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    res = run(task1)
    end = datetime.datetime.now()
    assert res == 1915, res
    print(end - start)


def part2():
    start = datetime.datetime.now()
    try:
        test(task2, expected=1707)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    run(task2)

    end = datetime.datetime.now()
    print(end - start)


if __name__ == "__main__":
    # part1()
    part2()
