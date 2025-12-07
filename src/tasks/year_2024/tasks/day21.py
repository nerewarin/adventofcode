"""
--- Day 21: Keypad Conundrum ---
https://adventofcode.com/2024/day/21
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import ClassVar

from src.utils.directions import (
    ORTHOGONAL_DIRECTION_SYMBOLS_BY_ENUM,
    SYMBOL_DOWN,
    SYMBOL_LEFT,
    SYMBOL_RIGHT,
    SYMBOL_UP,
)
from src.utils.logger import get_logger
from src.utils.position import Position2D, get_value_by_position
from src.utils.test_and_run import test

_logger = get_logger()


class Button(StrEnum):
    gap = ""
    up = SYMBOL_UP
    down = SYMBOL_DOWN
    left = SYMBOL_LEFT
    right = SYMBOL_RIGHT
    activate = "A"
    zero = "0"
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"
    six = "6"
    seven = "7"
    eight = "8"
    nine = "9"


class BaseKeypad(ABC):
    layout: ClassVar[tuple[tuple[Button]]]

    def __repr__(self):
        return f"{self.__class__.__qualname__}"

    # should it be classmethod or store current state there inside?
    def get_button_by_position(self, position: Position2D) -> Button:
        return get_value_by_position(position, self.layout)

    @lru_cache
    def get_button_position(self, button: Button) -> Position2D:
        for y, line in enumerate(self.layout):
            for x, value in enumerate(line):
                position = Position2D(x, y)
                if self.get_button_by_position(position) == button:
                    return position
        raise ValueError(f"No button {button} found in layout {self.layout}")


class NumericKeypad(BaseKeypad):
    """
    +---+---+---+
    | 7 | 8 | 9 |
    +---+---+---+
    | 4 | 5 | 6 |
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
        | 0 | A |
        +---+---+
    """

    layout = tuple(
        tuple(Button(ch) if ch.strip() else Button.gap for ch in row)
        for row in (
            "789",
            "456",
            "123",
            " 0A",
        )
    )


class DirectionalKeypad(BaseKeypad):
    """
        +---+---+
        | ^ | A |
    +---+---+---+
    | < | v | > |
    +---+---+---+
    """

    initial_position: Position2D = Position2D(2, 0)
    # path: list[OrthogonalPositionState] = dataclasses.field(default_factory=list)
    layout = (
        (Button.gap, Button.up, Button.activate),
        (Button.left, Button.down, Button.right),
    )


class BaseDevice(ABC):
    _keypad_cls: type[BaseKeypad]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keypad = self._keypad_cls()

    def __repr__(self):
        return f"{self.__class__.__qualname__}(keypad={self.keypad})"


class BaseAgent(ABC):
    _start_arm_position: ClassVar[Position2D]

    def __init__(self, controls: BaseDevice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control = controls
        self.arm_position = self._start_arm_position

    def __repr__(self):
        return f"{self.__class__.__qualname__}(controls={self.control}, arm_position={self.arm_position})"

    def execute_command(self, cmd) -> list[Button]:
        """
        Gets high-level command.
        If it controls another object:
            1. propagate command to it
            2. get back low-level commands needed to pass to it to execute your high-level command
            3. transform their low-level commands into commands on your kbd
            4. return back commands passed to your kbd
        if not:
            1. transform commands into commands on your kbd
            2. return back commands passed to your kbd

        """
        low_lvl_commands = self.control.execute_command(cmd)
        return low_lvl_commands


class Human(BaseAgent):
    _start_arm_position = Position2D(2, 0)  # TODO ??


class Robot(BaseAgent, BaseDevice):
    _keypad_cls = DirectionalKeypad
    _start_arm_position = Position2D(2, 3)

    def execute_command(self, cmd) -> list[Button]:
        target_button = Button(cmd)
        target_button_pos = self.control.keypad.get_button_position(target_button)
        actions = self.arm_position.get_actions_to(target_button_pos)
        buttons = [Button(ORTHOGONAL_DIRECTION_SYMBOLS_BY_ENUM[action]) for action in actions]
        self.arm_position = target_button_pos
        buttons.append(Button.activate)
        return buttons


class Door(BaseDevice):
    _keypad_cls = NumericKeypad


@dataclasses.dataclass
class DoorCodeProblem:
    """
    High-level problem of finding shortest sequence of button presses
    you need to type on your directional keypad in order to cause the code to be typed on the numeric keypad
    """

    door_code: str
    agent: BaseAgent

    def get_shortest_sequence(self) -> list[str]:
        cmds = []
        for code in self.door_code:
            new_cmds = self.agent.execute_command(code)
            cmds.extend(new_cmds)

        return cmds


class KeypadConundrum:
    def __init__(self, door_codes: list[str], task_num: int, agent: BaseAgent | None = None):
        self.agent = agent or Human(controls=Robot(controls=Robot(controls=Door())))

        self.door_codes = door_codes

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(door_codes={self.door_codes}. task_num={self.task_num})"

    def _get_complexity(self, door_code: str, shortest_sequence: list[str]) -> int:
        assert door_code[-1] == "A"
        numeric_part = door_code[:-1]
        assert len(numeric_part) == 3
        number = int(numeric_part)

        shortest_sequence_len = len(shortest_sequence)

        res = shortest_sequence_len * number
        return res

    def solve(self) -> int | None:
        res = []
        for door_code in self.door_codes:
            problem = DoorCodeProblem(door_code, self.agent)

            shortest_sequence = problem.get_shortest_sequence()

            res.append(self._get_complexity(door_code, shortest_sequence))

        return sum(res)


def task(inp: list[str], task_num: int | None = 1, **kw) -> int:
    return KeypadConundrum(inp, task_num, **kw).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


def _get_one_layer_agent() -> Human:
    agent = Human(controls=Robot(controls=Door()))
    return agent


if __name__ == "__main__":
    # In total, there are three shortest possible sequences of button presses on this directional keypad that would
    # cause the robot to type 029A: <A^A>^^AvvvA, <A^A^>^AvvvA, and <A^A^^>AvvvA.
    test(task1, 29 * len("<A^A>^^AvvvA"), test_data=["029A"], agent=_get_one_layer_agent())

    # test(task1, 126384)

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
    #
    # # test task2
    # _rexp = re.compile(r"(\d+).*?(\d+)")
    # for test_statement in """
    #         029A: <vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A
    #         980A: <v<A>>^AAAvA^A<vA<AA>>^AvAA<^A>A<v<A>A>^AAAvA<^A>A<vA>^A<A>A
    #         179A: <v<A>>^A<vA<A>>^AAvAA<^A>A<v<A>>^AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
    #         456A: <v<A>>^AA<vA<A>>^AAvAA<^A>A<vA>^A<A>A<vA>^A<A>A<v<A>A>^AAvA<^A>A
    #         379A: <v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
    #     """.split("\n"):
    #     if not test_statement:
    #         continue
    #     _res = _rexp.search(test_statement)
    #     if _res:
    #         _expected_path_with_cheats, _savings = map(int, _res.groups())
    #         test(task2, _expected_path_with_cheats, target_savings=_savings, strict=True)
    #
    # run(task2)  # 1011325
