"""
--- Day 21: Keypad Conundrum ---
https://adventofcode.com/2024/day/21
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
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
from src.utils.test_and_run import run, test

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

    def __str__(self):
        return self.value


class UnsortedButtons(dict[Button, int]):
    def __str__(self):
        data = {button.value: value for button, value in self.items()}
        return str(data)

    def __repr__(self):
        return self.__str__()


class PossibleButtonsSequence(list[UnsortedButtons]):
    def __str__(self):
        return f"{[{button.value: value for button, value in b2v.items()} for b2v in self]}"

    def __repr__(self):
        return self.__str__()


class BottonList(list[Button]):
    def __str__(self):
        return f"{[button.value for button in self]}"

    def __repr__(self):
        return self.__str__()


class BaseKeypad(ABC):
    layout: ClassVar[tuple[tuple[Button]]]
    initial_position: ClassVar[Position2D]

    def __init__(self):
        self.cursor = self.initial_position

    def __repr__(self):
        return f"{self.__class__.__qualname__}(cursor={self.cursor})"

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

    initial_position = Position2D(2, 3)
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

    initial_position = Position2D(2, 0)
    layout = (
        (Button.gap, Button.up, Button.activate),
        (Button.left, Button.down, Button.right),
    )


# def define_sequnece(possible_buttons_sequence: PossibleButtonsSequence, keypad: BaseKeypad)
#     possible_buttons_sequence


def define_buttons_order(button_to_amount: UnsortedButtons, keypad: BaseKeypad, cmd: str) -> list[Button]:
    # sort by distance to arm to choose best buttons order from possible options
    ordered_target_buttons = []

    # compute steps from arm to every button
    target_buttons_by_price = defaultdict(list)

    for target_button in button_to_amount:
        target_button_pos = keypad.get_button_position(target_button)
        price = keypad.cursor.manhattan_to(target_button_pos)
        target_buttons_by_price[price].append(target_button)
    if button_to_amount == {"<": 2, "^": 2}:
        _logger.debug(f"chosen ordering for {button_to_amount} problem: {target_buttons_by_price}")
        if cmd in "7":
            # target_buttons_by_price = {2: [Button.up], 0: [Button.left]}
            pass

    # resolve buttons in the closest order
    for price in sorted(target_buttons_by_price):
        target_buttons = target_buttons_by_price[price]
        for target_button in target_buttons:
            amount = button_to_amount[target_button]

            button_presses = [target_button] * amount

            ordered_target_buttons.extend(button_presses)

    return ordered_target_buttons


class BaseSlave(ABC):
    _keypad_cls: type[BaseKeypad]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keypad = self._keypad_cls()

    def __repr__(self):
        return f"{self.__class__.__qualname__}(keypad={self.keypad})"

    @abstractmethod
    def _transfer_to_self_commands(
        self, possible_buttons_sequence: PossibleButtonsSequence, cmd: str
    ) -> PossibleButtonsSequence: ...

    def execute_command(self, cmd) -> PossibleButtonsSequence:
        """executes high-level command e.g. "press 0"

        (old approach) # TODO actualize
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

        child_possible_buttons_sequence = self._propagate(cmd)
        possible_buttons_sequence = self._transfer_to_self_commands(child_possible_buttons_sequence, cmd)
        return possible_buttons_sequence

    @abstractmethod
    def _propagate(self, cmd: str) -> PossibleButtonsSequence:
        """Propagates high-level cmd to control"""
        ...

    # low-level options
    def perform_actions_to_press_button(self, target_button: Button) -> PossibleButtonsSequence:
        res = []
        keypad = self.control.keypad

        # collect move buttons to dict for further execution by master in order they prefers
        target_button_pos = keypad.get_button_position(target_button)
        actions_group = keypad.cursor.get_actions_to_another_as_dict(target_button_pos)
        if actions_group:
            # move_buttons_group
            res.append(
                UnsortedButtons(
                    **{
                        Button(ORTHOGONAL_DIRECTION_SYMBOLS_BY_ENUM[action]): count
                        for action, count in keypad.cursor.get_actions_to_another_as_dict(target_button_pos).items()
                    }
                )
            )
            # shift cursor
            keypad.cursor = target_button_pos

        # press activation
        # activate_buttons_group
        res.append({Button.activate: 1})

        return res


class BaseAgent(ABC):
    def __init__(self, controls: BaseSlave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control = controls

    def __repr__(self):
        return f"{self.__class__.__qualname__}(controls={self.control})"

    def _propagate(self, cmd: str) -> PossibleButtonsSequence:
        """Propagates high-level cmd to control

        push original cmd inside, e.g. 1

        From
            [{'<': 2, '^': 1},
            {'A': 1}]

        taken on 0 level, returns

        ^<<A

        [
            {<: 1},
            {A: 1},
            {<: 1, v: 1},
            {A: 1},
            {A: 1},
            {>: 2, ^: 1},
            {A: 1}
        ]
        to use for human or for further robot

        """
        possible_buttons_sequence = self.control.execute_command(cmd)

        return possible_buttons_sequence


class Human(BaseAgent):
    @classmethod
    def with_robots(cls, robots_amount: int):
        control = Robot(0, controls=Door())
        for i in range(robots_amount - 1):
            control = Robot(i + 1, controls=control)

        return Human(controls=control)

    def _get_ordered(self, possible_buttons_sequence: PossibleButtonsSequence, cmd) -> list[Button]:
        # orders commands in best order and returns action needed on self keypad to execute propagated commands
        my_ordered_buttons_for_child: list[Button] = []
        for target_button_group in possible_buttons_sequence:
            ordered_target_buttons = define_buttons_order(
                target_button_group, self.control.keypad, cmd
            )  # no sence whats order here
            my_ordered_buttons_for_child.extend(ordered_target_buttons)

        return my_ordered_buttons_for_child

    def compute_shortest_buttons_list(self, cmd: str) -> list[Button]:
        possible_buttons_sequence = self._propagate(cmd)

        my_ordered_buttons_for_child = self._get_ordered(possible_buttons_sequence, cmd)

        return my_ordered_buttons_for_child


class Robot(BaseAgent, BaseSlave):
    # robotic arm can be controlled remotely via a directional keypad
    _keypad_cls = DirectionalKeypad

    def __init__(self, level, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def __repr__(self):
        return f"{self.__class__.__qualname__}(level={self.level}, keypad={self.keypad}, controls={self.control})"

    def _transfer_to_self_commands(
        self, possible_buttons_sequence: PossibleButtonsSequence, cmd: str
    ) -> PossibleButtonsSequence:
        # orders commands in best order and returns action needed on self keypad to execute propagated commands
        my_ordered_buttons: list[Button] = []
        for target_button_group in possible_buttons_sequence:
            ordered_target_buttons = define_buttons_order(target_button_group, self.control.keypad, cmd)
            _logger.debug(
                f"Robot{self.level}: sorting {UnsortedButtons(target_button_group)} into {BottonList(ordered_target_buttons)}"
            )
            my_ordered_buttons.extend(ordered_target_buttons)

        new_string = "".join([cmd.value for cmd in my_ordered_buttons])

        _logger.debug(
            f"Robot{self.level}: thnking on task {new_string} of len {len(new_string)} to execute on {self.control.keypad}"
        )

        buttons_sequence_for_parent = []
        for button in my_ordered_buttons:
            actions = self.perform_actions_to_press_button(button)
            buttons_sequence_for_parent.extend(actions)

        _logger.debug(f"Robot{self.level} outputs {PossibleButtonsSequence(buttons_sequence_for_parent)}")

        return buttons_sequence_for_parent


class Door(BaseSlave):
    _keypad_cls = NumericKeypad

    def execute_command(self, cmd: str) -> PossibleButtonsSequence:
        return [{Button(cmd): 1}]

    def _propagate(self, cmd: str) -> PossibleButtonsSequence:
        """Propagates high-level cmd to control"""
        pass

    def _transfer_to_self_commands(
        self, possible_buttons_sequence: PossibleButtonsSequence, cmd: str
    ) -> PossibleButtonsSequence:
        return possible_buttons_sequence


@dataclasses.dataclass
class DoorCodeProblem:
    """
    High-level problem of finding shortest sequence of button presses
    you need to type on your directional keypad in order to cause the code to be typed on the numeric keypad
    """

    door_code: str
    human: Human

    def get_shortest_sequence(self) -> list[str]:
        cmds = []
        strgins = []
        for code in self.door_code:
            new_cmds = self.human.compute_shortest_buttons_list(code)
            new_string = "".join([cmd for cmd in new_cmds])
            strgins.append(new_string)
            cmds.extend(new_cmds)
            _logger.debug(
                f"get_shortest_sequence {code} -> {new_string} len of +{len(new_cmds)} ({len(cmds)} in total)"
            )

        # 379A: <v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
        #    3: <v<A>>^AvA^A # manual ordering from oroginal
        #    3: v<<A>>^AvA^A # original

        # tail <vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
        # 7:

        # res = [
        #     # expected
        #     "<v<A>>^AvA^A"
        #     "<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A"
        #     "v<<A>>^AvA^A"
        #     "v<<A>>^AAv<A"
        #     ""
        #     "<A>>^AAvAA^<A>Av<A>^AA<A>Av<A<A>>^AAAvA^<A>A"
        # ]
        # # string
        # [
        #     "v<<A>>^AvA^A",
        #     # expected
        #     "<v<A>>^AvA^Av<<A>>^AAv<A<A>>^AAvAA^<A>A",
        #     "v<A>^AA<A>A",
        #     "v<A<A>>^AAAvA^<A>A",
        # ]

        return cmds


class KeypadConundrum:
    def __init__(self, door_codes: list[str], task_num: int, robots_amount: int = 3):
        self.human = Human.with_robots(robots_amount)

        self.door_codes = door_codes

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(door_codes={self.door_codes}. task_num={self.task_num})"

    @staticmethod
    def get_numeric_part(door_code: str) -> int:
        assert door_code[-1] == "A"
        numeric_part = door_code[:-1]
        assert len(numeric_part) == 3
        number = int(numeric_part)
        return number

    @classmethod
    def _get_complexity(cls, door_code: str, shortest_sequence: list[str]) -> int:
        number = cls.get_numeric_part(door_code)
        shortest_sequence_len = len(shortest_sequence)
        res = shortest_sequence_len * number
        return res

    def solve(self) -> int | None:
        res = []
        for door_code in self.door_codes:
            problem = DoorCodeProblem(door_code, self.human)

            shortest_sequence = problem.get_shortest_sequence()

            res.append(self._get_complexity(door_code, shortest_sequence))

        return sum(res)


def task(inp: list[str], task_num: int | None = 1, **kw) -> int:
    return KeypadConundrum(inp, task_num, **kw).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2, **kw)


if __name__ == "__main__":
    for door_code_, expected_num_ in {
        "029A": 29,
        "980A": 980,
        "179A": 179,
        "456A": 456,
        "379A": 379,
    }.items():
        assert KeypadConundrum.get_numeric_part(door_code_) == expected_num_

    # # In total, there are three shortest possible sequences of button presses on this directional keypad that would
    # # cause the robot to type 029A: <A^A>^^AvvvA, <A^A^>^AvvvA, and <A^A^^>AvvvA.
    test(task1, 29 * len("<A^A>^^AvvvA"), test_data=["029A"], robots_amount=1)

    # add 2nd layer
    test(task1, 29 * len("v<<A>>^A<A>AvA<^AA>A<vAAA>^A"), test_data=["029A"], robots_amount=2)

    # add 3rd layer (default mode)
    test(task1, 29 * len("<vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A"), test_data=["029A"])

    for test_statement in """
            029A: <vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A
            980A: <v<A>>^AAAvA^A<vA<AA>>^AvAA<^A>A<v<A>A>^AAAvA<^A>A<vA>^A<A>A
            179A: <v<A>>^A<vA<A>>^AAvAA<^A>A<v<A>>^AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
            456A: <v<A>>^AA<vA<A>>^AAvAA<^A>A<vA>^A<A>A<vA>^A<A>A<v<A>A>^AAvA<^A>A
            379A: <v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
        """.split("\n"):
        test_statement = test_statement.strip()
        if not test_statement:
            continue
        door_code_, shortest_sequence_ = (x.strip() for x in test_statement.split(":"))
        if door_code_.startswith("#"):
            continue
        num_ = KeypadConundrum.get_numeric_part(door_code_)
        expected_res_ = num_ * len(shortest_sequence_)

        test(task1, expected_res_, test_data=[door_code_])

    # test(task1, 126384)

    run(task1)
    # from earliest to latest probes:
    # 157554 is too high!
    # 165062 is too high!
