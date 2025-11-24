import re
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run

_logger = get_logger()

_button_rexp = re.compile(r"Button [AB]: X([-+]\d+), Y([-+]\d+)")
_prize_rexp = re.compile(r"Prize: X=(\d+), Y=(\d+)")


@dataclass
class Machine:
    button_a: Position2D
    button_b: Position2D
    prize: Position2D

    button_a_price: int = 3
    button_b_price: int = 1


def _parse_input(inp: list[str]) -> list[Machine]:
    machines: list[Machine] = []

    for i, line in enumerate(inp):
        match i % 4:
            case 0:
                button_a = Position2D.integer_point_from_tuple_of_strings(_button_rexp.findall(line)[0])
            case 1:
                button_b = Position2D.integer_point_from_tuple_of_strings(_button_rexp.findall(line)[0])
            case 2:
                prize = Position2D.integer_point_from_tuple_of_strings(_prize_rexp.findall(line)[0])
                machines.append(Machine(button_a, button_b, prize))
            case 3:
                continue

    return machines


def calculate_tokens_to_prize(machine: Machine) -> int:
    a_slope = machine.button_a.slope()
    b_slope = machine.button_b.slope()
    prize_slope = machine.prize.slope()

    msg = f"{a_slope=}, {b_slope=}"
    slope_diff = abs(a_slope - b_slope)
    if not slope_diff:
        msg += " EQUAL!"
    elif slope_diff < 0.1:
        msg += f" almost EQUAL: {slope_diff}!"
    else:
        msg += f" {slope_diff=}"
    msg += f" {prize_slope=}. full state: {machine.button_a=}, {machine.button_b=}, {machine.prize=}"

    _logger.debug(msg)
    tokens = 0
    return tokens


def task(inp: list[str], task_num: int):
    machines = _parse_input(inp)
    total_tokens = 0
    for machine in machines:
        tokens = calculate_tokens_to_prize(machine)
        total_tokens += tokens
    _logger.debug("ask finished")


def task1(inp):
    return task(inp, 1)


def task2(inp):
    return task(inp, 2)


if __name__ == "__main__":
    # test(task1, 280)
    run(task1)
