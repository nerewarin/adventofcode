import re
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import test

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


def get_point_of_intersection(machine: Machine) -> Position2D | None:
    """
    The main idea:
    we need to combine two vectors a and b to get prize.
    for that we need to get a point of intersection of a started from 0 and going to top-right,
    and b started from prize and going bot left. for that we revert it:
     common eq: y = kx + b
     our case:
     A: a1 * y = b1 * x => y = x * b1/a1
     B: y = x * b2/a2
    A stays the same, do reversion of B:
    A: y = x * b1/a1  (1)
    B: y = prize_y - (x - price_x) * b2/a2  (2)
    combining (1) and (2), we get
    x * b1/a1  = prize_y + (x - price_x) * b2/a2
    move x to the left, rest to the right
    x * b1/a1  - (x - price_x) * b2/a2 = prize_y
    x * b1/a1 - x * b2/a2 = prize_y - price_x * b2/a2
    derive x
    x * (b1/a1 - b2/a2) = prize_y - price_x * b2/a2
    finally,
    x = (prize_y - price_x * b2/a2) /  (b1/a1 - b2/a2)
    if x is integer,
    then we will find y from (1)
    if its also integer, we found intersection.
    All other cases not works for us, return None

    Returns:
        Pos

    """
    a = machine.button_a
    b = machine.button_b

    # make variables for y = x * b1/a1
    # a1 = a.y
    # b1 = a.x
    # we then use b1/a1 and b2/a2 much, define them
    slope1 = a.y / a.x
    slope2 = b.y / b.x

    # x = (prize_y + price_x * b2/a2) /  (b1/a1 + b2/a2)
    x = (machine.prize.y - machine.prize.x * slope2) / (slope1 - slope2)

    diff_to_int = abs(x - int(x))
    if diff_to_int < 0.05:
        _logger.debug(f"{x=}, {diff_to_int=} OK")
    else:
        _logger.debug(f"{x=}, {diff_to_int=} bad =(")

    # now recheck
    a_clicks = 80
    b_clicks = 40

    a_x = a.x * a_clicks
    b_x = b.x * b_clicks
    expected_x = a_x + b_x

    a_y = a.y * a_clicks
    b_y = b.y * b_clicks
    expected_y = a_y + b_y

    if a == (94, 34) and b == (22, 67) and machine.prize == (8400, 5400):
        assert machine.prize == (expected_x, expected_y)
    return x


def calculate_tokens_to_prize(machine: Machine) -> int:
    a_slope = machine.button_a.slope()
    b_slope = machine.button_b.slope()
    prize_slope = machine.prize.slope()

    msg = f"{a_slope=}, {b_slope=}"
    slope_diff = abs(a_slope - b_slope)
    if not slope_diff:
        msg += " EQUAL!"
        raise ValueError(msg)  # we need to investigate that case separately
    elif slope_diff < 0.1:
        msg += f" almost EQUAL: {slope_diff}!"
        raise ValueError(msg)  # we need to investigate that case separately
    else:
        msg += f" {slope_diff=}"

    msg += f" {prize_slope=}. full state: {machine.button_a=}, {machine.button_b=}, {machine.prize=}"
    _logger.debug(msg)

    get_point_of_intersection(machine)

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
    test(task1, 280)
    # run(task1)
