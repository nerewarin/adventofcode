import re
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from timeit import Timer

from src.utils.logger import get_logger
from src.utils.position import Position2D
from src.utils.test_and_run import run, test

_logger = get_logger()

_button_rexp = re.compile(r"Button [AB]: X([-+]\d+), Y([-+]\d+)")
_prize_rexp = re.compile(r"Prize: X=(\d+), Y=(\d+)")


def timeit_deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = Timer(lambda: func(*args, **kwargs))
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {t.timeit(number=1):.6f}s")
        return result

    return wrapper


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


def fast_fail(machine: Machine, max_tokens: int) -> bool:
    """
    Checks if prize x and prize y is even reachable by spending no more than {max_tokens} tokens
    """
    for axis in (0, 1):
        max_pos = max_tokens * (machine.button_a[axis] + machine.button_b[axis])
        if max_pos < machine.prize[axis]:
            _logger.debug(
                f"Fast fail on {axis=}: {max_tokens=} * ({machine.button_a[axis]=} + {machine.button_b[axis]=}) = {max_pos} < {machine.prize[axis]=}"
            )
            return True
    return False


def get_tokens_to_pay(machine: Machine) -> int:
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
        int

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

    # now let's use Cramer rule to minimize fractions at slope calculation
    ax, ay = machine.button_a.x, machine.button_a.y
    bx, by = machine.button_b.x, machine.button_b.y
    px, py = machine.prize.x, machine.prize.y

    # Determinant of the 2x2 matrix [[ax, bx], [ay, by]]
    d = ax * by - ay * bx

    if d == 0:
        _logger.debug(f"{d=}: Parallel / no unique solution")
        return 0

    # Cramer's rule
    na = Fraction(px * by - py * bx, d)
    nb = Fraction(ax * py - ay * px, d)

    # Check that point coordinates are integers
    if na.denominator != 1 or nb.denominator != 1:
        _logger.debug(f"some of {na.denominator=} or {nb.denominator=} are fractions!")
        return 0

    na_int = na.numerator
    nb_int = nb.numerator

    if na_int < 0 or nb_int < 0:
        _logger.debug(f"some of {na_int=} or {nb_int=} are negative!")
        return 0

    res = na_int * machine.button_a_price + nb_int * machine.button_b_price
    _logger.debug(f"{res=}")
    return res


def calculate_tokens_to_prize(machine: Machine, max_tokens: int, use_fast_fail: bool = True) -> int:
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

    if use_fast_fail and fast_fail(machine, max_tokens):
        return 0

    return get_tokens_to_pay(machine)


def task(inp: list[str], task_num: int, max_tokens: int, use_fast_fail):
    return sum(calculate_tokens_to_prize(machine, max_tokens, use_fast_fail) for machine in _parse_input(inp))


@timeit_deco
def task1(inp, use_fast_fail=True):
    return task(inp, 1, max_tokens=100, use_fast_fail=use_fast_fail)


if __name__ == "__main__":
    test(task1, 480)
    assert run(task1, True) > 18025
    assert run(task1, False) > 18025
