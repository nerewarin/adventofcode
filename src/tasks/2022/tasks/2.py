"""--- Day 2: Rock Paper Scissors ---
https://adventofcode.com/2022/day/2
"""
from src.utils.input_formatters import lines_to_tuples
from src.utils.test_and_run import test, run

_SHAPES = {
        "A": 1,
        "B": 2,
        "C": 3,

        "X": 1,
        "Y": 2,
        "Z": 3,
    }

_SHAPE_SIZE = len(set(_SHAPES.values()))


def _get_shape(sign):
    # (1 for Rock, 2 for Paper, and 3 for Scissors)
    return _SHAPES[sign]


def _get_shape_for_goal(opponent_shape, goal="win"):
    term = {
        "win": 1,
        "draw": 0,
        "lose": -1,
    }[goal]

    # return shape to beat opponent_shape
    shape = (opponent_shape + term) % _SHAPE_SIZE

    return shape


def _calc_round_result(opponent_shape, your_shape):
    """Calc round result"""
    # 0 is draw, 1 is win, 2 is loose
    raw_res = your_shape - opponent_shape

    # 0 is loose, 1 is draw, 2 is a win
    res = _get_shape_for_goal(raw_res)

    # 0 is loose, 3 is draw, 6 is a win
    return res * 3


def _pick_given_sign(_, your_task_shape) -> int:
    return your_task_shape


def _pick_to_match_result(opponent_shape, your_task) -> int:
    """
        Anyway, the second column says how the round needs to end:
        X means you need to lose,
        Y means you need to end the round in a draw,
        and Z means you need to win. Good luck!
    """
    goal = {1: "lose", 2: "draw", 3: "win"}[your_task]
    return _get_shape_for_goal(opponent_shape, goal=goal) or 3


def rock_paper_scissors(inp, second_sign_interpreter=_pick_given_sign):
    res = 0
    for opponent_sign, your_task_sign in lines_to_tuples(inp):
        opponent_shape = _get_shape(opponent_sign)
        your_shape = second_sign_interpreter(opponent_shape, _get_shape(your_task_sign))
        res += your_shape + _calc_round_result(opponent_shape, your_shape)
    return res


if __name__ == "__main__":
    # # part 1
    test(rock_paper_scissors, expected=15)
    run(rock_paper_scissors)

    # part 2
    test(rock_paper_scissors, second_sign_interpreter=_pick_to_match_result, expected=12)
    run(rock_paper_scissors, second_sign_interpreter=_pick_to_match_result)
