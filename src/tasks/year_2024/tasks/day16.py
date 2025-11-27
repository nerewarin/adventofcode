from src.utils.input_formatters import cast_2d_list_elements
from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()

START = "S"
END = "E"
WALL = "#"
SPACE = "."


class ReindeerMaze: ...


def _parse_input(inp: list[str]) -> list[list[str]]:
    return cast_2d_list_elements(inp, type_=str)


def task(inp: list[str], task_num: int = 1) -> int: ...


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(task1, 7036)
    test(task1, 11048, test_part=2)
    run(task1)
