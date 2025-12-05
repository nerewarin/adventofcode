import logging

from src.utils.logger import get_message_only_logger
from src.utils.position import Position2D

_maze_logger = get_message_only_logger()


def parse_maze(
    inp: list[str], start_symbol: str | None = "S", end_symbol: str | None = "E"
) -> tuple[list[list[str]], Position2D, Position2D]:
    start = None
    end = None
    maze = [[] for _ in inp]
    for row, line in enumerate(inp):
        for col, symbol in enumerate(line):
            maze[row].append(symbol)
            if symbol == start_symbol:
                assert start is None
                start = Position2D(col, row)
            elif symbol == end_symbol:
                assert end is None
                end = Position2D(col, row)
    assert start and end
    return maze, start, end


def draw_maze(maze: list[list[str]], level: int = logging.DEBUG) -> None:
    level = level or logging.DEBUG
    for i, line in enumerate(maze):
        line_str = "".join(line)
        _maze_logger.log(level, line_str)
