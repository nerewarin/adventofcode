from src.utils.position import Position2D


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
