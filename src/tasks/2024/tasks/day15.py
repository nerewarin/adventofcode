import re

from src.utils.directions_orthogonal import DIRECTION_SYMBOLS, DIRECTIONS_BY_ENUM, DirectionEnum, go
from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()

_coordinate_pattern = r"(-?\d+)"
_point_pattern = f"{_coordinate_pattern},{_coordinate_pattern}"

_input_rexp = re.compile(rf"p={_point_pattern} v={_point_pattern}")


BOX = "O"
WALL = "#"
AGENT = "@"
SPACE = "."


def _parse_input(inp: list[str], task_num: int) -> tuple[list[list[str]], list[DirectionEnum]]:
    warehouse_map = []
    for i, line in enumerate(inp):
        warehouse_map.append([symbol for symbol in line.strip()])
        if not line:
            break

    directions = []
    for line in inp[i + 1 :]:
        if not line:
            break

        for symbol in line:
            directions.append(DIRECTION_SYMBOLS[symbol])

    return warehouse_map, directions


def make_moves(warehouse_map: list[list[str]], directions: list[DirectionEnum]) -> list[list[str]]:
    for i, line in enumerate(warehouse_map[1:-1]):
        try:
            agent_position = (i + 1, line.index(AGENT))
            break
        except ValueError:
            continue
    else:
        raise ValueError("Could not find agent position")

    wh = list(warehouse_map)

    def get(yx):
        y, x = yx
        return wh[y][x]

    def move(objects_coordinates: list[tuple[int, int]], direction: DirectionEnum):
        objects = [get(object_coordinates) for object_coordinates in objects_coordinates]

        agent_position_ = None
        assert objects_coordinates

        for i, (y, x) in enumerate(objects_coordinates):
            obj = objects[i]

            if i == len(objects) - 1:
                new_pos = go(direction, (y, x))
            else:
                new_pos = objects_coordinates[i + 1]

            ny, nx = new_pos

            if i == 0:
                wh[y][x] = SPACE
                agent_position_ = new_pos

            wh[ny][nx] = obj

        return agent_position_

    for action_idx, direction in enumerate(directions):
        direction_yx = DIRECTIONS_BY_ENUM[direction]

        target_pos = go(direction_yx, agent_position)
        target = get(target_pos)

        boxes_to_move = []
        while target == BOX:
            boxes_to_move.append(target_pos)

            target_pos = go(direction_yx, target_pos)
            target = get(target_pos)

        if target == SPACE:
            agent_position = move([agent_position] + boxes_to_move, direction)
        elif target == WALL:
            pass
        else:
            raise ValueError(f"unknown {target=} at {action_idx=}!")

    return wh


def get_sum_of_boxes_gps_coordinates(warehouse_map: list[list[str]]) -> int:
    gps_list = []
    for y, line in enumerate(warehouse_map):
        for x, value in enumerate(line):
            if value == BOX:
                gps = 100 * y + x
                gps_list.append(gps)
    return sum(gps_list)


def task(inp: list[str]) -> int:
    warehouse_map, directions = _parse_input(inp, 1)

    updated_warehouse_map = make_moves(warehouse_map, directions)

    return get_sum_of_boxes_gps_coordinates(updated_warehouse_map)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, **kw)


if __name__ == "__main__":
    test(task1, 10092)
    test(task1, 2028, test_part=2)
    run(task1)

    # test(task2)
