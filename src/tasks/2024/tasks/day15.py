import re

from src.utils.directions_orthogonal import DIRECTION_SYMBOLS, DIRECTIONS_BY_ENUM, DirectionEnum, go
from src.utils.logger import get_logger
from src.utils.test_and_run import test

_logger = get_logger()

_coordinate_pattern = r"(-?\d+)"
_point_pattern = f"{_coordinate_pattern},{_coordinate_pattern}"

_input_rexp = re.compile(rf"p={_point_pattern} v={_point_pattern}")


BOX = "O"
WALL = "#"
AGENT = "@"
SPACE = "."
BOX_LEFT = "["
BOX_RIGHT = "]"


def _parse_input(inp: list[str]) -> tuple[list[list[str]], list[DirectionEnum]]:
    warehouse_map = []
    for i, line in enumerate(inp):
        if not line:
            break
        warehouse_map.append([symbol for symbol in line.strip()])

    directions = []
    for line in inp[i + 1 :]:
        if not line:
            break
        for symbol in line:
            directions.append(DIRECTION_SYMBOLS[symbol])

    return warehouse_map, directions


def _scale_map(warehouse_map: list[list[str]]) -> list[list[str]]:
    new_map = []
    for y, line in enumerate(warehouse_map):
        new_line = []
        for x, value in enumerate(line):
            if value == BOX:
                new_symbols = [BOX_LEFT, BOX_RIGHT]
            elif value == WALL:
                new_symbols = [WALL, WALL]
            elif value == AGENT:
                new_symbols = [AGENT, SPACE]
            elif value == SPACE:
                new_symbols = [SPACE, SPACE]
            else:
                raise ValueError(f"unknown {value=}")
            new_line.extend(new_symbols)
        new_map.append(new_line)
    assert len(set(len(line) for line in warehouse_map)) == 1
    return new_map


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
        # how wide is area to check e.g. considering this construction goes up
        #  ######
        #  [
        #
        #     #
        #  [
        #  [  [  [
        # result must be like this
        #  ######
        #  [
        #
        #  [  #
        #  [  ]
        #  [  [  [

        # extend target
        if target in (BOX_LEFT, BOX_RIGHT):
            boxes_heights = {0: 1}
            if target == BOX_LEFT:
                # TODO handle we met box horizontally
                another_part_pos = go(DirectionEnum.right, target_pos)
                assert get(another_part_pos) == BOX_RIGHT, get(another_part_pos)
                boxes_heights[1] = 1
            else:
                # TODO handle we met box horizontally
                another_part_pos = go(DirectionEnum.left, target_pos)
                assert get(another_part_pos) == BOX_LEFT, get(another_part_pos)
                boxes_heights[-1] = 1
            boxes_to_move.extend([target_pos, another_part_pos])
        if target == SPACE:
            agent_position = move([agent_position] + boxes_to_move, direction)
            continue
        elif target == WALL:
            continue

        # like {-1: 2, 0: 1} means we move 2 boxes parts and left side has height 2 and right in front of us = 1

        def fringe_reaches_obstacle(boxes_heights):
            for axis, height in boxes_heights.items():
                pos_to_check = go([direction] * height, agent_position)
                if get(pos_to_check) == WALL:
                    return True
            return False

        max_height = 1
        while not fringe_reaches_obstacle(boxes_heights):
            # to this point, we check fringe (top layer) has no obstacle
            # cases left: here is a new box part there or spaces everywhere
            max_height += 1

            spaces_everywhere = True
            for axis, height in boxes_heights.items():
                pos_to_check = go([direction] * height, agent_position)
                target = get(pos_to_check)

                # move duplicated block to a func
                if target in (BOX_LEFT, BOX_RIGHT):
                    boxes_heights[axis] += 1
                    spaces_everywhere = False
                    if target == BOX_LEFT:
                        # TODO handle we met box horizontally
                        another_part_pos = go(DirectionEnum.right, target_pos)
                        assert get(another_part_pos) == BOX_RIGHT, get(another_part_pos)
                        boxes_heights[axis + 1] = max_height
                    else:
                        # TODO handle we met box horizontally
                        another_part_pos = go(DirectionEnum.left, target_pos)
                        assert get(another_part_pos) == BOX_LEFT, get(another_part_pos)
                        boxes_heights[axis - 1] = max_height
                    boxes_to_move.extend([target_pos, another_part_pos])
                if target == SPACE:
                    agent_position = move([agent_position] + boxes_to_move, direction)
                    continue
                elif target == WALL:
                    continue

            if spaces_everywhere:
                break

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


def task(inp: list[str], task_num: int = 1) -> int:
    warehouse_map, directions = _parse_input(inp)

    if task_num == 2:
        warehouse_map = _scale_map(warehouse_map)

    updated_warehouse_map = make_moves(warehouse_map, directions)

    return get_sum_of_boxes_gps_coordinates(updated_warehouse_map)


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, **kw):
    return task(inp, task_num=2)


if __name__ == "__main__":
    # test(task1, 10092)
    # test(task1, 2028, test_part=2)
    # run(task1)

    test(task2, 9021)
