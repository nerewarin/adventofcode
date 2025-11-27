import re

from tqdm import tqdm

from src.utils.directions_orthogonal import DIRECTION_SYMBOLS, DIRECTIONS_BY_ENUM, DirectionEnum, go, is_horizontal
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
        assert objects_coordinates

        new_objects_coordinates = set()

        agent_position_ = None
        for i, (y, x) in enumerate(objects_coordinates):
            obj = objects[i]

            new_pos = go(direction, (y, x))

            if obj == AGENT:
                agent_position_ = new_pos

            new_objects_coordinates.add(new_pos)

            ny, nx = new_pos
            wh[ny][nx] = obj

        for i, (y, x) in enumerate(objects_coordinates):
            if (y, x) not in new_objects_coordinates:
                wh[y][x] = SPACE

        return agent_position_

    for action_idx, direction in tqdm(enumerate(directions), desc="Processing moves...", total=len(directions)):
        assert get(agent_position) == AGENT, f"Missing agent at {action_idx=}"
        _logger.debug(f"Processing {action_idx=}")
        boxes_to_move = []
        boxes_heights = {}

        direction_yx = DIRECTIONS_BY_ENUM[direction]
        target_pos = go(direction_yx, agent_position)
        target = get(target_pos)
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
                another_part_pos = go(DirectionEnum.right, target_pos)
                assert get(another_part_pos) == BOX_RIGHT, get(another_part_pos)

                if is_horizontal(direction):
                    boxes_heights[0] = 2
                else:
                    boxes_heights[1] = 1
            else:
                another_part_pos = go(DirectionEnum.left, target_pos)
                assert get(another_part_pos) == BOX_LEFT, (
                    f"must be BOX_LEFT at {another_part_pos} at {action_idx=} but {get(another_part_pos)} returned"
                )

                if is_horizontal(direction):
                    boxes_heights[0] = 2
                else:
                    boxes_heights[-1] = 1

            boxes_to_move.extend([target_pos, another_part_pos])
        elif target == SPACE:
            agent_position = move([agent_position] + boxes_to_move, direction)
            continue
        elif target == WALL:
            continue

        # like {-1: 2, 0: 1} means we move 2 boxes parts and left side has height 2 and right in front of us = 1

        def fringe_reaches_obstacle(boxes_heights):
            for x_shift, height in boxes_heights.items():
                agent_shifted_by_x = agent_position[0], agent_position[1] + x_shift
                pos_to_check = go([direction] * (height + 1), agent_shifted_by_x)
                if get(pos_to_check) == WALL:
                    return True
            return False

        max_height = 1
        while not fringe_reaches_obstacle(boxes_heights):
            # to this point, we check fringe (top layer) has no obstacle
            # cases left: here is a new box part there or spaces everywhere
            max_height += 1

            spaces_everywhere = True
            boxes_heights_copy = boxes_heights.copy()
            for x_shift, height in boxes_heights_copy.items():
                # virtual position shifted by x_shift
                agent_shifted_by_x = agent_position[0], agent_position[1] + x_shift
                pos_to_check = go([direction] * (height + 1), agent_shifted_by_x)

                target = get(pos_to_check)

                # move duplicated block to a func
                if target in (BOX_LEFT, BOX_RIGHT):
                    spaces_everywhere = False
                    if target == BOX_LEFT:
                        another_part_pos = go(DirectionEnum.right, pos_to_check)
                        assert get(another_part_pos) == BOX_RIGHT, get(another_part_pos)
                        if another_part_pos in boxes_to_move:
                            ...
                            # already counted
                            continue
                        elif is_horizontal(direction):
                            boxes_heights[x_shift] += 2
                        else:
                            boxes_heights[x_shift] += 1
                            boxes_heights[x_shift + 1] = max_height  # TODO what if max_height already = 2?

                    else:
                        another_part_pos = go(DirectionEnum.left, pos_to_check)
                        if another_part_pos in boxes_to_move:
                            ...
                            # already counted
                            continue
                        elif is_horizontal(direction):
                            boxes_heights[x_shift] += 2
                        else:
                            boxes_heights[x_shift] += 1
                            boxes_heights[x_shift - 1] = max_height

                    boxes_to_move.extend([pos_to_check, another_part_pos])
                if target == SPACE:
                    continue
                elif target == WALL:
                    spaces_everywhere = False
                    # one wall is enough to stop checking, next we must exit loop "while not fringe_reaches_obstacle"
                    break

            if spaces_everywhere:
                agent_position = move([agent_position] + boxes_to_move, direction)
                break
    return wh


def get_sum_of_boxes_gps_coordinates(warehouse_map: list[list[str]]) -> int:
    gps_list = []
    for y, line in enumerate(warehouse_map):
        for x, value in enumerate(line):
            objects_to_check = (BOX, BOX_LEFT)

            # if x >= mid_x:
            #     objects_to_check = (BOX, BOX_RIGHT)
            # else:
            #     objects_to_check = (BOX, BOX_LEFT)

            if value in objects_to_check:
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
    run(task2)  # 1509780
