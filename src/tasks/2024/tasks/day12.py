from src.utils.directions_orthogonal import DIRECTIONS, DirectionEnum, get_2d_diff, go, out_of_borders
from src.utils.logger import get_logger
from src.utils.test_and_run import test

_logger = get_logger()


def build_fence_to_neighbor(pos, neighbor_pos):
    y, x = pos
    y1, x1 = neighbor_pos

    dy, dx = y1 - y, x1 - x

    human_direction = DIRECTIONS[(dy, dx)]
    match human_direction:
        case DirectionEnum.left:
            fence_start = go(DirectionEnum.down, pos)
            fence_direction = DirectionEnum.up
        case DirectionEnum.right:
            fence_start = go(DirectionEnum.right, pos)
            fence_direction = DirectionEnum.down
        case DirectionEnum.up:
            fence_start = pos
            fence_direction = DirectionEnum.right
        case DirectionEnum.down:
            fence_start = go([DirectionEnum.down, DirectionEnum.right], pos)
            fence_direction = DirectionEnum.left

    fence_end = go(fence_direction, fence_start)
    return fence_start, fence_end


def task(inp, task_num):
    regions = {}
    visited = set()
    current_value = None

    def explore_region(val, pos, square=0, fences=None):
        if not fences:
            fences = []
        if pos in visited:
            return square, fences

        visited.add(pos)

        y, x = pos

        square += 1

        for (dy, dx), direction_enum in DIRECTIONS.items():
            neighbor_y = y + dy
            neighbor_x = x + dx
            neighbor_pos = (neighbor_y, neighbor_x)
            if out_of_borders(neighbor_y, neighbor_x, inp) or (neighbor_value := inp[neighbor_y][neighbor_x]) != val:
                fence = build_fence_to_neighbor(pos, neighbor_pos)
                fences.append(fence)
                continue

            # if (neighbor_y, neighbor_x) in visited:  # TODO wrong
            #     continue
            # if (dy, dx) != direction:
            #     sides += 1

            square, fences = explore_region(
                neighbor_value,
                (neighbor_y, neighbor_x),
                square,
                fences,
            )

        return square, fences

    for y, row in enumerate(inp):
        for x, val in enumerate(row):
            if current_value is None:
                current_value = val

            pos = y, x
            if pos in visited:
                continue

            square, fences = explore_region(val, pos)

            fences_amount = len(fences)
            sides = 0
            if task_num == 2:
                # compute sides as turns on fences
                assert len(fences) == len(set(fences))

                current_direction = None
                if fences:
                    fence = fences[0]

                    while fence is not None:
                        fence_start, fence_end = fence
                        fences.remove(fence)

                        fence_direction = get_2d_diff(fence_start, fence_end)
                        fence_human_direction = DIRECTIONS[fence_direction]

                        if fence_human_direction != current_direction:
                            sides += 1
                        current_direction = fence_human_direction

                        next_fence = None
                        for fence_candidate in fences:
                            if fence_candidate[0] == fence_end:
                                next_fence = fence_candidate
                                break
                        else:
                            if fences:
                                next_fence = fences[0]
                                current_direction = None

                        fence = next_fence

            region_key = (val, pos)
            regions[region_key] = (square, fences_amount, sides)
            _logger.debug(f"{region_key}: {square}, {fences}, {sides}")

    res = 0
    for k, v in regions.items():
        square, fences, sides = v
        if task_num == 1:
            region_score = square * fences
        elif task_num == 2:
            region_score = square * sides
        else:
            raise NotImplementedError(f"task_num = {task_num} is not implemented")

        res += region_score
    return res


def task1(inp):
    return task(inp, 1)


def task2(inp):
    return task(inp, 2)


if __name__ == "__main__":
    # test(task1, 140)
    # test(task1, 772, test_part=2)
    # test(task1, 1930, test_part=3)
    # run(task1)

    test(task2, 80)
    # test(task2, 463, test_part=2)
    # test(task2, 236, test_part=4)
    # test(task2, 368, test_part=5)
    # run(task2)
