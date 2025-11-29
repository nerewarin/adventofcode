from src.utils.directions import (
    ORTHOGONAL_DIRECTIONS,
    OrthogonalDirectionEnum,
    get_2d_diff,
    get_abs,
    go,
    out_of_borders,
)
from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


def build_fence_to_neighbor(pos, neighbor_pos):
    y, x = pos
    y1, x1 = neighbor_pos

    dy, dx = y1 - y, x1 - x

    human_direction = ORTHOGONAL_DIRECTIONS[(dy, dx)]
    match human_direction:
        case OrthogonalDirectionEnum.left:
            fence_start = go(OrthogonalDirectionEnum.down, pos)
            fence_direction = OrthogonalDirectionEnum.up
        case OrthogonalDirectionEnum.right:
            fence_start = go(OrthogonalDirectionEnum.right, pos)
            fence_direction = OrthogonalDirectionEnum.down
        case OrthogonalDirectionEnum.up:
            fence_start = pos
            fence_direction = OrthogonalDirectionEnum.right
        case OrthogonalDirectionEnum.down:
            fence_start = go([OrthogonalDirectionEnum.down, OrthogonalDirectionEnum.right], pos)
            fence_direction = OrthogonalDirectionEnum.left

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

        for (dy, dx), direction_enum in ORTHOGONAL_DIRECTIONS.items():
            neighbor_y = y + dy
            neighbor_x = x + dx
            neighbor_pos = (neighbor_y, neighbor_x)
            if out_of_borders(neighbor_y, neighbor_x, inp) or (neighbor_value := inp[neighbor_y][neighbor_x]) != val:
                fence = build_fence_to_neighbor(pos, neighbor_pos)
                fences.append(fence)
                continue

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
                fences_copy = list(fences)
                if fences:
                    draw = [["."] + [str(s) for s in row] + ["."] for row in inp]
                    draw = [["."] * len(draw[0])] + draw + [["."] * len(draw[0])]
                    fence = fences[0]

                    _logger.debug(f"{fences=}")

                    while fence is not None:
                        fence_start, fence_end = fence
                        fence_idx = fences.index(fence)

                        fence_direction = get_2d_diff(fence_start, fence_end)
                        if not current_direction:
                            sides += 1
                            draw[fence_start[0] + 1][fence_start[1] + 1] = "i"
                            _logger.debug(f"{sides=}: initital turn at {fence=}")
                            for row in draw:
                                _logger.debug("".join(row))
                        elif get_abs(*fence_direction) != get_abs(*current_direction):
                            sides += 1
                            draw[fence_start[0] + 1][fence_start[1] + 1] = "x"
                            _logger.debug(
                                f"{sides=}: turn detected at {fence=} ({fence_direction=}, {current_direction=})"
                            )
                            for row in draw:
                                _logger.debug("".join(row))
                        else:
                            _logger.debug(f"no turn detecjed at {fence=}")

                        current_direction = fence_direction

                        del fences[fence_idx]

                        next_fence = None
                        valid_candidates = []
                        for fence_candidate in fences:
                            if fence_candidate[0] == fence_end:
                                valid_candidates.append(fence_candidate)

                        if valid_candidates:
                            if len(valid_candidates) == 1:
                                next_fence = valid_candidates[0]
                            else:
                                # get index closer to deleted fence (workaround. if failed, we need use linked lists)
                                next_fence = None
                                distance = float("inf")
                                for fence_candidate in valid_candidates:
                                    candidate_idx = fences.index(fence_candidate)
                                    candidate_distance = abs(candidate_idx - fence_idx)
                                    if candidate_distance < distance:
                                        distance = candidate_distance
                                        next_fence = fence_candidate

                        else:
                            if get_2d_diff(*fence, absolute=True) == get_2d_diff(*fences_copy[0], absolute=True):
                                sides -= 1
                                _logger.debug(
                                    f"{sides=}. decremented cause started with +1 from midle and now returned"
                                    f" there with no turn "
                                )

                            if fences:
                                next_fence = fences[0]
                                current_direction = None
                                fences_copy = list(fences)

                        fence = next_fence

            region_key = (val, pos)
            regions[region_key] = (square, fences_amount, sides)

    res = 0
    for region_key, v in regions.items():
        square, fences, sides = v
        if task_num == 1:
            price = square * fences
        elif task_num == 2:
            price = square * sides
            _logger.debug(f"{region_key}: {square=} * {sides=} = {price=}")
        else:
            raise NotImplementedError(f"task_num = {task_num} is not implemented")

        res += price
    _logger.debug(f"task finished with total price = {res}")
    return res


def task1(inp):
    return task(inp, 1)


def task2(inp):
    return task(inp, 2)


if __name__ == "__main__":
    test(task1, 140)
    test(task1, 772, test_part=2)
    test(task1, 1930, test_part=3)
    run(task1)

    test(task2, 80)
    test(task2, 436, test_part=2)
    test(task2, 236, test_part=4)
    test(task2, 368, test_part=5)
    run(task2)
