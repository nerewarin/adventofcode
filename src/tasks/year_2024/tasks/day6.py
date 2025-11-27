import collections
import copy

from src.utils.test_and_run import run

OBSTACLE = "O"
WALL = "#"
TURN = "+"

_DIRECTIONS = {
    "^": (-1, 0),  # up
    ">": (0, 1),  # right
    "v": (1, 0),  # down
    "<": (0, -1),  # left
}

TURN_RIGHT = {"^": ">", ">": "v", "v": "<", "<": "^"}

PATH_TO_SYMBOL = {
    "^": "|",
    ">": "-",
    "v": "|",
    "<": "-",
    TURN: TURN,
}


def find_start(grid):
    """Find starting position and direction of the guard"""
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell in _DIRECTIONS:
                return i, j, cell
    raise ValueError("No starting position found")


def is_valid_position(pos, grid):
    """Check if position is within grid bounds"""
    return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])


def simulate_guard_path(data):
    # Convert input to grid
    grid = [list(line) for line in data]

    # Find start position
    row, col, direction = find_start(grid)
    visited = {(row, col)}
    path = []

    while True:
        # Check position in front
        dr, dc = _DIRECTIONS[direction]
        next_row, next_col = row + dr, col + dc

        # If position is invalid or has obstacle, turn right
        if not is_valid_position((next_row, next_col), grid) or grid[next_row][next_col] in (WALL, OBSTACLE):
            direction = TURN_RIGHT[direction]
            continue

        path.append((row, col, direction))

        # Move forward
        row, col = next_row, next_col
        visited.add((row, col))

        # Check if guard has left the map
        if not is_valid_position((row + dr, col + dc), grid):
            break

    return len(visited), path


def part1(data, return_path=False):
    res = simulate_guard_path(data)
    if return_path:
        # return path up to the very final move
        return res
    return res[0]


def can_get_stuck(grid, start_row, start_col, direction, possible_positions):
    """Check if the guard can get stuck in a loop with an obstruction placed."""
    visited = set()
    row, col = start_row, start_col
    path = []

    while True:
        if (row, col, direction) in visited:
            return True, path  # Guard is stuck in a loop

        visited.add((row, col, direction))

        dr, dc = _DIRECTIONS[direction]
        next_row, next_col = row + dr, col + dc

        # If position is invalid or has an obstacle, turn right
        if not is_valid_position((next_row, next_col), grid) or grid[next_row][next_col] in (WALL, OBSTACLE):
            direction = TURN_RIGHT[direction]
            path.append((row, col, direction))

            dr, dc = _DIRECTIONS[direction]
            next_row, next_col = row + dr, col + dc
            row, col = next_row, next_col

            continue

        path.append((row, col, direction))

        # Move forward
        row, col = next_row, next_col

        # Check if guard has left the map
        if not is_valid_position((row + dr, col + dc), grid):
            break

    return False, path  # Guard did not get stuck


def display(grid, path: list[str], ind):
    print(f"Path #{ind} found:")
    grid_to_show = copy.deepcopy(grid)

    visited_to_direction = collections.defaultdict(set)

    for i, (row, col, direction) in enumerate(path):
        if i > 0 and direction != path[i - 1][-1]:
            symbol = TURN
        elif (row, col) in visited_to_direction and (
            PATH_TO_SYMBOL[direction] not in visited_to_direction[(row, col)]
            or TURN in visited_to_direction[(row, col)]
        ):
            symbol = TURN
        else:
            symbol = PATH_TO_SYMBOL[direction]

        visited_to_direction[(row, col)].add(symbol)

        grid_to_show[row][col] = symbol

    # do not override start
    row, col, direction = path[0]
    grid_to_show[row][col] = direction

    for row in grid_to_show:
        print("".join(row))


def part2(data, debug=False):
    """
    ....#.....
    ....+---+#
    ....|...|.
    ..#.|...|.
    ..+-|--#|.
    ..|.|.|.|.
    .O+-^---+.
    ..|...|.#.
    ##+---+...
    ......##..


    so we need to consider every position of initial (open) path.

    for every position consider place obstacle if:
        to the right of it there is an obstacle;
            to the right of that obstacle there is another obstacle
               ...
                    and this is in a while until we either reach one of our (row, col, direction) visited
                    (we update this list every time virtually on our way) -> True,
                    or we get out of the bounds -> False


    ....#.....
    ....+---+#
    ....|...|.
    ..#.|...|.
    ..+-|--#|.
    ..|.|.|.|.
    .O+-^---+.
    .....#|...
    .....+|+#.
    .....|||#.
    ##..#+++..
    ......##..


    seems like we also need to cache these "good" obstacles.
    if our algo will take too much time we can us them to first check if our obstacle is on the same line with cached,
    and we have no other walls between them. that means this is automatically an answer too.
    """
    grid = [list(line) for line in data]

    _, initial_path = part1(grid, True)

    possible_positions = 0
    obstacles_cache = set()

    for step, (row, col, direction) in enumerate(initial_path):
        # Check all adjacent positions for potential obstructions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if abs(dr) + abs(dc) == 1:  # Only check orthogonal positions
                    new_row, new_col = row + dr, col + dc

                    if (new_row, new_col) in obstacles_cache:
                        continue
                    obstacles_cache.add((new_row, new_col))

                    if is_valid_position((new_row, new_col), grid) and (new_row, new_col) != (row, col):
                        if grid[new_row][new_col] == WALL:
                            continue
                        # Temporarily place an obstruction
                        original_cell = grid[new_row][new_col]
                        grid[new_row][new_col] = OBSTACLE  # Place obstruction

                        is_stuck, path = can_get_stuck(grid, row, col, direction, possible_positions)
                        if is_stuck:
                            full_path = initial_path[:step] + path
                            if debug:
                                display(grid, full_path, possible_positions)

                            possible_positions += 1
                        grid[new_row][new_col] = original_cell  # Restore original cell

    return possible_positions


# Update the main block to include part2
if __name__ == "__main__":
    # test(part1, 41)
    # run(part1)
    # test(part2, 6, debug=True)
    res2 = run(part2)
    assert res2 == 1516
