import os

from src.utils.test_and_run import test

DEBUG = False


def find_xmas_occurrences(grid: list[str]) -> int:
    height = len(grid)
    width = len(grid[0])
    count = 0

    # Helper function to check if coordinates are within grid bounds
    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    # Check all possible directions for "XMAS"
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1)]
    target = "XMAS"
    all_coordinates = []
    for y in range(height):
        for x in range(width):
            if grid[y][x] != "X":
                continue

            # Try each direction
            for dx, dy in directions:
                coordinates = []
                valid = True
                for i in range(len(target)):
                    new_x = x + i * dx
                    new_y = y + i * dy
                    if not in_bounds(new_x, new_y) or grid[new_y][new_x] != target[i]:
                        valid = False
                        break
                    else:
                        point = (new_x, new_y)
                        # if point not in coordinates:
                        coordinates.append(point)
                if valid:
                    count += 1
                    all_coordinates.append(coordinates)

    if DEBUG:
        for i, coordinates in enumerate(all_coordinates):
            print(f"==== {i} ====")
            grid = [[" "] * width for _ in range(height)]
            for i, (x, y) in enumerate(coordinates):
                grid[y][x] = target[i]
            for row in grid:
                print(" ".join(row))

    return count


def find_xmas_x_pattern(grid: list[str]) -> int:
    height = len(grid)
    width = len(grid[0])
    count = 0

    def get(x, y):
        for dim, bound in (
            (x, width),
            (y, height),
        ):
            if dim < 0:
                return None
            if dim >= bound:
                return None

        return grid[y][x]

    # For each center position that could be the middle of an X
    for y in range(height):
        for x in range(width):
            if grid[y][x] != "A":  # Center must be 'A'
                continue

            # Only check one basic X configuration and its variations
            dx1, dy1 = -1, -1  # up-left
            dx2, dy2 = 1, -1  # up-right
            dx3, dy3 = -1, 1  # down-left
            dx4, dy4 = 1, 1  # down-right

            x1, y1 = x + dx1, y + dy1
            x2, y2 = x + dx2, y + dy2
            x3, y3 = x + dx3, y + dy3
            x4, y4 = x + dx4, y + dy4

            values = [
                get(x1, y1),
                get(x2, y2),
                get(x3, y3),
                get(x4, y4),
            ]
            # 2xM top
            if values[0] == "M" and values[1] == "M" and values[2] == "S" and values[3] == "S":
                count += 1
            # 2xM bot
            elif values[0] == "S" and values[1] == "S" and values[2] == "M" and values[3] == "M":
                count += 1
            # 2xM left
            elif values[0] == "M" and values[1] == "S" and values[2] == "M" and values[3] == "S":
                count += 1
            # 2xM right
            elif values[0] == "S" and values[1] == "M" and values[2] == "S" and values[3] == "M":
                count += 1

            # # M top left and bot right
            # if values[0] == "M" and values[1] == "S" and values[2] == "S" and values[3] == "M":
            #     count += 1
            #     continue
            # # M top right and bot left
            # if values[0] == "S" and values[1] == "M" and values[2] == "M" and values[3] == "S":
            #     count += 1
            #     continue

    return count


def solve_part1(input_data: str | list[str]) -> int:
    if isinstance(input_data, list):
        grid = input_data
    else:
        grid = [line.strip() for line in input_data.splitlines() if line.strip()]
    return find_xmas_occurrences(grid)


def solve_part2(input_data: str | list[str]) -> int:
    if isinstance(input_data, list):
        grid = input_data
    else:
        grid = [line.strip() for line in input_data.splitlines() if line.strip()]
    return find_xmas_x_pattern(grid)


if __name__ == "__main__":
    # test(solve_part1, expected=18)

    # Get the absolute path to the input file
    test(solve_part2, test_part=2, expected=9)

    current_dir = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(current_dir, "inputs", "4", "run")

    # # Read the input file
    with open(input_file) as f:
        input_data = f.read()

    res1 = solve_part1(input_data)
    print(f"Part 1: {res1}")
    assert res1 == 2603
    #
    res2 = solve_part2(input_data)
    assert 1900 < res2 < 1968, res2

    print(f"Part 2: {res2}")
