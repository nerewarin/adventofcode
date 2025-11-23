from src.utils.directions import DIRECTIONS, out_of_borders
from src.utils.test_and_run import run, test


def task1(inp):
    regions = {}
    visited = set()
    current_value = None

    def explore_region(val, pos, square=0, fences=0):
        if pos in visited:
            return square, fences

        visited.add(pos)

        y, x = pos

        square += 1

        for dy, dx in DIRECTIONS:
            neighbor_y = y + dy
            neighbor_x = x + dx
            if out_of_borders(neighbor_y, neighbor_x, inp):
                fences += 1
                continue

            neighbor_value = inp[neighbor_y][neighbor_x]
            if neighbor_value != val:
                fences += 1
                continue

            square, fences = explore_region(neighbor_value, (neighbor_y, neighbor_x), square, fences)

        return square, fences

    for y, row in enumerate(inp):
        for x, val in enumerate(row):
            if current_value is None:
                current_value = val

            pos = y, x
            if pos in visited:
                continue

            square, fences = explore_region(val, pos)
            region_key = (val, pos)
            regions[region_key] = (square, fences)

    res = 0
    for k, v in regions.items():
        square, fences = v
        region_score = square * fences
        res += region_score
    return res


if __name__ == "__main__":
    test(task1, 140)
    test(task1, 772, test_part=2)
    test(task1, 1930, test_part=3)
    run(task1)
