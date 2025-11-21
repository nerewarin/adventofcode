"""--- Day 14: Regolith Reservoir ---
https://adventofcode.com/2022/day/14
"""

from src.utils.test_and_run import run, test


def _parse_puzzle(inp):
    rock_segments = []
    all_rocks = []
    for line in inp:
        # 498,4 -> 498,6 -> 496,6
        path = [tuple(int(x.strip()) for x in elm.split(",")) for elm in line.split("->")]
        start = path[0]
        rocks = [start]
        for point in path[1:]:
            diff = point[0] - start[0] + (point[1] - start[1])
            is_pos = diff > 0

            if start[0] == point[0]:
                for i in range(abs(diff)):
                    if is_pos:
                        rocks.append((start[0], start[1] + i + 1))
                    else:
                        rocks.append((start[0], start[1] - i - 1))
            else:
                for i in range(abs(diff)):
                    if is_pos:
                        rocks.append((start[0] + i + 1, start[1]))
                    else:
                        rocks.append((start[0] - i - 1, start[1]))

            start = point

        rock_segments.append(rocks)
        all_rocks.extend(rocks)

    return set(all_rocks)


def down(pos):
    return pos[0], pos[1] + 1


def down_left(pos):
    return pos[0] - 1, pos[1] + 1


def down_right(pos):
    return pos[0] + 1, pos[1] + 1


class Sand:
    pouring_point = (500, 0)

    def __init__(self, closed, bottom_lvl, bottom):
        self.closed = closed
        self.bottom_lvl = bottom_lvl
        self.bottom = bottom

    def drop(self):
        sand_pos = self.pouring_point
        while True:
            for move in (
                down,
                down_left,
                down_right,
            ):
                new_sand_pos = move(sand_pos)
                if new_sand_pos in self.closed or (self.bottom and new_sand_pos[1] == self.bottom_lvl):
                    # reaching the floor in one direction
                    if sand_pos == self.pouring_point:
                        # no way from start
                        continue
                    continue
                else:
                    sand_pos = new_sand_pos
                    break
            else:
                # reaching the floor in all the direction
                return sand_pos

            if sand_pos[1] == self.bottom_lvl:
                return False


def task(inp, bottom=None):
    """
    Using your scan, simulate the falling sand. How many units of sand come to rest before sand starts flowing into the abyss below?
    """
    all_rocks = _parse_puzzle(inp)

    sand_count = 0

    bottom_lvl = max(x[1] for x in all_rocks) + 2

    dropped_sand_position = True
    while dropped_sand_position:
        dropped_sand_position = Sand(all_rocks, bottom_lvl, bottom=bottom).drop()
        if dropped_sand_position:
            all_rocks.add(dropped_sand_position)
            sand_count += 1

        if dropped_sand_position == Sand.pouring_point:
            break

    return sand_count


if __name__ == "__main__":
    # # part 1
    test(task, expected=24)
    res = run(task)
    assert res == 578, f"{res=} but it should be 578 for my input!"

    # part 2
    test(task, bottom="floor", expected=93)
    res = run(task, bottom="floor")
    assert res == 24377, f"{res=} but it should be 24377 for my input!"
