# from math lessons:
#
# y
# ^
# |
# |
# |
# .----> x
#
# from computer history (first display)
# 0...-> v
# ---->
# ...... >    ..so
#
# .----> x
# |
# |
# |
# v
# y
DIRECTIONS = [
    # y, x
    (0, -1),  # left
    (-1, 0),  # down
    (0, 1),  # right
    (1, 0),  # top
]


def out_of_borders(y, x, grid):
    max_y = len(grid)
    if y < 0 or y >= max_y:
        return True

    max_x = len(grid[y])
    if x < 0 or x >= max_x:
        return True

    return False
