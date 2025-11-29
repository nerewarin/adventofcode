"""--- Day 12: Hill Climbing Algorithm ---
https://adventofcode.com/2022/day/12
"""

from src.utils.pathfinding import manhattan_distance
from src.utils.test_and_run import run, test


class Node:
    """A node class for A* Pathfinding"""

    def __init__(
        self, parent=None, position=None, signal=0, closest_signal_in=0, closest_signal_pos_from_end=0, path=None
    ):
        self.parent = parent
        self.position = position
        self.signal = signal
        self.closest_signal_in = closest_signal_in
        self.closest_signal_pos_from_end = closest_signal_pos_from_end
        self.path = path or []

        """
        F is the total cost of the node.
        G is the distance between the current node and the start node.
        H is the heuristic — estimated distance from the current node to the end node.
        """
        self.f = 0
        self.g = 0
        self.h = 0

    def __str__(self):
        return f"({self.position}) f={self.f} signal={self.signal!r}"

    def __repr__(self):
        base = self.__str__()

        parent_str = ""
        c = 1
        parent = self.parent
        while parent:
            parent_str += "\n" + "\t" * c + f"parent = {parent!r}"
            # c += 1
            # parent = parent.parent
            parent = None

        return base + parent_str

    def __eq__(self, other):
        return self.position == other.position


def _parse_puzzle(inp):
    maze = []
    start = None
    end = None
    for row_num, line in enumerate(inp):
        row = []
        for col_num, elm in enumerate(line):
            if elm == "S":
                elm = "a"
                start = (row_num, col_num)
            elif elm == "E":
                elm = "z"
                end = (row_num, col_num)

            signal = ord(elm) - 96  # a = 1

            row.append(signal)

        maze.append(row)

    return maze, start, end


def get_signal(maze, pos):
    return maze[pos[0]][pos[1]]


def astar(
    maze,
    start,
    end,
    allow_diagonal=True,
    signal_limit=None,
    get_signal_func=None,
    max_blocks_in_a_single_direction=None,
):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    if get_signal_func is None:
        get_signal_func = get_signal

    # Create start and end node
    start_node = Node(None, start, signal=get_signal_func(maze, start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end, signal=get_signal_func(maze, end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = set()

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    cycle = 0
    while len(open_list) > 0:
        # Get the current node
        open_list = sorted(open_list, key=lambda node: (-node.signal, node.f))
        current_node = open_list[0]

        # for index, item in enumerate(open_list):
        #     if item.f < current_node.f:
        #         current_node = item
        #         current_index = index

        cycle, maze_str = print_maze(closed_list, current_node, cycle, maze, open_list)

        # Pop current off open list, add to closed list
        # open_list.pop(current_index)
        open_list.remove(current_node)
        closed_list.add(current_node.position)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent

            print()
            for i, step in enumerate(path[::-1]):
                maze_str[step[0]][step[1]] = str(i % 10)
            for i, line in enumerate(maze_str):
                s = "".join(line)
                print(s)
            # print(f"{cycle=}: {current_node=}")
            print(cycle, ":", current_node)
            print()

            # print("Printing path")
            # for step in path[::-1]:
            #     print(step[0] +1 , step[1] + 1)
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        adjacent_squares = [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ]
        if allow_diagonal:
            adjacent_squares += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for new_position in adjacent_squares:  # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if (
                node_position[0] > (len(maze) - 1)
                or node_position[0] < 0
                or node_position[1] > (len(maze[len(maze) - 1]) - 1)
                or node_position[1] < 0
            ):
                continue

            if max_blocks_in_a_single_direction:
                _last = current_node.path[: max_blocks_in_a_single_direction - 1] + [new_position]
                if len(_last) == max_blocks_in_a_single_direction and set(_last) != 1:
                    print(f"reached max_blocks_in_a_single_direction: {_last}")
                    continue

            # Make sure walkable terrain
            # if maze[node_position[0]][node_position[1]] != 0:
            #     continue
            new_signal = get_signal_func(maze, node_position)
            if signal_limit:
                if new_signal > (current_node.signal + signal_limit):
                    continue

            # Create new node

            next_signal_value = new_signal + 1
            closest_signal_in = float("infinity")
            closest_signal_pos = None
            for row_num_, row_ in enumerate(maze):
                for col_num_, v_ in enumerate(row_):
                    if v_ == next_signal_value:
                        closest_signal_cand = abs(node_position[0] - row_num_) + abs(node_position[1] - col_num_)
                        if closest_signal_cand < closest_signal_in:
                            closest_signal_in = closest_signal_cand
                            closest_signal_pos = row_num_, col_num_
            if not closest_signal_pos:
                closest_signal_pos = end
            closest_signal_pos_from_end = manhattan_distance(closest_signal_pos, end)

            new_node = Node(
                current_node,
                node_position,
                new_signal,
                closest_signal_in,
                closest_signal_pos_from_end,
                path=current_node.path + [new_position],
            )

            # Append
            children.append(new_node)

        # Loop through children
        filtered_children = [child for child in children if child.position not in closed_list]

        for child in filtered_children:
            # Create the f, g, and h values
            child.g = current_node.g + 1
            h = 1.1 * child.closest_signal_in + child.closest_signal_pos_from_end
            child.h = h
            # child.h =  * child.signal
            # if cycle < 2000:
            #     child.f = child.g + 1.5 * (child.h - 10000000 * child.signal)
            # else:
            # TODO just go to max signal closest to the end you have!
            # TODO no... prefer step that is closest to the NEXT signal
            # child.f = 0.1 * child.g + child.h * 1 - 1 * child.signal
            # child.f = child.g + child.h * 2 + 2 * child.closest_signal_in
            child.f = child.g + child.h
            # надо позицию до ближайшего сигнала + 1 и от него!

            # child.f = child.g + child.closest_signal_in * 2 # 494 / 490
            # if child.signal - current_node.signal == 1:
            #     child.f -= 10000
            if child.position in ((16 - 1 - 1, 88 - 1), (16 - 1 + 1, 88 - 1)):
                pass  # for debug

            # Child is already in the open list
            skip = False
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    skip = True
                    break
            if skip:
                continue

            # Add the child to the open list
            open_list.append(child)

        raise ValueError


def print_maze(closed_list, current_node, cycle, maze, open_list):
    maze_str = [[chr(x + 96) for x in list(line)] for line in maze]
    for node_ in open_list:
        # maze_str[node_.position[0]][node_.position[1]] = " "
        maze_str[node_.position[0]][node_.position[1]]
        # if ord('a') <= ord(v) <= ord('z'):
        #     v = chr(ord(v) - 32)
        #     maze_str[node_.position[0]][node_.position[1]] = v
        maze_str[node_.position[0]][node_.position[1]] = " "
    for position in closed_list:
        maze_str[position[0]][position[1]] = "."
    maze_str[current_node.position[0]][current_node.position[1]] = "*"
    cycle += 1
    if not cycle % 1000:
        print()
        for i, line in enumerate(maze_str):
            s = "".join(line)
            print(s)
        # print(f"{cycle=}: {current_node=}")
        print(cycle, ": current_node", current_node)
        print()
    return cycle, maze_str


def task(inp):
    """
    hill_climbing_algorithm
    """
    maze, start, end = _parse_puzzle(inp)

    path = astar(maze, start, end, allow_diagonal=False, signal_limit=1)

    # print(path)
    print(f"res: {len(path) - 1}")

    return len(path) - 1


if __name__ == "__main__":
    # part 1
    test(task, expected=31)
    res = run(task)
    assert res == 490, f"res {res} != {490}"

    # part 2
    # test(task, mode=1, expected=29)
    # run(task, mode=1)
    print(res - 2)  # solved with eyes
