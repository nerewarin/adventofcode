import heapq
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        entry = (priority, self.count, item)
        # entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        #  (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, signal=0, closest_signal_in=0, closest_signal_pos_from_end=0):
        self.parent = parent
        self.position = position
        self.signal = signal
        self.closest_signal_in = closest_signal_in
        self.closest_signal_pos_from_end = closest_signal_pos_from_end

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
            parent_str += ("\n" + "\t" * c + f"parent = {parent!r}")
            # c += 1
            # parent = parent.parent
            parent = None

        return base + parent_str

    def __eq__(self, other):
        return self.position == other.position


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_signal(maze, pos):
    return maze[pos[0]][pos[1]]


def astar(maze, start, end, allow_diagonal=True, signal_limit = None):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start, signal=get_signal(maze, start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end, signal=get_signal(maze, end))
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

        maze_str = [[chr(x + 96) for x in list(line)] for line in maze]
        for node_ in open_list:
            # maze_str[node_.position[0]][node_.position[1]] = " "
            v = maze_str[node_.position[0]][node_.position[1]]
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
        adjacent_squares = [(-1, 0), (0, 1), (1, 0), (0, -1),]
        if allow_diagonal:
            adjacent_squares += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for new_position in adjacent_squares:  # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            # if maze[node_position[0]][node_position[1]] != 0:
            #     continue
            new_signal = get_signal(maze, node_position)
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
            closest_signal_pos_from_end = dist(closest_signal_pos, end)

            new_node = Node(current_node, node_position, new_signal, closest_signal_in, closest_signal_pos_from_end)

            # Append
            children.append(new_node)

        # Loop through children
        filtered_children = [child for child in children if child.position not in closed_list]
        a = 0

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
            if child.position in ((16 - 1 - 1, 88 - 1),  (16 - 1 + 1, 88 - 1)):
                a = 90 # for debug

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

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state =  problem.getStartState()
    start_successors = problem.getSuccessors(start_state)
    successors = {start_state : start_successors}
    # print "Start:", start_state
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", start_successors

    fringe = PriorityQueue()
    pushed = set([])
    best_cost = {}
    for state, action, cost in start_successors:
        moving = state, action, cost
        score = cost
        fringe.push((state, [action], cost), score)

        best_cost[moving] = score
        pushed.add(moving)
    closed = [start_state]

    while not fringe.isEmpty():
        moving = fringe.pop()
        if problem.isGoalState(moving[0]):
            return moving[1]
        closed.append(moving[0])
        if moving[0] not in successors.keys(): # the main change is to define a dict
            successors[moving[0]] = problem.getSuccessors(moving[0])

        for child in successors[moving[0]]:
            if (child[0] not in closed) and (child not in pushed):
                best_cost[child[0]] = moving[2] + child[2]
                fringe.push( (child[0], moving[1] + [child[1]], moving[2] + child[2]), moving[2] + child[2] )
                pushed.add(child)
    else:
        return []

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    start_successors = problem.getSuccessors(start_state)
    successors = {start_state: start_successors}
    fringe = PriorityQueue()
    pushed = set([])
    best_cost = {}
    for state, action, cost in start_successors:
        moving = state, action, cost
        score = cost + heuristic(state, problem)
        fringe.push((state, [action], cost), score)
        best_cost[state] = score
        pushed.add(moving)
    closed = [start_state]

    while not fringe.isEmpty():
        moving = fringe.pop()
        state, path, cost = moving

        if problem.isGoalState(state):
            return path

        closed.append(state)
        if state not in successors.keys():
            successors[state] = problem.getSuccessors(state)

        for child in successors[state]:
            child_state, child_action, child_cost = child

            full_cost = cost + child_cost
            score = full_cost + heuristic(child_state, problem)

            if (child_state not in closed) and (child not in pushed):
                best_cost[child_state] = score
                fringe.push((child_state, path + [child_action], full_cost), score)
                pushed.add(child)
    else:
        return []


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    start_successors = problem.getSuccessors(start_state)
    fringe = Queue()
    pushed = set([])
    for position, action, dummy_cost in start_successors:
        fringe.push((position, [action]))
        pushed.add(position)

    closed = [start_state]

    while not fringe.isEmpty():
        position, actions = fringe.pop()
        if problem.isGoalState(position):
            return actions
        closed.append(position)
        successors = problem.getSuccessors(position)
        for child_position, child_action, dummy_cost in successors:
            if (child_position not in closed) and (child_position not in pushed):
                fringe.push((child_position, actions + [child_action]))
                pushed.add(child_position)
    else:
        return []


if __name__ == '__main__':
    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)

    assert astar(maze, start, end) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 5), (7, 6)]

    assert astar(maze, start, end, allow_diagonal=False) == [
        (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3),
        (4, 3), (5, 3), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)
    ]
