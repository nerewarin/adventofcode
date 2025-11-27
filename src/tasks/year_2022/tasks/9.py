"""--- Day 9: Rope Bridge ---
https://adventofcode.com/2022/day/9
"""

from src.utils.test_and_run import run, test


class Position:
    def __init__(self, x=0, y=0, label=None):
        self.x, self.y = x, y
        self.label = label

    @property
    def pos(self) -> tuple[int, int]:
        return self.x, self.y

    def __repr__(self):
        if self.label:
            return self.label

        if not self.x and not self.y:
            return "ROOT"

        return f"({self.x}, {self.y})"

    def go(self, other):
        self.x += other.x
        self.y += other.y

    def _is_close(self, other):
        return abs(self.x - other.x) < 2 and abs(self.y - other.y) < 2

    def follow(self, other):
        if self._is_close(other):
            return self.pos

        diff = Position(
            max(-1, min(other.x - self.x, 1)),
            max(-1, min(other.y - self.y, 1)),
        )

        self.x += diff.x
        self.y += diff.y

        return self.x, self.y


_DIRECTIONS = {
    "L": Position(-1, 0, "LEFT"),
    "R": Position(1, 0, "RIGHT"),
    "U": Position(0, 1, "UP"),
    "D": Position(0, -1, "DOWN"),
}


class Rope:
    def __init__(self, tails):
        self.head: Position = Position()
        self.tails = [Position() for t in range(tails)]
        self.visited = {self.tails[-1].pos}

    def go(self, cmd):
        raw_direction, steps = cmd.split(" ")
        direction = _DIRECTIONS[raw_direction]

        for step in range(int(steps)):
            self.head.go(direction)

            heads = [self.head, *self.tails[:-1]]
            for x, tail in enumerate(self.tails):
                tail.follow(heads[x])

            self.visited.add(self.tails[-1].pos)


class Steps:
    def __init__(self, start: Position, direction: str, steps: int):
        self.start = start
        self.direction = direction
        self.steps = steps


def rope_bridge(inp, tails=1):
    rope = Rope(tails=tails)

    for line in inp:
        rope.go(line)

    return len(rope.visited)


if __name__ == "__main__":
    task = rope_bridge

    # part 1
    test(task, expected=13)
    run(task)

    # part 2
    test(task, tails=9, expected=1)
    run(task, tails=9)
