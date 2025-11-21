"""
--- Day 22: Sand Slabs ---
https://adventofcode.com/2023/day/22
"""
import collections
import dataclasses
from collections import defaultdict

from src.utils.input_formatters import cast_2d_list_elements
from src.utils.position_search_problem import PositionSearchProblem
from src.utils.test_and_run import run, test


@dataclasses.dataclass
class Brick:
    start: list[int]
    end: list[int]
    name: str

    @classmethod
    def from_line(cls, line, line_num):
        return cls(*[[int(s) for s in side.split(",")] for side in line.split("~")], chr(65 + line_num))

    @property
    def x0(self):
        return self.start[0]

    @property
    def x1(self):
        return self.end[0]

    @property
    def y0(self):
        return self.start[1]

    @property
    def y1(self):
        return self.end[1]

    @property
    def z0(self):
        return self.start[2]

    @property
    def z1(self):
        return self.end[2]

    @property
    def shape(self):
        return [
            abs(self.end[i] - self.start[i] + 1) for i in range(3)
        ]

    @property
    def length(self):
        return max(self.shape)

    def move_bot(self, new_start_z):
        height = self.end[2] - self.start[2]
        self.start[2] = new_start_z
        self.end[2] = new_start_z + height

    def is_connected_to(self, brick):
        stacked_x = False
        if self.x0 <= brick.x0 <= self.x1:
            stacked_x = True
        elif self.x0 <= brick.x1 <= self.x1:
            stacked_x = True
        elif brick.x0 <= self.x0 <= brick.x1:
            stacked_x = True
        elif brick.x0 <= self.x1 <= brick.x1:
            stacked_x = True
        if not stacked_x:
            return False

        if self.y0 <= brick.y0 <= self.y1:
            return True
        elif self.y0 <= brick.y1 <= self.y1:
            return True
        elif brick.y0 <= self.y0 <= brick.y1:
            return True
        elif brick.y0 <= self.y1 <= brick.y1:
            return True
        return False


class SandSlabs:
    def __init__(self, inp):
        self._world_shape = None
        self.bricks = self._parse_input(inp)
        self._update_world_shape()

        self._print_world()

        self._land_bricks()
        self._update_world_shape()

        print("landed:")
        self._print_world()

    def _land_bricks(self):
        sorted_by_z0 = sorted(self.bricks, key=lambda b: b.z0)
        for i, brick in enumerate(sorted_by_z0):
            if brick.z0 == 1:
                continue

            new_start_z = 1
            bricks_below = list(reversed(sorted_by_z0[:i]))
            for brick_below in bricks_below:
                if brick.is_connected_to(brick_below):
                    new_start_z = brick_below.z1 + 1
                    break

            if brick.z0 != new_start_z:
                brick.move_bot(new_start_z)

    def _print_world(self):
        for i in range(2):
            lines = []
            ax_shape = self._world_shape[i]
            if i == 0:
                lines.append(" " * (ax_shape // 2) + "x")
            else:
                lines.append(" " * (ax_shape // 2) + "y")

            lines.append("".join([str(x) for x in range(ax_shape + 1)]))

            az_shape = self._world_shape[-1]
            for z in reversed(range(az_shape + 1)):
                line = ""
                for x in range(ax_shape + 1):
                    for brick in self.bricks:
                        if brick.start[i] <= x <= brick.end[i] and brick.z0 <= z <= brick.z1:
                            line += brick.name
                            break
                    else:
                        line += "."

                line += f" {z}"
                lines.append(line)

            for line in lines:
                line = line.replace("...", '._.')
                print(line)
            print("="*100)

    def _update_world_shape(self):
        self._world_shape = [
            max(brick.end[i] for brick in self.bricks)
            for i in range(3)
        ]

    @staticmethod
    def _parse_input(inp):
        return [Brick.from_line(line, line_num) for line_num, line in enumerate(inp)]

    def get_number_of_bricks_for_be_disintegrated_safely(self):
        brick_name2brick = {
            brick.name: brick for brick in self.bricks
        }

        brick_name2supports = defaultdict(list)
        brick_name2supported_by = defaultdict(list)

        sorted_by_z0 = sorted(self.bricks, key=lambda brick: brick.z0)
        sorted_by_z1 = sorted(self.bricks, key=lambda brick: brick.z1)

        for i, brick in enumerate(sorted_by_z0):
            name = brick.name
            z1 = brick.z1
            top_floor = z1 + 1

            # find all bricks above
            bricks_above = list(sorted_by_z0[i + 1:])
            for brick_above in bricks_above:
                brick_above_z0 = brick_above.z0
                if brick_above_z0 > top_floor:
                    break

                if brick.is_connected_to(brick_above):
                    brick_name2supports[name].append(brick_above.name)

            # find all bricks below
            bot_floor = brick.z0 - 1
            if bot_floor == 0:
                continue

            bricks_below = list(reversed(sorted_by_z1[:i]))
            for brick_below in bricks_below:
                brick_below_z1 = brick_below.z1
                if brick_below_z1 < bot_floor:
                    break

                if brick.is_connected_to(brick_below):
                    brick_name2supported_by[name].append(brick_below.name)

        res = []
        for brick in self.bricks:
            name = brick.name
            supports = brick_name2supports.get(name)
            if not supports:
                res.append(name)
                continue

            could_be_removed = True
            for supported in supports:
                if len(brick_name2supported_by[supported]) == 1:
                    could_be_removed = False
                    break

            if could_be_removed:
                res.append(name)
                continue

        return len(res)


def sand_slabs(inp):
    return SandSlabs(inp).get_number_of_bricks_for_be_disintegrated_safely()


if __name__ == "__main__":
    # test(sand_slabs, expected=5)
    assert run(sand_slabs) < 1040

