"""--- Day 15: Beacon Exclusion Zone ---
https://adventofcode.com/2022/day/15
"""

import dataclasses
import re
from functools import cached_property

from src.utils.pathfinding import manhattan_distance
from src.utils.test_and_run import run, test

_REXP = re.compile(r"Sensor at x=(-*\d+), y=(-*\d+): closest beacon is at x=(-*\d+), y=(-*\d+)")


def _parse_puzzle(inp):
    # map
    m = []

    sensors = []
    for line in inp:
        res = [int(x) for x in _REXP.match(line).groups()]
        beacon = Beacon(*res[2:])
        sensor = Sensor(*res[:2], beacon)

        sensors.append(sensor)

    # for debug only
    if DEBUG:
        f = min
        min_x = f(
            f(s.x for s in sensors),
            f(s.beacon.x for s in sensors),
        )
        min_y = f(
            f(s.y for s in sensors),
            f(s.beacon.y for s in sensors),
        )

        f = max
        max_x = f(
            f(s.x for s in sensors),
            f(s.beacon.x for s in sensors),
        )
        max_y = f(
            f(s.y for s in sensors),
            f(s.beacon.y for s in sensors),
        )
        sensors_pos = [(s.x, s.y) for s in sensors]
        beacons_pos = [(s.beacon.x, s.beacon.y) for s in sensors]

        print(
            min_x,
            "...",
            max_x,
            " / ",
            min_y,
            "...",
            max_y,
        )
        for j in range(min_y, max_y + 1):
            line = []
            for i in range(min_x, max_x + 1):
                if (i, j) in sensors_pos:
                    line.append("S")
                elif (i, j) in beacons_pos:
                    line.append("B")
                else:
                    line.append(".")
            m.append(line)

            print(str(j).zfill(2), "".join(line))

    return sensors


@dataclasses.dataclass
class Point2D:
    x: int
    y: int

    @property
    def pos(self):
        return self.x, self.y


@dataclasses.dataclass
class Beacon(Point2D): ...


@dataclasses.dataclass
class Sensor(Point2D):
    beacon: Beacon

    def __str__(self):
        return f"{super().__str__()} range={self.dist_to_beacon}"

    @cached_property
    def dist_to_beacon(self):
        return manhattan_distance(self.pos, self.beacon.pos)

    def is_reaching(self, point: Point2D):
        """Does sensor cover the point?"""
        dist_to_beacon = self.dist_to_beacon
        dist_to_point = manhattan_distance(self.pos, point.pos)
        return dist_to_point <= dist_to_beacon


def task(inp, y):
    """
    In the y row, how many positions cannot contain a beacon?
    """
    sensors = _parse_puzzle(inp)

    min_x = min(s.x - s.dist_to_beacon for s in sensors)
    max_x = max(s.x + s.dist_to_beacon for s in sensors)

    covered = []
    print(f"Inspecting {min_x} - {max_x}")
    percent = (max_x - min_x) // 100 or 1
    for x in range(min_x, max_x + 1):
        if not (min_x + x) % percent:
            print(f"Inspecting {x}")
        point = Point2D(x, y)
        for sensor in sensors:
            if point.pos != sensor.beacon.pos and sensor.is_reaching(point):
                covered.append(point.pos)
                break

    return len(covered)


def tuning_frequency(inp, field):
    sensors = _parse_puzzle(inp)

    # probe 4 intervals
    sorted_sensors = sorted(sensors, key=lambda s: (s.y - s.dist_to_beacon, s.x))

    def get_interval(min_x, max_x):
        return (
            min(max(0, min_x), field),
            min(max(0, max_x), field),
        )

    for y in range(field):
        intervals = []

        for sensor in sorted_sensors:
            if sensor.y >= y:
                sensor_min_y = sensor.y - sensor.dist_to_beacon
                existence = y - sensor_min_y
                if existence >= 0:
                    interval = get_interval(sensor.x - existence, sensor.x + existence)
                    intervals.append(interval)
            else:
                sensor_max_y = sensor.y + sensor.dist_to_beacon
                existence = sensor_max_y - y
                if existence >= 0:
                    interval = get_interval(sensor.x - existence, sensor.x + existence)
                    intervals.append(interval)

        intervals = sorted(intervals)
        super_interval = intervals[0]
        if super_interval[0] != 0:
            return 0 * 4000000 + y

        for x_start, x_end in intervals[1:]:
            # overlapping
            sx_start, sx_end = super_interval
            if x_start > sx_end + 1:
                return (sx_end + 1) * 4000000 + y

            super_interval = (sx_start, max(sx_end, x_end))

        if super_interval[1] != field:
            return field * 4000000 + y


if __name__ == "__main__":
    # # part 1
    # DEBUG = True
    DEBUG = False

    # test(task, y=10, expected=26)
    # res = run(task, y=2000000)
    # assert res == 5040643, res

    # part 2
    test(tuning_frequency, field=20, expected=56000011)
    print("test ok")
    import datetime

    x = datetime.datetime.now()

    res = run(tuning_frequency, field=4000000)

    print(datetime.datetime.now() - x)

    # assert res == 24377, f"{res=} but it should be 24377 for my input!"
