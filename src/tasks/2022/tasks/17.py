"""--- Day 17: Pyroclastic Flow ---
https://adventofcode.com/2022/day/17
"""

import datetime
from functools import lru_cache

from src.utils.test_and_run import run, test


@lru_cache
def _parse_puzzle(inp):
    axes = []
    for sym in inp[0]:
        axes.append(-1 if sym == "<" else 1)
    return axes


class Figure:
    """
    ####

    .#.
    ###
    .#.

    ..#
    ..#
    ###

    #
    #
    #
    #

    ##
    ##
    """

    # shapes = dict(
    #     horizontal=[[1, 1, 1, 1]],
    #     cross=[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    #     letter_L=[[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    #     vertical=[[1], [1], [1], [1]],
    #     square=[[1] * 2] * 2,
    # )
    # shapes = dict(
    #     vertical=[[1], [1], [1], [1]],
    #         horizontal=[[1, 1, 1, 1]],
    #         square=[[1] * 2] * 2,
    #     # cross=[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    # )
    _shape_pos = dict(
        horizontal=[(0, 0), (1, 0), (2, 0), (3, 0)],
        cross=[(0, 1), (1, 1), (1, 0), (1, 2), (2, 1)],
        letter_L=[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        vertical=[(0, 0), (0, 1), (0, 2), (0, 3)],
        square=[(0, 0), (0, 1), (1, 0), (1, 1)],
    )

    def __init__(self, name, state_depth):
        self.name = name
        self.state_depth = state_depth
        self.rest = False
        self.y = 3
        self.x = 2
        self.shape2d = self._shape_pos[name]

    def __repr__(self):
        return f"{self.__class__.__qualname__}(name={self.name}, xy=({self.x},{self.y})"

    def get_points(self):
        # # update state
        for x, y in self.shape2d:
            posy = self.y - y
            posx = self.x + x

            yield posx, posy

    def _collision(self, state, flow):
        for x_, y in self.get_points():
            x = x_ + flow
            if not 0 <= x < Map.width:
                return True

            is_closed = state.tiles[y][x]
            if is_closed:
                return True

        return False

    def shift(self, flow, state: "Map"):
        # check collision
        if self._collision(state, flow):
            return

        self.x += flow

    @classmethod
    def get_figures(cls) -> list[str]:
        return list(cls._shape_pos)


class Map:
    # 2d values from down to bot
    width = 7
    start_offset = [0, 0]

    def __init__(self):
        # tiles[y][x] == 1 means closed
        self.tiles = [[0 for row in range(self.width)] for col in range(7)]
        self.depth = [3] * self.width
        self.figure = None
        self.fig_depth = None

    def view(self):
        r = []
        for row in self.tiles:
            r.append("".join(["#" if x else "." for x in row]))
        return r

    def __repr__(self):
        return f"{self.__class__.__qualname__}(tiles={self.tiles})"

    def add_figure(self, figure: Figure):
        max_depth = max(self.depth)
        if 3 > max_depth:
            add = 3 - -max_depth
            self.depth = [d + add for d in self.depth]

        self.figure = figure
        return

    def _get_height_diff(self):
        for i, line in enumerate(self.tiles):
            for s in line:
                if s:
                    return 7 - i

    def place_figure(self, figure: Figure):
        state = self
        state_depth = state.depth.copy()

        old_min_depth = min(state.depth)
        assert old_min_depth == 3

        height_diff = old_min_depth - min(state_depth)
        state_depth = [s + height_diff for s in state_depth]
        self.depth = state_depth

        # update state with figure
        for x, y in figure.shape2d:
            posy = figure.y - y
            posx = figure.x + x
            self.tiles[posy][posx] = 1

        height_diff = self._get_height_diff()
        # add new lines
        self.tiles = [[0 for x in range(self.width)] for y in range(height_diff)] + self.tiles

        return height_diff

    def move_down(self, figure: Figure) -> bool:
        # move figure down increasing its 'depth_from_start'
        # depth = list(figure.depth_from_start)
        # for i, elm in enumerate(depth):
        #     if elm is None:
        #         continue
        #
        #     elm += 1
        #     if elm > self.depth[i]:
        #         return True
        #
        #     depth[i] = elm
        #
        # figure.depth_from_start = depth

        for x, y_ in figure.get_points():
            # check what will be with y + 1
            y = y_ + 1
            if not 0 <= x < Map.width:
                return True
            if not y < len(self.tiles):
                return True
            is_closed = self.tiles[y][x]
            if is_closed:
                return True

        figure.y += 1
        return False


def task1(inp, limit=2022):
    flow_cycle = _parse_puzzle(tuple(inp))
    flow_len = len(flow_cycle)

    def drop_figures(limit=None):
        state = Map()
        figures_names = Figure.get_figures()
        figure_idx = 0
        flow_idx = 0

        figures_len = len(figures_names)

        def is_goal():
            if limit:
                return limit == figure_idx
            return figure_idx == figures_len
            return figure_idx == flow_len

        height = 0
        while not is_goal():
            name = figures_names[figure_idx % figures_len]
            figure = Figure(name, state.depth)
            state.add_figure(figure)

            while not figure.rest:
                flow = flow_cycle[flow_idx % flow_len]
                if flow_idx % (flow_len * figures_len) == 0 and flow_idx != 0:
                    print(figure_idx, height)
                elif figure_idx == 36 + 35 + 15:
                    print(figure_idx, height)

                flow_idx += 1
                figure.shift(flow, state)  # flow_idx % flow_len == 0 and figure_idx % figures_len == 0

                figure.rest = state.move_down(figure)

            state.place_figure(figure)
            figure_idx += 1
            height = len(state.tiles) - 7
            a = 0
            a = 0

        return height  # 1586627906917 < res < 1586627906922
        # not 1586627906920
        #  1586627906921 ?

    return drop_figures(limit=limit)


def part1(expected):
    start = datetime.datetime.now()

    try:
        test(task1, expected=expected)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    res = run(task1)  # 3171
    end = datetime.datetime.now()
    print(end - start)


def part2(expected):
    start = datetime.datetime.now()

    # TODO nake algorithm after manual solve. here is how it was
    """
    First, find figures placed (fig_idx) and height (height) diff between cycls
     N * flow_len * fig_len
    and
     (N + 1) * flow_len * fig_len
    then collect start_flow and start_h as a result of stabling system to cyclic mdoe (somewhere from 0 to N)
    then add the tail (rest) manyally  from cycle (N + rest)


    trl = 1000000000000
    divmod(trl, 35)
    (28571428571, 15)
    start_flow = 36
    start_h = 61
    start_fig = 36
    fig_cycle = 35
    h_cycle = 53
    cycles, rest = divmod(trl - start_fig, fig_cycle)
    res = start_fig + cycles * h_cycle + 23
    res
    1514285714269
    res = start_fig + cycles * h_cycle +18
    res
    1514285714264
    1514285714288 - res
    24
    res = start_h + cycles * h_cycle +18
    """
    try:
        test(task1, limit=1000000000000, expected=expected)
    except AssertionError as e:
        print(e)
        print("test not ok")
    else:
        print("test ok")

    # res = run(task1, limit=1000000000000)
    end = datetime.datetime.now()
    print(end - start)


if __name__ == "__main__":
    # part1(expected=3068)
    part2(expected=1514285714288)
    part2(expected=1514285714269)
