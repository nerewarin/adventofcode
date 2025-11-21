"""
--- Day 13: Care Package ---

https://adventofcode.com/2019/day/13

"""

import collections

import _tools
from _intcode_computer import IntcodeComputer

BLOCK_SYMBOL = '='
BALL_SYMBOL = 'o'
HORIZONTAL_PADDLE = '-'
TILE2DRAW = {
    0: '.',  # is an empty tile. No game object appears in this tile.
    1: 'X',  # is a wall tile. Walls are indestructible barriers.
    2: BLOCK_SYMBOL,  # is a block tile. Blocks can be broken by the ball.
    3: HORIZONTAL_PADDLE,  # is a horizontal paddle tile. The paddle is indestructible.
    4: BALL_SYMBOL ,  # is a ball tile. The ball moves diagonally and bounces off objects.
}


class IntcodeComputer13_1(IntcodeComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canvas = collections.defaultdict(str)

    def _on_step_start(self):
        if len(self.output) == 3:
            print('_step', self._step)

            x, y, tile_id = self.output
            print(self.output)
            draw_symbol = TILE2DRAW[tile_id]
            _pos = (x, y)
            self.canvas[_pos] = draw_symbol

            ys = [pos[1] for pos in self.canvas.keys()]
            xs = [pos[0] for pos in self.canvas.keys()]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)

            print('y from {} to {}'.format(min_y, max_y))
            print('x from {} to {}'.format(min_x, max_x))
            for _y in range(min_y, max_y + 1):
                line = [self.canvas.get((_x, _y), '.') for _x in range(min_x, max_x + 1)]
                print(''.join(line))
            self.output = []


class IntcodeComputer13_2(IntcodeComputer13_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory[0] = 2
        self.score = None

    def _on_step_start(self):
        if len(self.output) == 3:
            x, y, out3 = self.output
            # print(output)

            _pos = (x, y)
            if _pos == (-1, 0):
                # the new score to show in the segment display
                self.score = out3
            else:
                draw_symbol = TILE2DRAW[out3]
                self.canvas[_pos] = draw_symbol

            self.output = []

    def _get_op3_input(self):
        ball_symbol_x = [pos for pos, tile_symbol in self.canvas.items() if tile_symbol == BALL_SYMBOL][0][0]
        horizontal_paddle_x = [pos for pos, tile_symbol in self.canvas.items() if tile_symbol == HORIZONTAL_PADDLE][0][0]

        if ball_symbol_x > horizontal_paddle_x:
            shift_action = 1
        elif ball_symbol_x == horizontal_paddle_x:
            shift_action = 0
        else:
            shift_action = -1

        return shift_action

def part1():
    computer = IntcodeComputer13_1()
    computer.compute()

    blocks = 0
    for tile_symbol in computer.canvas.values():
        if tile_symbol == BLOCK_SYMBOL:
            blocks += 1

    return blocks

def part2():
    computer = IntcodeComputer13_2()
    computer.compute()
    return computer.score


if __name__ == '__main__':
    for res in (
        part1(), # 376
        part2(), # 18509
    ):
        print(res)
