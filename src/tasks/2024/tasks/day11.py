"""
--- Day 11: Plutonian Pebbles ---

https://adventofcode.com/2024/day/11

"""

import logging
import os

from src.utils.test_and_run import run, test

# Configure logging based on environment variable
log_level = os.getenv("level", "INFO")
logging.basicConfig(level=log_level)
_logger = logging.getLogger(__name__)


def compute_stones_list(inp, blinks):
    initial_ids = [int(stone_id) for stone_id in inp[0].split(" ")]

    res = list(initial_ids)

    for blink in range(blinks):
        new_res = []
        for ind in range(len(res)):
            stone_id = res[ind]

            if stone_id == 0:
                stone_id = 1
                new_stones = [stone_id]
            elif len(stone_id_str := str(stone_id)) % 2 == 0:
                mid = len(stone_id_str) // 2
                left = stone_id_str[:mid]
                right = stone_id_str[mid:]
                new_stones = [int(left), int(right)]
            else:
                new_stones = [stone_id * 2024]
            new_res.extend(new_stones)

        res = new_res

    return res


def compute_stones_string(inp, blinks):
    return " ".join([str(x) for x in compute_stones_list(inp, blinks)])


def task1(inp: list[str]) -> int:
    return len(compute_stones_list(inp, blinks=25))


if __name__ == "__main__":
    test(compute_stones_string, "1 2024 1 0 9 9 2021976", blinks=1)
    test(
        compute_stones_string,
        "2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2",
        test_part=2,
        blinks=6,
    )
    test(task1, 55312, test_part=2)
    run(task1)
