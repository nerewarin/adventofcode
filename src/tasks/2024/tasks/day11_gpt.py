from functools import lru_cache

from src.utils.test_and_run import run


@lru_cache(None)
def count_stones(stone: int, blinks: int) -> int:
    if blinks == 0:
        return 1

    # Rule 1
    if stone == 0:
        return count_stones(1, blinks - 1)

    s = str(stone)
    # Rule 2: even digit count → split
    if len(s) % 2 == 0:
        mid = len(s) // 2
        left = int(s[:mid])
        right = int(s[mid:])
        return count_stones(left, blinks - 1) + count_stones(right, blinks - 1)

    # Rule 3: multiply
    return count_stones(stone * 2024, blinks - 1)


def task1(inp):
    stones = [int(x) for x in inp[0].split()]
    return sum(count_stones(s, 25) for s in stones)


def task2(inp):
    stones = [int(x) for x in inp[0].split()]
    return sum(count_stones(s, 75) for s in stones)


if __name__ == "__main__":
    # test(compute_stones_string, "1 2024 1 0 9 9 2021976", blinks=1)
    # test(
    #     compute_stones_string,
    #     "2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2",
    #     test_part=2,
    #     blinks=6,
    # )
    # test(task1, 55312, test_part=2)
    # run(task1)
    run(task2)  # 218817038947400

    print(count_stones.cache_info())
