"""
--- Day 4: Scratchcards  ---
https://adventofcode.com/2023/day/4
"""

from src.utils.test_and_run import run, test


def scratchcards(inp, part=1):
    win_nums_list = []
    for line in inp:
        r = line.split(":")[-1].strip().split("|")
        winning_nums, nums = [set([int(x) for x in v.split(" ") if x]) for v in r]
        win_nums = len(winning_nums.intersection(nums))
        win_nums_list.append(win_nums)

    if part == 1:
        return sum([2 ** (win_nums - 1) for win_nums in win_nums_list if win_nums])

    res = 0
    card_counts = [1] * len(win_nums_list)
    for i, win_nums in enumerate(win_nums_list):
        score = card_counts[i]
        for j in range(win_nums):
            card_counts[i + j + 1] += score

        res += score

    return res


if __name__ == "__main__":
    test(scratchcards, expected=13)
    run(scratchcards)

    test(scratchcards, part=2, expected=30)
    run(scratchcards, part=2)
