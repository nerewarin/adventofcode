"""--- Day 3: Rucksack Reorganization ---
https://adventofcode.com/2022/day/3
"""

from src.utils.test_and_run import run, test


def _estimate_letter_score(letter):
    idx = ord(letter)
    if idx >= 97:
        return idx - 96
    return idx - 64 + 26


def rucksack_reorganization(inp, task="rucksack_reorganization"):
    res = 0

    datasets = inp
    if task == "group_badges":
        datasets = [inp[idx * 3 : idx * 3 + 3] for idx in range(len(datasets) // 3)]

    for dataset in datasets:
        if task == "group_badges":
            duplicates = set.intersection(*(set(line) for line in dataset))
        else:
            middle = len(dataset) // 2
            parts = set(dataset[:middle]), set(dataset[middle:])
            duplicates = set.intersection(*parts)
        duplicate = duplicates.pop()
        assert not duplicates, "more than one duplicate left"

        one_res = _estimate_letter_score(letter=duplicate)
        res += one_res

    return res


if __name__ == "__main__":
    # # part 1
    test(rucksack_reorganization, expected=157)
    run(rucksack_reorganization)

    # part 2
    test(rucksack_reorganization, task="group_badges", expected=70)
    run(rucksack_reorganization, task="group_badges")
