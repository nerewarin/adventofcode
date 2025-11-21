"""--- Day 13: Distress Signal ---
https://adventofcode.com/2022/day/13
"""

import math

from src.utils.test_and_run import run, test


class BlocksComparator:
    @staticmethod
    def _compare_ints(left, right):
        if left == right:
            return None
        return left < right

    @classmethod
    def order_is_correct(cls, left, right):
        # multi-type elements comparison
        print(f"Compare {left} and {right}")
        if isinstance(left, int) and isinstance(right, int):
            return cls._compare_ints(left, right)

        if isinstance(left, list) and isinstance(right, list):
            ll = len(left)
            lr = len(right)
            for idx, l in enumerate(left):
                if idx >= len(right):
                    print("Right side ran out of items, so inputs are not in the right order")
                    return False

                r = right[idx]
                is_valid = cls.order_is_correct(l, r)
                if isinstance(is_valid, bool):
                    return is_valid

            if ll < lr:
                print("Left side ran out of items, so inputs are in the right order")
                return True

            return None

        if isinstance(left, int):
            left = [left]
        elif isinstance(right, int):
            right = [right]
        return cls.order_is_correct(left, right)


class Pair:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def is_valid(self):
        print(f"Compare\n\t{self.left}\nvs\n\t{self.right}")
        return BlocksComparator.order_is_correct(self.left, self.right)


def _parse_puzzle(inp):
    pairs = []
    pairs_data = []
    for row_num, line in enumerate(inp):
        if line:
            v = eval(line)
            pairs_data.append(v)
        else:
            pair = Pair(*pairs_data)
            pairs.append(pair)

            if pairs_data[0] == pairs_data[1]:
                pass

            pairs_data = []

    if pairs_data:
        pair = Pair(*pairs_data)
        pairs.append(pair)

    return pairs


def task(inp):
    """
    Determine which pairs of packets are already in the right order. What is the sum of the indices of those pairs?
    """
    pairs = _parse_puzzle(inp)

    valid_pairs_indexes = []
    for pair_idx, pair in enumerate(pairs):
        pair_idx_ = pair_idx + 1
        print(f"== Pair {pair_idx_} ==")
        v = pair.is_valid()
        if v in (True, None):
            print(f"Pair {pair_idx_} is valid")
            valid_pairs_indexes.append(pair_idx_)
        else:
            print(f"Pair {pair_idx_} is NOT valid")
        print()

    print(valid_pairs_indexes)
    return sum(valid_pairs_indexes)


def get_decoder_key(inp):
    pairs = _parse_puzzle(inp)
    divider_blocks = (
        [[2]],
        [[6]],
    )
    blocks = list(divider_blocks)
    for pair in pairs:
        blocks.append(pair.left)
        blocks.append(pair.right)

    ordered_blocks = []
    for i, left in enumerate(blocks):
        # insert block in the right place
        for j, right in enumerate(ordered_blocks):
            if BlocksComparator.order_is_correct(left, right):
                ordered_blocks.insert(j, left)
                break
        else:
            ordered_blocks.append(left)

    divider_indexes = []
    for i, block in enumerate(ordered_blocks):
        if block in divider_blocks:
            divider_indexes.append(i + 1)

    return math.prod(divider_indexes)


if __name__ == "__main__":
    # part 1
    test(task, expected=13)
    res = run(task)
    assert res == 5720, res

    # part 2
    test(get_decoder_key, expected=140)
    res = run(get_decoder_key)
    assert res == 23504, res
