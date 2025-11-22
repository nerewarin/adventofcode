import logging
import os
from typing import Protocol

from src.utils.test_and_run import run

# Configure logging based on environment variable
log_level = os.getenv("level", "INFO")
logging.basicConfig(level=log_level)


class MovementStrategy(Protocol):
    def move(self, blocks: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]: ...


class SplitBlockStrategy:
    """Strategy for part 1: split blocks and fill all gaps."""

    def move(self, blocks: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        total_length = sum(1 for _, _, length in blocks for _ in range(length)) * 2
        occupied = [False] * total_length
        result = []

        # Mark initially occupied positions
        for file_id, start, length in blocks:
            for pos in range(start, start + length):
                occupied[pos] = True

        # Process blocks from right to left
        for i in range(len(blocks) - 1, -1, -1):
            file_id, orig_start, length = blocks[i]
            logging.debug(f"\nMoving block {file_id} (length {length})")

            # Unmark original positions
            for pos in range(orig_start, orig_start + length):
                occupied[pos] = False

            # Find gaps and fill them
            remaining_length = length
            current_pos = 0

            while remaining_length > 0:
                # Find next free position
                while current_pos < total_length and occupied[current_pos]:
                    current_pos += 1

                # Find length of this gap
                gap_end = current_pos
                while gap_end < total_length and not occupied[gap_end]:
                    gap_end += 1

                # Fill gap
                fill_length = min(gap_end - current_pos, remaining_length)
                result.append((file_id, current_pos, fill_length))

                # Mark positions as occupied
                for pos in range(current_pos, current_pos + fill_length):
                    occupied[pos] = True

                logging.debug(f"Filled gap at {current_pos} with {fill_length} blocks")

                remaining_length -= fill_length
                current_pos = gap_end

        return sorted(result, key=lambda x: x[1])


class WholeFileStrategy:
    """Strategy for part 2: move whole files only."""

    def find_next_space(self, occupied: list[bool], needed_length: int, min_pos: int) -> int:
        """Find the first space that can fit needed_length starting from min_pos."""
        current_pos = min_pos

        while current_pos < len(occupied):
            # Check if we have enough contiguous space
            if all(not occupied[pos] for pos in range(current_pos, min(current_pos + needed_length, len(occupied)))):
                return current_pos
            current_pos += 1

        return current_pos

    def move(self, blocks: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        total_length = sum(1 for _, _, length in blocks for _ in range(length)) * 2
        occupied = [False] * total_length
        result = list(blocks)
        moved = set()  # Keep track of which blocks have been moved

        # Process files from highest ID to lowest
        for i in range(len(blocks) - 1, -1, -1):
            file_id, orig_start, length = blocks[i]
            logging.debug(f"\nMoving file {file_id} (length {length})")

            # For first block (highest ID), find position after block 0
            if i == len(blocks) - 1:
                _, start0, length0 = blocks[0]
                min_pos = start0 + length0
            else:
                # Find the rightmost position of unmoved blocks to our left
                min_pos = 0
                for j in range(i):
                    if j not in moved:
                        _, other_start, other_length = blocks[j]
                        min_pos = max(min_pos, other_start + other_length)

            # Find first available space after min_pos
            new_pos = self.find_next_space(occupied, length, min_pos)

            # Update result and mark positions
            result[i] = (file_id, new_pos, length)
            for pos in range(new_pos, new_pos + length):
                occupied[pos] = True
            moved.add(i)
            logging.debug(f"Moved file to position {new_pos}")

            logging.debug(f"Current state: {visualize_state(result, total_length)}")

        return result


def parse_disk_map(input_data: str | list) -> list[int]:
    if isinstance(input_data, list):
        input_data = input_data[0]
    return [int(x) for x in input_data.strip()]


def get_initial_state(disk_map: list[int]) -> list[tuple[int, int, int]]:
    """Convert disk map to list of (file_id, start_pos, length) tuples."""
    blocks = []
    file_id = 0
    current_pos = 0

    for i, length in enumerate(disk_map):
        if i % 2 == 0:  # File block
            blocks.append((file_id, current_pos, length))
            file_id += 1
        current_pos += length
    return blocks


def get_holes_by_position(disk_map: list[int]) -> dict[int, int]:
    # optional TODO: must be combines to get_initial_state to reduce double O(N)
    holes_by_position = dict()
    file_id = 0
    current_pos = 0

    for i, length in enumerate(disk_map):
        if i % 2 == 1:  # File block
            if length == 0:
                continue
            holes_by_position[current_pos] = length
            file_id += 1
        current_pos += length
    return holes_by_position


def solve_part2(input_data: str | list) -> int:
    """Move files according to part 2 rules."""

    disk_map = parse_disk_map(input_data)
    files_state = get_initial_state(disk_map)

    total_length = sum(disk_map)
    logging.debug(f"Initial: {visualize_state(files_state, total_length)}")

    disk = ["." for _ in range(total_length)]
    holes_by_position = get_holes_by_position(disk_map)

    # First, mark all initial positions
    for file_id, start, length in files_state:
        for i in range(length):
            disk[start + i] = str(file_id)

    def get_available_hole(length, holes_by_position):
        for position, hole_len in holes_by_position.items():
            if hole_len >= length:
                return position
        return None

    # Process files from the highest ID to lowest
    for block_id_to_move in range(len(files_state) - 1, -1, -1):
        file_id, start, length = files_state[block_id_to_move]
        logging.debug(f"\nMoving file {file_id} (length {length})")

        hole_position = get_available_hole(length, holes_by_position)
        if hole_position is None:
            continue
        else:
            # update holes
            hole_len = holes_by_position.pop(hole_position)
            new_hole_len = hole_len - length
            # merge holes
            if not new_hole_len:
                pass
            else:
                new_hole_pos = hole_position + length
                holes_by_position[new_hole_pos] = new_hole_len
                holes_by_position = dict(sorted(holes_by_position.items()))  # not sure we need it TODO

            # add hole respecting old file position
            # either find prior hole and prolongate it... but wait why we need it? TODO if not works - do it

            # update disk state
            files_state[block_id_to_move] = (file_id, hole_position, length)

            disk[start : start + length] = ["." for _ in range(length)]
            disk[hole_position : hole_position + length] = [str(file_id) for _ in range(length)]

            # this is for show only
            logging.debug(f"Current state: {''.join(disk)}")

        # # Move one digit at a time
        # for digit in range(length):
        #     # Clear current position of this digit
        #     disk[start + digit] = "."
        #
        #     # Find position after last unmoved file
        #     pos = 0
        #     for j in range(block_id_to_move):
        #         # Look for positions of unmoved files
        #         other_id = files_state[j][0]
        #         for k in range(total_length):
        #             if disk[k] == str(other_id):
        #                 pos = max(pos, k + 1)
        #
        #     # Place this digit
        #     disk[pos + digit] = str(file_id)
        #     logging.debug(f"Current state: {''.join(disk)}")
        #
        # # Record final position of this file
        # result.append((file_id, pos, length))

    final_state = sorted(files_state, key=lambda x: x[0])  # Sort by file_id

    logging.debug(f"Final: {visualize_state(final_state, total_length)}")
    return calculate_checksum(final_state)


def calculate_checksum(positions: list[tuple[int, int, int]]) -> int:
    """Calculate checksum based on final positions."""
    return sum(
        pos * file_id for file_id, start_pos, length in positions for pos in range(start_pos, start_pos + length)
    )


def visualize_state(blocks: list[tuple[int, int, int]], total_length: int) -> str:
    """Helper function to visualize the disk state."""
    disk = ["." for _ in range(total_length)]
    list_repr = [None] * total_length
    for file_id, start_pos, length in blocks:
        for i in range(length):
            disk[start_pos + i] = str(file_id)
            list_repr[start_pos + i] = file_id
    return "".join(disk)


def solve(input_data: str | list, strategy: MovementStrategy) -> int:
    disk_map = parse_disk_map(input_data)
    initial_state = get_initial_state(disk_map)

    logging.debug(f"Initial: {visualize_state(initial_state, sum(disk_map))}")
    final_state = strategy.move(initial_state)
    logging.debug(f"Final: {visualize_state(final_state, sum(disk_map))}")

    return calculate_checksum(final_state)


def solve_part1(input_data: str | list) -> int:
    return solve(input_data, SplitBlockStrategy())


# test(solve_part1, expected=1928)
# assert run(solve_part1) == 6307275788409
# test(solve_part2, expected=2858)
res2 = run(solve_part2)
assert res2 < 8039434080946
