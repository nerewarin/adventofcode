from src.utils.test_and_run import test, run
from typing import List, Tuple, Protocol
import logging
import os

# Configure logging based on environment variable
log_level = os.getenv('level', 'INFO')
logging.basicConfig(level=log_level)

class MovementStrategy(Protocol):
    def move(self, blocks: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        ...

class SplitBlockStrategy:
    """Strategy for part 1: split blocks and fill all gaps."""
    def move(self, blocks: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
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
    def find_next_space(self, occupied: List[bool], needed_length: int, min_pos: int) -> int:
        """Find the first space that can fit needed_length starting from min_pos."""
        current_pos = min_pos
        
        while current_pos < len(occupied):
            # Check if we have enough contiguous space
            if all(not occupied[pos] for pos in range(current_pos, min(current_pos + needed_length, len(occupied)))):
                return current_pos
            current_pos += 1
            
        return current_pos

    def move(self, blocks: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
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

def parse_disk_map(input_data: str | list) -> List[int]:
    if isinstance(input_data, list):
        input_data = input_data[0]
    return [int(x) for x in input_data.strip()]

def get_initial_state(disk_map: List[int]) -> List[Tuple[int, int, int]]:
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

def move_files_part2(blocks: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Move files according to part 2 rules."""
    total_length = sum(1 for _, _, length in blocks for _ in range(length)) * 2
    disk = ['.' for _ in range(total_length)]
    result = []
    
    # First, mark all initial positions
    for file_id, start, length in blocks:
        for i in range(length):
            disk[start + i] = str(file_id)
    
    # Process files from highest ID to lowest
    for i in range(len(blocks) - 1, -1, -1):
        file_id, start, length = blocks[i]
        logging.debug(f"\nMoving file {file_id} (length {length})")
        
        # Move one digit at a time
        for digit in range(length):
            # Clear current position of this digit
            disk[start + digit] = '.'
            
            # Find position after last unmoved file
            pos = 0
            for j in range(i):
                # Look for positions of unmoved files
                other_id = blocks[j][0]
                for k in range(total_length):
                    if disk[k] == str(other_id):
                        pos = max(pos, k + 1)
            
            # Place this digit
            disk[pos + digit] = str(file_id)
            logging.debug(f"Current state: {''.join(disk)}")
        
        # Record final position of this file
        result.append((file_id, pos, length))
    
    return sorted(result, key=lambda x: x[0])  # Sort by file_id

def calculate_checksum(positions: List[Tuple[int, int, int]]) -> int:
    """Calculate checksum based on final positions."""
    return sum(pos * file_id for file_id, start_pos, length in positions 
              for pos in range(start_pos, start_pos + length))

def visualize_state(blocks: List[Tuple[int, int, int]], total_length: int) -> str:
    """Helper function to visualize the disk state."""
    disk = ['.' for _ in range(total_length)]
    for file_id, start_pos, length in blocks:
        for i in range(length):
            disk[start_pos + i] = str(file_id)
    return ''.join(disk)

def solve(input_data: str | list, strategy: MovementStrategy) -> int:
    disk_map = parse_disk_map(input_data)
    initial_state = get_initial_state(disk_map)
    
    logging.debug(f"Initial: {visualize_state(initial_state, sum(disk_map))}")
    final_state = strategy.move(initial_state)
    logging.debug(f"Final: {visualize_state(final_state, sum(disk_map))}")
    
    return calculate_checksum(final_state)

def solve_part1(input_data: str | list) -> int:
    return solve(input_data, SplitBlockStrategy())

def solve_part2(input_data: str | list) -> int:
    disk_map = parse_disk_map(input_data)
    initial_state = get_initial_state(disk_map)
    
    total_length = sum(disk_map)
    logging.debug(f"Initial: {visualize_state(initial_state, total_length)}")
    final_state = move_files_part2(initial_state)
    logging.debug(f"Final: {visualize_state(final_state, total_length)}")
    
    return calculate_checksum(final_state)

test(solve_part1, expected=1928)
# test(solve_part2, expected=2858)
# run(solve_part1)
# run(solve_part2)
# 6307275788409