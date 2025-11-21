from pathlib import Path
from collections import Counter

def read_input():
    input_file = Path(__file__).parent.parent / "inputs" / "1" / "run"
    with open(input_file) as f:
        lines = f.readlines()
    
    left_list = []
    right_list = []
    
    for line in lines:
        left, right = line.strip().split()
        left_list.append(int(left))
        right_list.append(int(right))
    
    return left_list, right_list

def calculate_total_distance(left_list, right_list):
    # Sort both lists
    left_sorted = sorted(left_list)
    right_sorted = sorted(right_list)
    
    # Calculate total distance between paired numbers
    total_distance = 0
    for left, right in zip(left_sorted, right_sorted):
        distance = abs(left - right)
        total_distance += distance
    
    return total_distance

def calculate_similarity_score(left_list, right_list):
    # Count occurrences of numbers in right list
    right_counter = Counter(right_list)
    
    # Calculate similarity score
    total_score = 0
    for num in left_list:
        # Multiply number by its frequency in right list
        total_score += num * right_counter[num]
    
    return total_score

def solve_part1():
    left_list, right_list = read_input()
    return calculate_total_distance(left_list, right_list)

def solve_part2():
    left_list, right_list = read_input()
    return calculate_similarity_score(left_list, right_list)

if __name__ == "__main__":
    print(f"Part 1: {solve_part1()}")
    print(f"Part 2: {solve_part2()}")
