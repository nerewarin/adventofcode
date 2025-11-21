from pathlib import Path

def read_input():
    input_file = Path(__file__).parent.parent / "inputs" / "2" / "run"
    with open(input_file) as f:
        return [list(map(int, line.strip().split())) for line in f]

def is_safe_report(levels):
    if len(levels) < 2:
        return True
    
    # Check first difference to determine if we should be increasing or decreasing
    diff = levels[1] - levels[0]
    if diff == 0:  # No change is not allowed
        return False
    
    increasing = diff > 0
    
    # Check each adjacent pair
    for i in range(len(levels) - 1):
        diff = levels[i + 1] - levels[i]
        
        # Check if difference is between 1 and 3 (inclusive)
        if abs(diff) < 1 or abs(diff) > 3:
            return False
        
        # Check if direction is consistent
        if increasing and diff <= 0:
            return False
        if not increasing and diff >= 0:
            return False
    
    return True

def is_safe_with_dampener(levels):
    # First check if it's safe without dampener
    if is_safe_report(levels):
        return True
    
    # Try removing each level one at a time
    for i in range(len(levels)):
        # Create new list without current level
        dampened_levels = levels[:i] + levels[i+1:]
        if is_safe_report(dampened_levels):
            return True
    
    return False

def solve_part1():
    reports = read_input()
    return sum(1 for report in reports if is_safe_report(report))

def solve_part2():
    reports = read_input()
    return sum(1 for report in reports if is_safe_with_dampener(report))

if __name__ == "__main__":
    print(f"Part 1: {solve_part1()}")
    print(f"Part 2: {solve_part2()}")
