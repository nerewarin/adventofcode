import re
import os

def solve_part1(input_data: str) -> int:
    # Regular expression to match valid mul(X,Y) patterns
    # Matches: mul(digits,digits) where digits are 1-3 characters
    pattern = r'mul\((\d{1,3}),(\d{1,3})\)'
    
    # Find all matches in the input
    matches = re.finditer(pattern, input_data)
    
    total = 0
    count = 0
    for match in matches:
        x = int(match.group(1))
        y = int(match.group(2))
        result = x * y
        total += result
        count += 1
        if count < 5:  # Print first few matches to verify
            print(f"Found: mul({x},{y}) = {result}")
    
    print(f"Total matches found: {count}")
    return total

def solve_part2(input_data: str) -> int:
    # Find all instructions (mul, do, don't) in order of appearance
    pattern = r'(?:mul\((\d{1,3}),(\d{1,3})\)|do\(\)|don\'t\(\))'
    matches = re.finditer(pattern, input_data)
    
    total = 0
    enabled = True  # Multiplications are enabled by default
    count = 0
    
    for match in matches:
        instruction = match.group(0)
        
        if instruction == "do()":
            enabled = True
            continue
        elif instruction == "don't()":
            enabled = False
            continue
            
        # Must be a multiplication if we get here
        if enabled:
            x = int(match.group(1))
            y = int(match.group(2))
            result = x * y
            total += result
            count += 1
            if count < 5:  # Print first few enabled multiplications
                print(f"Found enabled: mul({x},{y}) = {result}")
    
    print(f"Total enabled matches found: {count}")
    return total

if __name__ == "__main__":
    # Get the absolute path to the input file
    current_dir = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(current_dir, 'inputs', '3', 'run')
    
    # Debug: Print the file path
    print(f"Trying to read file: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File does not exist at {input_file}")
        exit(1)
    
    # Read the input file
    with open(input_file, 'r') as f:
        input_data = f.read()
        print(f"Input data length: {len(input_data)}")
        # Print first 100 characters to verify content
        print(f"First 100 chars: {input_data[:100]}")
    
    print(f"Part 1: {solve_part1(input_data)}")
    print(f"Part 2: {solve_part2(input_data)}")
