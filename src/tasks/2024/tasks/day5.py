from collections import defaultdict

def parse_input(input_text):
    rules_text, updates_text = input_text.strip().split('\n\n')
    
    # Parse rules into a graph
    rules = defaultdict(set)
    for line in rules_text.split('\n'):
        if '|' not in line:
            continue
        before, after = map(int, line.split('|'))
        rules[before].add(after)
    
    return rules, updates_text.strip().split('\n')

def is_valid_order(pages, rules):
    for i, page in enumerate(pages):
        if page in rules:
            for must_come_after in rules[page]:
                if must_come_after in pages:
                    after_pos = pages.index(must_come_after)
                    if after_pos < i:
                        return False
    return True

def topological_sort(pages, rules):
    # Build adjacency list and in-degree count
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all pages
    all_pages = set(pages)
    
    # Build the graph based on rules
    for page in pages:
        if page in rules:
            for after_page in rules[page]:
                if after_page in all_pages:
                    graph[page].add(after_page)
                    in_degree[after_page] += 1
    
    # Find all nodes with no incoming edges
    queue = [page for page in pages if in_degree[page] == 0]
    result = []
    
    # Process queue
    while queue:
        # Sort queue to ensure deterministic ordering
        queue.sort(reverse=True)  # Take highest number when multiple options
        current = queue.pop()
        result.append(current)
        
        # Remove edges from current node
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If result length matches input length, we have a valid ordering
    if len(result) == len(pages):
        return result
    return None

def solve_day5(input_text):
    rules, updates = parse_input(input_text)
    
    # Part 1: Find valid updates and their middle numbers
    middle_sum_part1 = 0
    invalid_updates = []
    
    for update in updates:
        if ',' not in update:
            continue
        pages = list(map(int, update.split(',')))
        if is_valid_order(pages, rules):
            middle_index = len(pages) // 2
            middle_sum_part1 += pages[middle_index]
        else:
            invalid_updates.append(pages)
    
    # Part 2: Fix invalid updates and sum their middle numbers
    middle_sum_part2 = 0
    print(f"Processing {len(invalid_updates)} invalid updates...")
    
    for i, pages in enumerate(invalid_updates):
        print(f"Processing update {i+1}/{len(invalid_updates)}: {pages}")
        ordered_pages = topological_sort(pages, rules)
        if ordered_pages:
            middle_index = len(ordered_pages) // 2
            middle_value = ordered_pages[middle_index]
            middle_sum_part2 += middle_value
            print(f"  Original: {pages}")
            print(f"  Reordered: {ordered_pages}")
            print(f"  Middle number: {middle_value}")
        else:
            print(f"Warning: Could not find valid ordering for {pages}")
    
    print(f"Total invalid updates processed: {len(invalid_updates)}")
    print(f"Total middle sum for part 2: {middle_sum_part2}")
    
    return middle_sum_part1, middle_sum_part2

def main():
    with open('./inputs/5/run', 'r') as file:
        input_text = file.read()
    
    part1_result, part2_result = solve_day5(input_text)
    print(f"Part 1 - Sum of middle page numbers from correctly-ordered updates: {part1_result}")
    print(f"Part 2 - Sum of middle page numbers from fixed updates: {part2_result}")

if __name__ == "__main__":
    main()
