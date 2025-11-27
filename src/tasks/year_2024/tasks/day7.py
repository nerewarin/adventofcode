import itertools

from src.utils.test_and_run import run, test

# Add project root to Python path

# Define operators for part1 and part2
PART1_OPERATORS = ["+", "*"]
PART2_OPERATORS = ["+", "*", "||"]


def evaluate_expression(numbers, operators):
    """Evaluate the expression based on the numbers and operators."""
    result = numbers[0]
    for i in range(len(operators)):
        if operators[i] == "+":
            result += numbers[i + 1]
        elif operators[i] == "*":
            result *= numbers[i + 1]
        elif operators[i] == "||":
            # Concatenate the numbers
            result = int(str(result) + str(numbers[i + 1]))
    return result


def can_make_true(test_value, numbers, operators):
    """Check if the test value can be made true with any combination of operators."""
    for ops in itertools.product(operators, repeat=len(numbers) - 1):
        if evaluate_expression(numbers, ops) == test_value:
            return True
    return False


def part1(data):
    total_sum = 0
    for line in data:
        test_value, numbers_str = line.split(":")
        test_value = int(test_value.strip())
        numbers = list(map(int, numbers_str.strip().split()))

        # Check if the test value can be made true with the given numbers using part1 operators
        if can_make_true(test_value, numbers, PART1_OPERATORS):
            total_sum += test_value

    return total_sum


def part2(data):
    total_sum = 0
    for line in data:
        test_value, numbers_str = line.split(":")
        test_value = int(test_value.strip())
        numbers = list(map(int, numbers_str.strip().split()))

        # Check if the test value can be made true with the given numbers using part2 operators
        if can_make_true(test_value, numbers, PART2_OPERATORS):
            total_sum += test_value

    return total_sum


if __name__ == "__main__":
    # Use the test function to validate part1
    test(part1, 3749)  # Expected output for the test data is 3749

    # Run the actual input data
    run(part1)

    # Use the test function to validate part2
    test(part2, 11387)  # Expected output for the test data is 11387

    # Run the actual input data for part2
    run(part2)
