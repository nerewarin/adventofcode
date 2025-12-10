"""
--- Day 10: Factory ---
https://adventofcode.com/2025/day/10
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from src.utils.logger import get_logger
from src.utils.numbers import bits_to_int
from src.utils.test_and_run import run, test

_logger = get_logger()
_rexp = re.compile(
    r"""
    \[(?P<indicators>[.#]+)\]
  | \((?P<toggle>\d+(?:,\d+)*)\)
  | \{(?P<joltage>\d+(?:,\d+)*)\}
    """,
    re.VERBOSE,
)


class Indicator(StrEnum):
    off = "."
    on = "#"

    @property
    def value(self):
        return int(self == Indicator.on)


class BitInt(int):
    def __new__(cls, value: int, width: int | None = None):
        obj = int.__new__(cls, value)
        obj.width = width or max(1, value.bit_length())
        return obj

    def __repr__(self):
        bits = format(int(self), f"0{self.width}b")
        return f"{bits} {int(self)}"


@dataclass
class Problem:
    indicators: BitInt
    toggles: list[BitInt]
    joltage: set[BitInt]
    task_num: int

    def solve(self) -> int:
        """
        Theory: Linear Algebra over GF(2) (XOR field)

        Problem: Find minimum toggles such that:
            start_state XOR (toggle1*x1 XOR toggle2*x2 XOR ...) = indicators

        Since start_state is all zeros (all lights off), this becomes:
            toggle1*x1 XOR toggle2*x2 XOR ... = indicators

        Algorithm:
        1. Build matrix A: rows = bit positions, columns = toggles
           - Entry A[i][j] = 1 if toggle j affects bit i, else 0
        2. Build vector b: b[i] = i-th bit of indicators (target pattern)
        3. Solve Ax = b over GF(2) where x[j] ∈ {0,1} means use toggle j
           - Use Gaussian elimination over GF(2) to find solution space
        4. Find minimum-weight solution (fewest toggles = minimum Hamming weight)
           - Try all combinations of free variables
           - Return minimum weight

        Key insight: Each toggle can be applied 0 or 1 times (since XOR is its own inverse),
        so we're solving: Ax = b over GF(2) where x is a binary vector.
        Note: Joltage is irrelevant and ignored.
        """
        # Example: if indicators = 8 bits, toggles = [toggle1, toggle2, ...]
        # Matrix structure:
        #   Row 0 (bit 0): [toggle1 affects bit0?, toggle2 affects bit0?, ...]
        #   Row 1 (bit 1): [toggle1 affects bit1?, toggle2 affects bit1?, ...]
        #   ...
        #   Row 7 (bit 7): [toggle1 affects bit7?, toggle2 affects bit7?, ...]

        width = self.indicators.width
        num_toggles = len(self.toggles)

        # Build matrix: rows = bit positions, columns = toggles
        matrix = []
        for bit_pos in range(width):
            row = []
            for toggle in self.toggles:
                # Check if this toggle affects this bit position
                # Extract the bit at position bit_pos from toggle
                bit_value = (int(toggle) >> (width - 1 - bit_pos)) & 1
                row.append(bit_value)
            matrix.append(row)

        # Now matrix[i][j] = 1 if toggle j affects bit i, else 0

        # Next steps - MISSING THEORY:
        #
        # 1. GAUSSIAN ELIMINATION OVER GF(2):
        #    - Same as regular Gaussian elimination, but:
        #      * Addition = XOR (not regular addition)
        #      * No subtraction needed (XOR is its own inverse)
        #      * No multiplication by scalars (only 0 and 1 exist)
        #    - Goal: Reduce to row-echelon form to find solution space
        #
        # 2. AUGMENTED MATRIX [A|b]:
        #    - Append target vector b as last column
        #    - After elimination, solve for pivot variables
        #
        # 3. FREE VARIABLES:
        #    - Variables (toggles) not in pivot positions are "free"
        #    - Free variables can be 0 or 1 (2^k combinations if k free vars)
        #    - Each combination gives a solution
        #
        # 4. MINIMUM WEIGHT SOLUTION:
        #    - Try all 2^k combinations of free variables
        #    - For each, compute resulting solution vector x
        #    - Count number of 1s in x (Hamming weight)
        #    - Return minimum weight

        # Build target vector b: the indicator pattern itself
        # (machine starts with all lights off, so target = indicators)
        b = []
        for bit_pos in range(width):
            bit_value = (int(self.indicators) >> (width - 1 - bit_pos)) & 1
            b.append(bit_value)

        # Create augmented matrix [A|b]
        aug_matrix = [row + [b[i]] for i, row in enumerate(matrix)]

        # Gaussian elimination over GF(2) to find minimum weight solution
        solution = self._gaussian_elimination_gf2(aug_matrix, num_toggles)

        if solution is None:
            return 0  # No solution (shouldn't happen in valid problems)

        return sum(solution)  # Count number of toggles used

    def _gaussian_elimination_gf2(self, aug_matrix: list[list[int]], num_vars: int) -> list[int] | None:
        """
        Gaussian elimination over GF(2) to solve Ax = b.

        Theory:
        - Operations are XOR (addition in GF(2))
        - No subtraction needed (XOR is self-inverse)
        - Reduce to row-echelon form
        - Identify pivot variables and free variables
        - Find minimum weight solution by trying all free variable combinations

        Returns: Solution vector x (which toggles to use) or None if no solution.
        """
        rows = len(aug_matrix)
        cols = num_vars + 1  # +1 for augmented column (b)

        # Copy matrix to avoid modifying original
        mat = [row[:] for row in aug_matrix]

        # Track which row has which pivot column
        # pivot_row[i] = column index of pivot in row i, or -1 if row i has no pivot
        pivot_row = [-1] * rows

        # Track which column is pivot in which row
        # pivot_col[j] = row index where column j is pivot, or -1 if column j is free
        pivot_col = [-1] * num_vars

        # Forward elimination
        current_row = 0
        for col in range(num_vars):  # Don't process augmented column
            # Find a row with 1 in this column (pivot)
            pivot_row_idx = None
            for r in range(current_row, rows):
                if mat[r][col] == 1:
                    pivot_row_idx = r
                    break

            if pivot_row_idx is None:
                # No pivot in this column - this is a free variable
                continue

            # Swap pivot row to current position
            if pivot_row_idx != current_row:
                mat[current_row], mat[pivot_row_idx] = mat[pivot_row_idx], mat[current_row]

            # Record this pivot
            pivot_row[current_row] = col
            pivot_col[col] = current_row

            # Eliminate this column from all other rows (XOR operation)
            for r in range(rows):
                if r != current_row and mat[r][col] == 1:
                    # XOR row r with pivot row
                    for c in range(cols):
                        mat[r][c] ^= mat[current_row][c]

            current_row += 1

        # Check for inconsistency (0 = 1)
        for r in range(current_row, rows):
            if mat[r][num_vars] == 1:  # Non-zero in augmented column
                # Check if left side is all zeros
                if all(mat[r][c] == 0 for c in range(num_vars)):
                    return None  # No solution

        # Identify free variables (columns that are not pivots)
        free_vars = [col for col in range(num_vars) if pivot_col[col] == -1]

        # Find minimum weight solution by trying all free variable combinations
        min_weight = float("inf")
        best_solution = None

        # Try all 2^k combinations of free variables
        num_free = len(free_vars)
        for free_combination in range(1 << num_free):  # 2^num_free combinations
            # Build solution vector
            x = [0] * num_vars

            # Set free variables
            for i, free_col in enumerate(free_vars):
                x[free_col] = (free_combination >> i) & 1

            # Solve for pivot variables (back substitution)
            # Process rows with pivots in reverse order
            for r in range(current_row - 1, -1, -1):
                col = pivot_row[r]
                if col == -1:
                    continue

                # Compute what this variable should be
                # b[r] = sum of (A[r][c] * x[c]) for all c
                # So: x[col] = b[r] XOR sum of (A[r][c] * x[c]) for c != col
                result = mat[r][num_vars]  # Start with b value
                for c in range(num_vars):
                    if c != col and mat[r][c] == 1:
                        result ^= x[c]
                x[col] = result

            # Check if this solution is valid (satisfies all equations)
            # Use original augmented matrix for validation
            valid = True
            for r in range(rows):
                lhs = 0
                for c in range(num_vars):
                    if aug_matrix[r][c] == 1:
                        lhs ^= x[c]
                if lhs != aug_matrix[r][num_vars]:
                    valid = False
                    break

            if valid:
                weight = sum(x)
                if weight < min_weight:
                    min_weight = weight
                    best_solution = x

        return best_solution


class Factory:
    def __init__(self, problems: Iterable[Problem], task_num: int):
        self.problems = problems
        if _logger.level <= logging.DEBUG:
            # make sure parsing part2 returns same problems len as part1 with an eye
            problems_copy = list(problems)
            _logger.debug("len(problems) = %d", len(problems_copy))
            self.problems = iter(problems_copy)

        if task_num not in (1, 2):
            raise ValueError(f"Invalid task number: {task_num}")
        self.task_num = task_num

    def __repr__(self):
        return f"{self.__class__.__qualname__}(problems={self.problems}, task_num={self.task_num})"

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        return cls(cls._parse_input(inp, task_num), task_num)

    @classmethod
    def _parse_line(cls, line: str) -> tuple[BitInt, list[BitInt], set[int]]:
        indicators_str: str | None = None
        toggles: list[tuple[int, ...]] = []
        joltage_str: str | None = None

        for m in _rexp.finditer(line):
            if m.group("indicators") is not None:
                indicators_str = m.group("indicators")
            elif m.group("toggle") is not None:
                toggle_raw = m.group("toggle")  # e.g. "1,3" or "3"
                toggles.append(tuple(map(int, toggle_raw.split(","))))
            elif m.group("joltage") is not None:
                joltage_str = m.group("joltage")  # e.g. "3,5,4,7"

        if indicators_str is None or joltage_str is None:
            raise ValueError(f"Line not in expected format: {line!r}")

        indicators_strings = list(indicators_str)  # ".##." -> ['.', '#', '#', '.']
        indicators_bits_list = [Indicator(value).value for value in indicators_strings]
        indicators = bits_to_int(indicators_bits_list)

        toggles_integers = []
        for toggle in toggles:
            int_val = 0
            for ind in toggle:
                power = len(indicators_strings) - ind - 1
                int_val += 2**power
            toggles_integers.append(int_val)

        # "3,5,4,7" -> {3, 4, 5, 7}
        joltage = set(map(int, joltage_str.split(",")))

        width = len(indicators_strings)
        return (BitInt(indicators, width=width), [BitInt(x, width=width) for x in toggles_integers], joltage)

    @classmethod
    def _parse_input(cls, inp: list[str], task_num: int) -> Iterable[Problem]:
        for line in inp:
            m = _rexp.match(line)
            if not m:
                raise ValueError("no match")

            indicators, toggles, joltage = cls._parse_line(line)
            yield Problem(indicators, toggles, joltage, task_num)

    def solve(self) -> int:
        return sum(problem.solve() for problem in self.problems)


def task(inp: list[str], task_num: int = 1) -> int:
    return Factory.from_multiline_input(inp, task_num).solve()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


if __name__ == "__main__":
    test(
        task1,
        2,
        test_data=[
            "[#....###] (1,2,4,5,6,7) (0,1,2,3,7) (1,2,3,4,5) (0,2,5,6,7) (0,1,2,3,5) (0,3,4,5,6) (3,4) (2) (0,1,2,3,5,7) {140,165,188,164,53,44,29,153}"
        ],
    )

    test(task1, 7)
    run(task1)
