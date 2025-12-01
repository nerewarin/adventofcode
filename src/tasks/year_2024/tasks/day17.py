"""
--- Day 17: Chronospatial Computer ---
https://adventofcode.com/2024/day/17

"""

from src.utils.logger import get_logger
from src.utils.test_and_run import run, test

_logger = get_logger()


class ChronospatialComputer:
    def __init__(self, program, a=0, b=0, c=0):
        self.a, self.b, self.c = a, b, c
        self.program = program
        if len(self.program) % 2:
            raise ValueError(f"program length must be even but passed value is {self.program}")
        self.output = []

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"(program={self.program}, a={self.a}, b={self.b}, c={self.c}, output={self.output})"
        )

    @classmethod
    def from_multiline_input(cls, inp, task_num):
        a, b, c, program = cls._parse_input(inp)
        return cls(program, a, b, c)

    @staticmethod
    def _parse_input(inp: list[str]) -> tuple[int, int, int, list[int]]:
        raw_values = []
        for line in inp:
            if line:
                raw_values.append(line.split(":")[-1].strip())

        *raw_registers, raw_program = raw_values
        return *map(int, raw_registers), list(map(int, raw_program.split(",")))

    @property
    def _registers(self) -> tuple[int]:
        return self.a, self.b, self.c

    def _get_combo(self, value: int) -> int:
        if value <= 3:
            return value
        return self._registers[value - 4]

    def _get_division(self, operand, op):
        numerator = self.a
        combo = self._get_combo(operand)
        denominator = 2**combo
        a = numerator // denominator
        _logger.debug(f"{op}: {numerator=} // {denominator=} = {a} written to {op[0].capitalize()}")
        return a

    def run_program(self) -> str:
        instruction_pointer = 0

        program = self.program
        while instruction_pointer < len(program):
            instruction, operand = program[instruction_pointer : instruction_pointer + 2]
            _logger.debug(f"{instruction=}, {operand=}")

            to_increment_pointer = True
            match instruction:
                case 0:
                    # The adv instruction (opcode 0) performs division
                    op = "op"
                    self.a = self._get_division(operand, op)
                case 1:
                    # The bxl instruction (opcode 1) calculates the bitwise XOR of register B
                    # and the instruction's literal operand, then stores the result in register B.
                    b = self.b ^ operand
                    _logger.debug(f"bxl: {self.b=} ^ {operand=} = {b} written to B")
                    self.b = b
                case 2:
                    combo = self._get_combo(operand)
                    b = combo % 8
                    _logger.debug(f"bst: modulo 8 of {combo=} is {b=} written to B")
                    self.b = b
                case 3:
                    if self.a == 0:
                        _logger.debug("jnz: self.a == 0, no jump")
                        pass
                    else:
                        new_instruction_pointer = operand
                        if new_instruction_pointer != instruction_pointer:
                            to_increment_pointer = False
                            _logger.debug(
                                f"jnz: shifting instruction_pointer from {instruction_pointer} to {new_instruction_pointer}"
                            )
                            instruction_pointer = new_instruction_pointer
                        else:
                            _logger.debug(f"jnz: already on {instruction_pointer=}")
                case 4:
                    # he bxc instruction (opcode 4) calculates the bitwise XOR of register B and register C,
                    # then stores the result in register B. (For legacy reasons, this instruction reads
                    # an operand but ignores it.)
                    op = "bxc"
                    b = self.b ^ self.c
                    _logger.debug(f"{op}: {self.b=} ^ {self.c=} = {b} written to B")
                    self.b = b
                case 5:
                    # The out instruction (opcode 5) calculates the value of its combo operand modulo 8,
                    # then outputs that value. (If a program outputs multiple values, they are separated by commas.)
                    op = "out"
                    combo = self._get_combo(operand)
                    to_output = combo % 8
                    _logger.debug(f"{op}: {to_output=} written to output")
                    self.output.append(to_output)
                case 6:
                    # The bdv instruction (opcode 6) works exactly like the adv instruction except that the result
                    # is stored in the B register. (The numerator is still read from the A register.)
                    op = "bdv"
                    self.b = self._get_division(operand, op)
                case 7:
                    op = "cdv"
                    self.c = self._get_division(operand, op)

            if to_increment_pointer:
                instruction_pointer += 2

        return ",".join(str(x) for x in self.output)


def task(inp: list[str], task_num: int = 1) -> str:
    return ChronospatialComputer.from_multiline_input(inp, task_num).run_program()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp):
    return task(inp, task_num=2)


def test1():
    """If register C contains 9, the program 2,6 would set register B to 1."""
    state = ChronospatialComputer([2, 6], c=9)
    state.run_program()
    assert state.b == 1, state
    _logger.debug("test1 passed")


def test2():
    """If register A contains 10, the program 5,0,5,1,5,4 would output 0,1,2."""
    state = ChronospatialComputer([5, 0, 5, 1, 5, 4], a=10)
    state.run_program()
    assert state.output == [0, 1, 2], state
    _logger.debug("test2 passed")


def test3():
    """If register A contains 2024, the program 0,1,5,4,3,0 would output 4,2,5,6,7,7,7,7,3,1,0
    and leave 0 in register A."""
    state = ChronospatialComputer([0, 1, 5, 4, 3, 0], a=2024)
    state.run_program()
    assert state.output == [4, 2, 5, 6, 7, 7, 7, 7, 3, 1, 0], state
    assert state.a == 0, state
    _logger.debug("test3 passed")


def test4():
    """If register B contains 29, the program 1,7 would set register B to 26."""
    state = ChronospatialComputer([1, 7], b=29)
    state.run_program()
    assert state.b == 26, state
    _logger.debug("test4 passed")


def test5():
    """If register B contains 2024 and register C contains 43690, the program 4,0 would set register B to 44354."""
    state = ChronospatialComputer([4, 0], b=2024, c=43690)
    state.run_program()
    assert state.b == 44354, state
    _logger.debug("test5 passed")


def test6():
    """
    For part2:
    Register A: 2024
    Register B: 0
    Register C: 0

    Program: 0,3,5,4,3,0
    This program outputs a copy of itself if register A is instead initialized to 117440.
    (The original initial value of register A, 2024, is ignored.)."""
    program = [0, 3, 5, 4, 3, 0]
    state = ChronospatialComputer(program, a=117440, b=0, c=0)
    state.run_program()
    assert state.output == program, state
    _logger.debug("test6 passed")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()

    test(task1, "4,6,3,5,6,3,5,2,1,0")
    run(task1)

    test6()
    test(task2, "0,1,5,4,3,0")
    run(task2, "2,4,1,3,7,5,1,5,0,3,4,3,5,5,3,0")
