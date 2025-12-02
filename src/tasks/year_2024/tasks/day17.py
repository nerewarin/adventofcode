"""
--- Day 17: Chronospatial Computer ---
https://adventofcode.com/2024/day/17

"""

from src.utils.logger import get_logger
from src.utils.test_and_run import run

_logger = get_logger()


def pretty_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")


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
    def _registers(self) -> tuple[int, int, int]:
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
        counter = 0

        program = self.program
        while instruction_pointer < len(program):
            instruction, operand = program[instruction_pointer : instruction_pointer + 2]
            to_increment_pointer = True
            _logger.debug(f"{counter=}. {instruction=}, {operand=}")
            match instruction:
                case 0:
                    # The adv instruction (opcode 0) performs division in register A by combo of operator
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
                                f"jnz: a={self.a}. shifting instruction_pointer from {instruction_pointer} to {new_instruction_pointer}"
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
            counter += 1

        return ",".join(str(x) for x in self.output)

    def analyze_backward(self):
        reversed_program = self.program[::-1]
        expected_output = reversed_program

        self.a = 0
        self.b = None
        self.c = None
        instruction_pointer = 0
        cycle = 1
        counter = 1
        # registers constraints
        # constraints = [[], [], []]
        for output_value in expected_output:
            while instruction_pointer < len(reversed_program):
                to_increment_pointer = True
                operand, instruction = reversed_program[instruction_pointer : instruction_pointer + 2]
                _logger.debug(f"step={cycle}.{counter // (len(reversed_program) // 2)}. {instruction=}, {operand=}")
                match instruction:
                    case 3:
                        if cycle == 1:
                            assert self.a == 0
                            _logger.debug("jnz: self.a == 0, no jump")
                            pass
                        else:
                            # TODO
                            new_instruction_pointer = operand
                            if new_instruction_pointer != instruction_pointer:
                                to_increment_pointer = False
                                _logger.debug(
                                    f"jnz: a={self.a}. shifting instruction_pointer from {instruction_pointer} to {new_instruction_pointer}"
                                )
                                instruction_pointer = new_instruction_pointer
                            else:
                                _logger.debug(f"jnz: already on {instruction_pointer=}")
                    case 5:
                        to_output = output_value
                        # The out instruction (opcode 5) calculates the value of its combo operand modulo 8,
                        # then outputs that value. (If a program outputs multiple values, they are separated by commas.)
                        op = "out"

                        combo = self._get_combo(operand)
                        to_output = combo % 8
                        _logger.debug(f"{op}: {to_output=} written to output")
                        self.output.append(to_output)
                if to_increment_pointer:
                    instruction_pointer += 2
                counter += 1
            print(output_value)


def task(inp: list[str], task_num: int = 1) -> str:
    return ChronospatialComputer.from_multiline_input(inp, task_num).run_program()


def task1(inp, **kw):
    return task(inp, **kw)


def task2(inp, min_a=None, max_a=None, delta=None, matches_to_print=None, counter_to_pint_matches=1_000):
    """
    Now we need to debug program in reverse.
    consider input from run:

    Register A: 47006051 (??)
    Register B: 0
    Register C: 0

    Program: 2,4,1,3,7,5,1,5,0,3,4,3,5,5,3,0

    first cycle:
    3,0
    this will stop if only A=0 to this point.
    A=0
    B=?
    C=?

    5,5
    B % 8 -> output
    Since output must be 0, B must be divisible by 8
    A=0
    B % 8 == 0 like ...000
    C=?

    4,3
    C ^ B -> B
    from prior step we know that new B must be divisible by 8
    A=0
    B after is ...000 but now - depends on C!
    C = B ^ old_B - depends on B now

    0,3
    A // (2**3) = A // 8 -> A
    so
    A < 8
    B % 8 == 0 like ...000

    1,5
    B ^ 5 = B ^ 101 -> B
    we know after that, B must have the last 3 bits as zero (omitting depending on C tho), but can be big like ...000
    so at that point B is like
    A < 8
    B = ...101

    7,5
    A // (2**B) -> C
    B = ...101
    C = A // (2**B) and from prior step (4,3): B ^ old_B

    1,3
    B ^ 3 = B ^ 011 -> B

    2,4
    A // 8 -> B

    and before that we must jump somewhere (presumably into 2,4) having A != ...000

    then  B % 8 -> output must be 3 now: so B = xxx011

    """
    _initial_computer = ChronospatialComputer.from_multiline_input(inp, 1)
    program = _initial_computer.program

    expected_len = len(program)

    counter = 0
    matches = 0

    a = min_a
    _logger.info(
        f"searching for A between min_a={pretty_int(min_a)} and max_a={pretty_int(max_a)}: delta={max_a - min_a}"
    )
    while a < max_a:
        computer = ChronospatialComputer.from_multiline_input(inp, 1)
        computer.a = a
        computer.run_program()

        output = computer.output
        output_len = len(output)

        if output == computer.program:
            # found = a
            _logger.info(f"found {a=}")
            return a

        matches = 0
        if output_len == expected_len:
            # already_found_full_len = True

            for i in range(expected_len):
                if output[-i - 1] == program[-i - 1]:
                    matches += 1
                else:
                    break
        #     min_step = (expected_len - matches) ** 10
        #     if min_step:
        #         import random
        #         entropy = random.randrange(1, 11)
        #         min_step = min_step * entropy // 10
        #
        #     if matches < (prior_matches - 1):
        #         delta = -(max(delta // 2, min_step))
        #     elif matches > prior_matches:
        #         delta = max(delta // 2, min_step)
        #     else:
        #         next_digit_to_find = program[-matches - 1]
        #         our_digit = output[-matches - 1]
        #         if our_digit < next_digit_to_find:
        #             delta = abs(delta)
        #         elif our_digit > next_digit_to_find:
        #             delta = -abs(delta)
        #         else:
        #             raise RuntimeError()
        #
        #     prior_matches = matches
        #
        # elif output_len < expected_len:
        #     if already_found_full_len:
        #         if min_a is None:
        #             min_a = a
        #         else:
        #             min_a = max(min_a, a)
        #     delta = (a or 1) * 2
        # else:
        #     delta = int(-(a or 1) * 0.1)

        # delta += 1000
        delta = delta or 10_000
        if matches_to_print and matches >= matches_to_print and not counter % counter_to_pint_matches:
            _logger.info(
                f"matches! {counter=}. a={pretty_int(a)}, {output=}, {output_len=}, {expected_len=}, {delta=}. {matches=}"
            )
        elif not counter % 10_000:
            _logger.info(
                f"{counter=}. a={pretty_int(a)}, {output=}, {output_len=}, {expected_len=}, {delta=}. {matches=}"
            )

        a += delta
        # if min_a is None:
        #     a += delta
        # else:
        #     if output_len < expected_len:
        #         a = min(a + delta, min_a + 1)
        counter += 1

    _logger.info(
        f"lastly, {counter=}. a={pretty_int(a)}, {output=}, {output_len=}, {expected_len=}, {delta=}. {matches=}"
    )
    _logger.info(f"(False) Done searching for A between {min_a=} and {max_a=}: delta={max_a - min_a}")
    return False


class UnitTest:
    @staticmethod
    def test1():
        """If register C contains 9, the program 2,6 would set register B to 1."""
        state = ChronospatialComputer([2, 6], c=9)
        state.run_program()
        assert state.b == 1, state
        _logger.debug("test1 passed")

    @staticmethod
    def test2():
        """If register A contains 10, the program 5,0,5,1,5,4 would output 0,1,2."""
        state = ChronospatialComputer([5, 0, 5, 1, 5, 4], a=10)
        state.run_program()
        assert state.output == [0, 1, 2], state
        _logger.debug("test2 passed")

    @staticmethod
    def test3():
        """If register A contains 2024, the program 0,1,5,4,3,0 would output 4,2,5,6,7,7,7,7,3,1,0
        and leave 0 in register A."""
        state = ChronospatialComputer([0, 1, 5, 4, 3, 0], a=2024)
        state.run_program()
        assert state.output == [4, 2, 5, 6, 7, 7, 7, 7, 3, 1, 0], state
        assert state.a == 0, state
        _logger.debug("test3 passed")

    @staticmethod
    def test4():
        """If register B contains 29, the program 1,7 would set register B to 26."""
        state = ChronospatialComputer([1, 7], b=29)
        state.run_program()
        assert state.b == 26, state
        _logger.debug("test4 passed")

    @staticmethod
    def test5():
        """If register B contains 2024 and register C contains 43690, the program 4,0 would set register B to 44354."""
        state = ChronospatialComputer([4, 0], b=2024, c=43690)
        state.run_program()
        assert state.b == 44354, state
        _logger.debug("test5 passed")

    @staticmethod
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
    # UnitTest.test1()
    # UnitTest.test2()
    # UnitTest.test3()
    # UnitTest.test4()
    # UnitTest.test5()
    #
    # test(task1, "4,6,3,5,6,3,5,2,1,0")
    run(task1)
    #
    # UnitTest.test6()
    # test(task2, True)
    # test(task2, True, test_part=2)

    # well, for the task2 I've just used brute-force to estimate how output growths, then picked up ranges of A for
    # which output is at least needed length and then used gradient descent

    # First, I did full run from sublime to find at least len_putput match, see day_17_research.txt
    # run(task2, min_a=237_755_333_196_000, max_a=287_755_333_196_000, delta=100_000_000, matches_to_print=5)  # full matches

    # here are some insights from it...
    # 5 matches
    # run(task2, min_a=216580542908000, max_a=216600542908000)
    # run(task2, min_a=215250542908000, max_a=216250542908000)

    # 7 matches
    # run(task2, min_a=216591542908000, max_a=216593542908000)

    # 9 matches
    # run(task2, min_a=216592042908000, max_a=216592072908000, delta=1000)
    # run(task2, min_a=216592982908000, max_a=216593012908000, delta=1000)

    # run(task2, min_a=216592042908000, max_a=216592062908000, delta=100)
    # run(task2, min_a=216592982908000, max_a=216592995908000, delta=100)

    # run(task2, min_a=216592049908000, max_a=216592053908000, delta=10)
    # run(task2, min_a=216592989908000, max_a=216592992908000, delta=10)
    #
    # run(task2, min_a=216592050808000, max_a=216592053008000, delta=5)
    # run(task2, min_a=216592990308000, max_a=216592992508000, delta=5)

    # but I haven't found anything in these regions...so I run it again with more precise steps not to miss smth...
    # from this I got new regions:
    # 1. matches=5
    # 216550542908000...216650542908000
    # 234150542908000...234250542908000
    # run(task2, min_a=215_250_542_908_000, max_a=237_755_333_196_000, delta=10_000_000, matches_to_print=5)
    # 2. matches = 4
    # 216050542908000 ,,,216250542908000
    # 218250542908000 ,,,218450542908000
    # 233650542908000 ,,,233850542908000
    # 235850542908000 ..236050542908000
    # 236450542908000 .. 236650542908000

    # 1. inspect 5 matches
    # run(task2, min_a=216550542908000, max_a=216650542908000, delta=1_000_000, matches_to_print=6)
    # run(task2, min_a=234150542908000, max_a=234250542908000, delta=1_000_000, matches_to_print=6)
    # inspect 6 matches from this
    # run(task2, min_a=216_590_542_908_000, max_a=216_600_542_908_000, delta=100_000, matches_to_print=7)
    # run(task2, min_a=234_180_542_908_000, max_a=234_190_542_908_000, delta=100_000, matches_to_print=7)
    # inspect 7 matches from this
    # run(task2, min_a=216_591_542_908_000, max_a=216_593_542_908_000, delta=10_000, matches_to_print=8)
    # run(task2, min_a=234_183_542_908_000, max_a=234_185_542_908_000, delta=10_000, matches_to_print=8)
    # inspect 8 matches from this
    # run(task2, min_a=216_592_042_908_000, max_a=216_592_142_908_000, delta=1_000, matches_to_print=9)
    # run(task2, min_a=216_592_942_908_000, max_a=216_593_042_908_000, delta=1_000, matches_to_print=9)
    # run(task2, min_a=234_184_142_908_000, max_a=234_184_342_908_000, delta=1_000, matches_to_print=9)
    # run(task2, min_a=234_185_142_908_000, max_a=234_185_242_908_000, delta=1_000, matches_to_print=9)
    # inspect 9 matches from this
    # run(task2, min_a=216_592_042_908_000, max_a=216_592_062_908_000, delta=100, matches_to_print=10)
    # run(task2, min_a=216_592_982_908_000, max_a=216_593_002_908_000, delta=100, matches_to_print=10)
    # run(task2, min_a=234_184_232_908_000, max_a=234_184_242_908_000, delta=100, matches_to_print=10)
    # run(task2, min_a=234_185_172_908_000, max_a=234_185_182_908_000, delta=100, matches_to_print=10)
    # only 2 sections have 10 matches from this, pick best from each
    # run(task2, min_a=216_592_049_908_000, max_a=216_592_053_908_000, delta=10, matches_to_print=10)
    # run(task2, min_a=216_592_989_908_000, max_a=216_592_992_908_000, delta=10, matches_to_print=10)
    # run(task2, min_a=216_592_995_908_000, max_a=216_592_996_908_000, delta=10, matches_to_print=10)
    # run(task2, min_a=234_184_235_908_000, max_a=234_184_239_908_000, delta=10, matches_to_print=10)
    # run(task2, min_a=234_185_175_908_000, max_a=234_185_178_908_000, delta=10, matches_to_print=10)
    # run(task2, min_a=234_185_179_908_000, max_a=234_185_182_408_000 + 1000, delta=10, matches_to_print=10)
    # again, narrow the sectors
    # run(task2, min_a=216_592_050_808_000, max_a=216_592_052_908_000, delta=5, matches_to_print=10) # only 9 there
    # run(task2, min_a=216_592_990_308_000, max_a=216_592_992_508_000, delta=5, matches_to_print=10) # only 9 there
    #
    # run(task2, min_a=216_592_996_168_000, max_a=216_592_996_208_000, delta=1, matches_to_print=10) # more hopes on that, even 11 found!
    # run(task2, min_a=216_592_996_008_000, max_a=216_592_996_408_000, delta=5, matches_to_print=10) # same section wider
    #
    # run(task2, min_a=234_184_236_808_000, max_a=234_184_238_908_000, delta=5, matches_to_print=10) # only 9 there
    #
    # run(task2, min_a=234_185_176_308_000, max_a=234_185_178_508_000, delta=5, matches_to_print=10) # only 9 there
    #
    # run(task2, min_a=234_185_182_108_000, max_a=234_185_182_408_000 + 200_000, delta=5, matches_to_print=10) # 10
    # again, narrow the sectors (2)
    # no progress between min_a=216592050808000 and max_a=216592052908000: delta=2100000
    # no progress between min_a=216592050808000 and max_a=216592052908000: delta=2100000
    # ...
    # run only where 11 matches found
    # run(task2, min_a=216_592_996_171_000, max_a=216_592_996_205_000, delta=1, matches_to_print=12, counter_to_pint_matches=1)
    # run(task2, min_a=234_185_182_213_000, max_a=234_185_182_253_000, delta=1, matches_to_print=12, counter_to_pint_matches=1)
    # nothing found :( only 12 matches max - best result is 234185182253000

    # 2. find more precisely another zones with 4 matches
    # for start, end in [
    #     (216050542908000, 216250542908000),
    #     (218250542908000, 218450542908000),
    #     (233650542908000, 233850542908000),
    #     (235850542908000, 236050542908000),
    #     (236450542908000, 236650542908000),
    # ]:
    #     run(task2, min_a=start, max_a=end, delta=1_000_000, matches_to_print=5)
    # ok from last region I found 6 matches, inspect them...damn I lost end :) search again...
    # run(task2, min_a=236450542908000, max_a=236650542908000, delta=1_000_000, matches_to_print=5)
    # run(task2, min_a=236_530_542_908_000, max_a=236_600_542_908_000, delta=100_000, matches_to_print=6) # ook found 7
    # run(task2, min_a=236_547_542_908_000, max_a=236_548_442_908_000, delta=10_000, matches_to_print=7) # ook found 7
    # found 2 regions - 7 and 9
    # run(task2, min_a=236_547_542_908_000, max_a=236_547_742_908_000, delta=1_000, matches_to_print=8) # 7 is max =(
    # run(task2, min_a=236_548_242_908_000, max_a=236_548_302_908_000, delta=100, matches_to_print=10) # 9 and 11 found
    # take 2 regions - with 9 and 11 - from last region
    # run(task2, min_a=236_548_279_908_000, max_a=236_548_283_908_000, delta=100, matches_to_print=10) # still 9
    # run(task2, min_a=236_548_286_908_000, max_a=236_548_289_908_000, delta=10, matches_to_print=11) # still 11

    run(
        task2, min_a=236_548_287_608_000, max_a=236_548_287_908_000, delta=1, matches_to_print=12
    )  # FOUND 236548287712877
    run(task2, min_a=236_548_288_708_000, max_a=236_548_288_808_000, delta=1, matches_to_print=12)  #
