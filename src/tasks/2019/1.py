"""
https://adventofcode.com/2019/day/1
"""


def calc_fuel(mass):
    # to find the fuel required for a module, take its mass, divide by three, round down, and subtract 2.
    return mass // 3 - 2


def part1():
    with open("inputs/1.txt") as f:
        res = sum(calc_fuel(int(line)) for line in f.readlines())
        print(res)


def part2():
    with open("inputs/1.txt") as f:
        res = 0
        for line in f.readlines():
            mass = int(line)
            fuel = calc_fuel(mass)
            res += fuel
            while fuel > 0:
                fuel = calc_fuel(fuel)
                if fuel > 0:
                    res += fuel
        print(res)


part1()
part2()
