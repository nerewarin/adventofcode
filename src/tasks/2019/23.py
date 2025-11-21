"""
--- Day 23: Category Six ---

https://adventofcode.com/2019/day/23

"""

import re
import collections

from _intcode_computer import IntcodeComputer


class LongIDLE(Exception):
    pass


class IntcodeComputer23(IntcodeComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idle_cycles = 0
        self.idle = False

    def feed(self, value):
        super().feed(value)
        self._idle_cycles = 0
        self.idle = False

    def _get_op3_input(self):
        if self.signals:
            val = self.signals.pop(0)
            print(f'used {val}')
            self._idle_cycles = 0
            return val
        print('used default')
        self._idle_cycles += 1
        return -1

    def _on_step_start(self):
        if self._idle_cycles > 5:
            self.gen = self._compute()
            self.idle = True
            raise LongIDLE()


class NAT:
    def __init__(self):
        self.packet = collections.deque([], maxlen=2)

    def feed(self, value):
        self.packet.append(value)

    def __next__(self):
        return self.packet.popleft()

    def __str__(self):
        return f'NAT {self.packet}'


class CategorySix:
    _nat_address = 255

    def __init__(self, *args, network_size=50, **kwargs):
        self.network_size = network_size
        self.network = {
            computer_num: IntcodeComputer23(signals=[computer_num]) for computer_num in range(network_size)
        }
        self.network[self._nat_address] = NAT()

    @property
    def nat(self):
        return self.network[self._nat_address]

    def get_first_packet_to(self, expected_address):
        address = None
        computer_num = 0
        while address != expected_address:
            # receive
            print(f'receive from {computer_num}:')
            computer = self.network[computer_num]

            try:
                address = next(computer)
            except LongIDLE:
                print('IDLE address', end='\n\n')
                computer_num = (computer_num + 1) % self.network_size
                continue
            except StopIteration:
                print('StopIteration address', end='\n\n')
                computer_num = (computer_num + 1) % self.network_size
                continue

            try:
                x = next(computer)
            except LongIDLE:
                print('IDLE x', end='\n\n')
                computer_num = (computer_num + 1) % self.network_size
                continue

            y = next(computer)
            print(f'{address} {x} {y}')

            if address == expected_address:
                return y

            # send
            next_computer = self.network[address]
            next_computer.feed(x)
            next_computer.feed(y)
            print(f'send to {address}: {x} {y}', end='\n\n')

            computer_num = address

    def get_first_duplicate_y_from_NAT_to(self, nat_output_address):
        computer_num = 0
        last_NAT_y = None
        while True:
            # receive
            print(f'receive from {computer_num}:')
            computer = self.network[computer_num]
            if computer is self.nat:
                address = nat_output_address
            else:
                try:
                    address = next(computer)
                except LongIDLE:
                    print('IDLE address', end='\n\n')
                    if all(self.network[idx].idle for idx in range(self.network_size)):
                        print('network in IDLE')
                        computer_num = self._nat_address
                    else:
                        computer_num = (computer_num + 1) % self.network_size
                    continue
                except StopIteration:
                    print('StopIteration address', end='\n\n')
                    computer_num = (computer_num + 1) % self.network_size
                    continue

            x = next(computer)
            y = next(computer)
            print(f'{address} {x} {y}')

            # send
            next_computer = self.network[address]
            next_computer.feed(x)
            next_computer.feed(y)
            print(f'send to {address}: {x} {y}', end='\n\n')

            if computer is self.nat:
                if last_NAT_y == y:
                    return y
                last_NAT_y = y

            if next_computer is self.nat:
                computer_num = (computer_num + 1) % self.network_size
            else:
                computer_num = address


def part1(*args, **kwargs):
    return CategorySix(*args, **kwargs).get_first_packet_to(255)


def part2(*args, **kwargs):
    return CategorySix(*args, **kwargs).get_first_duplicate_y_from_NAT_to(0)


def test_nat():
    nat = NAT()
    for x in range(10):
        nat.feed(x)

    a, b = nat.packet
    assert a, b == (9, 8)

    a = next(nat)
    assert a == 9
    assert list(nat.packet) == [8], nat.packet

    b = next(nat)
    assert b == 8
    assert list(nat.packet) == [], nat.packet

if __name__ == '__main__':
    assert part1() == 15969
    assert part2() == 10650
