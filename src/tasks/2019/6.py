"""
--- Day 6: Universal Orbit Map ---

https://adventofcode.com/2019/day/6

"""

import os
import re

import networkx


class UniversalOrbitMap:
    _inp_rexp = re.compile(r'(\w+)\)(\w+)', re.MULTILINE)

    def __init__(self, inp=None, mode=None):
        if inp is None:
            with open(os.path.join('inputs', '{}.txt'.format(__file__.split('/')[-1].split('.')[0]))) as f:
                inp = f.read()

        edges = [self._inp_rexp.match(line.strip()).groups() for line in inp.split('\n') if line.strip()]
        graph_class = networkx.DiGraph if not mode else networkx.Graph

        self.graph = graph_class()
        self.graph.add_edges_from(edges)

    def orbit_count_checksums(self):
        return sum(len(networkx.descendants(self.graph, node)) for node, nbrsdict in self.graph.adjacency())

    def get_moves_to_santa(self):
        return len(networkx.shortest_path(self.graph, source='YOU', target='SAN')) - 3


inp1 = """
    COM)B
    B)C
    C)D
    D)E
    E)F
    B)G
    G)H
    D)I
    E)J
    J)K
    K)L
"""

inp2 = """
    COM)B
    B)C
    C)D
    D)E
    E)F
    B)G
    G)H
    D)I
    E)J
    J)K
    K)L
    K)YOU
    I)SAN
"""


def test(test_num):
    if test_num == 1:
        res = UniversalOrbitMap(inp1).orbit_count_checksums()
        assert res == 42, 'test{} failed!: {}'.format(test_num, res)
    elif test_num == 2:
        res = UniversalOrbitMap(inp2, mode=2).get_moves_to_santa()
        assert res == 4, 'test{} failed!: {}'.format(test_num, res)
    else:
        raise ValueError('test{} not implemented'.format(test_num))
    return 'test{} ok'.format(test_num)


def part1():
    return UniversalOrbitMap().orbit_count_checksums()


def part2():
    return UniversalOrbitMap(mode=2).get_moves_to_santa()


if __name__ == '__main__':
    for res in (
        test(1),
        part1(),
        test(2),
        part2(),
    ):
        print(res)
