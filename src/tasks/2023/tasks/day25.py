"""
--- Day 25: Snowverload ---
https://adventofcode.com/2023/day/25
"""
import dataclasses
from collections import defaultdict
from itertools import combinations

import networkx as nx

from src.utils.test_and_run import run, test


class Snowverload:
    def __init__(self, inp, area=None):
        self.area = area
        self.graph, self.nx_graph = self._parse_input(inp)

    @staticmethod
    def _parse_input(inp):
        nx_graph = nx.Graph()

        graph = defaultdict(set)
        for line in inp:
            node, neighbors = line.split(":")
            neighbors = [n.strip() for n in neighbors.split()]

            nx_graph.add_node(node)

            for neighbor in neighbors:
                graph[node].add(neighbor)
                graph[neighbor].add(node)

                nx_graph.add_node(neighbor)
                nx_graph.add_edge(node, neighbor)

        return graph, nx_graph

    def disconnect_and_multiply(self):
        centrality = nx.edge_betweenness_centrality(self.nx_graph)
        # Its critical to sort edges by centrality first
        sorted_edges = sorted(self.nx_graph.edges(), key=lambda edge: centrality[edge], reverse=True)

        # Try removing sets of three edges
        for edges in combinations(sorted_edges, 3):
            # Create a copy of the graph for each trial
            G = self.nx_graph.copy()
            G.remove_edges_from(edges)

            # Check if the graph is now disconnected into two components of specific sizes
            if nx.is_connected(G):
                continue

            components = list(nx.connected_components(G))
            if len(components) == 2:
                return len(components[0]) * len(components[1])

        return None


def snowverload(inp):
    return Snowverload(inp).disconnect_and_multiply()


if __name__ == "__main__":
    test(snowverload, expected=54)
    run(snowverload)
