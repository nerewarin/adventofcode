"""--- Day 7: No Space Left On Device ---
https://adventofcode.com/2022/day/7
"""
import pathlib

from src.utils.test_and_run import test, run
import re

_CMD_REXP = re.compile(r"cd ([\S+])]")


class Dir:
    def __init__(self, name: str, path: pathlib.Path):
        self.name = name
        self.path = path
        self.size = 0
        self.children: list[Dir] = []

    def __str__(self):
        return f"{self.name}   {self.path}"

    def add_file(self, size):
        self.size += size

    def add_subdir(self, directory: "Dir"):
        self.children.append(directory)

    def get_size(self):
        res = self.size
        for subdir in self.children:
            res += subdir.get_size()
        return res


def no_space_left_on_device(inp, part2=None, max_size=100000):
    current_path = pathlib.Path("/")
    current_dir = Dir(current_path.name, current_path)
    visited = {
        current_path: current_dir,
    }

    def step(path):
        if path in visited:
            return visited[path]

    for line in inp:
        if line.startswith("$"):
            line = line[2:]
            if line.startswith("cd"):
                value = line[2:].strip()
                if value == "/":
                    current_path = pathlib.Path("/")
                    current_dir = step(current_path)
                elif value == "..":
                    current_path = current_path.parent
                    current_dir = step(current_path)
                else:
                    current_path = current_path / value
                    current_dir = step(current_path)
            elif line.startswith("ls"):
                pass
            continue

        # else content
        p1, p2 = line.split()
        if p1 == "dir":
            path = current_path / p2
            dr = Dir(p2, path)
            visited[path] = dr
            current_dir.add_subdir(dr)
        else:
            current_dir.add_file(int(p1))

    if not part2:
        res = 0
        for dr in visited.values():
            size = dr.get_size()
            if size <= max_size:
                res += size

        return res

    smallest = 70000000
    total = step(pathlib.Path("/")).get_size()
    limit = total - 40000000
    if limit <= 0:
        return 0

    for path, dr in visited.items():
        size = dr.get_size()
        if limit <= size <= smallest:
            smallest = size

    return smallest


if __name__ == "__main__":
    task = no_space_left_on_device

    # part 1
    test(task, expected=95437)
    run(task)

    # part 2
    test(task, part2=True, expected=24933642)
    run(task, part2=True)
