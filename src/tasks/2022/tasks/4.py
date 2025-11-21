"""---- Day 4: Camp Cleanup ---
https://adventofcode.com/2022/day/4
"""
import re
from src.utils.test_and_run import test, run
_SECTION_REXP = re.compile("(\d+)-(\d+),(\d+)-(\d+)")


def camp_cleanup(inp, overlapping="full"):
    res = 0
    for line in inp:
        raw_sections = _SECTION_REXP.match(line)
        flatten_sections = [int(raw_sections.group(x + 1)) for x in range(4)]
        sections1, sections2 = flatten_sections[:2], flatten_sections[2:]
        if overlapping == "full":
            if sections1[0] <= sections2[0] and sections1[1] >= sections2[1]:
                res += 1
            elif sections2[0] <= sections1[0] and sections2[1] >= sections1[1]:
                res += 1
        elif overlapping == "partial":
            s1 = set(range(sections1[0], sections1[1] + 1))
            s2 = set(range(sections2[0], sections2[1] + 1))
            if s1.intersection(s2):
                res += 1
    return res


if __name__ == "__main__":
    # part 1
    test(camp_cleanup, expected=2)
    run(camp_cleanup)

    # part 2
    test(camp_cleanup, overlapping="partial", expected=4)
    run(camp_cleanup, overlapping="partial")
