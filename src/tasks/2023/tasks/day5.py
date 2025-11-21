"""
--- Day 5: If You Give A Seed A Fertilizer  ---
https://adventofcode.com/2023/day/5
"""
from src.utils.test_and_run import run, test


def print(*_):
    pass


def parse_almanac(inp, part=1):
    seeds = [int(x.strip()) for x in inp[0].split(":")[1].strip().split()]
    last_row_idx = 2

    seeds_section = []
    if part == 2:
        s = None
        for i, x in enumerate(seeds):
            if not i % 2:
                s = x
            else:
                length_ = x
                # finalize
                seeds_section.append((s, s + length_, 0))
    else:
        # seeds_section = {seed: seed for seed in seeds}
        for seed in seeds:
            v = (seed, seed, 0)
            seeds_section.append(v)
    seeds_section = sorted(seeds_section)

    res = [seeds_section]
    section_idx = 0
    prev_section = seeds_section
    section = prev_section.copy()
    print(fr"seed {section=}")
    for i, line in enumerate(inp):

        if i < last_row_idx:
            continue
        last_row_idx += 1

        if not line:
            section_idx += 1
            prev_section = section
            res.append(section)
            section = prev_section.copy()
            # shift destinations
            for i in range(len(section)):
                start, end, shift = section[i]
                section[i] = (start + shift, end + shift, 0)

            print("")
            print(fr"{prev_section=}")
            print(fr"{section=}")
            _ = 0
            continue

        if ":" in line:
            print("===========")
            print(line)
            print("===========")
            # header
            continue

        destination_start, source_start, length = map(int, line.split())
        source_end = source_start + length
        shift = destination_start - source_start
        print(f"search for replacements for line {i} ({line}): {source_start}-{source_end}, {shift}")
        # merge transformations with new line
        replacements = {}  # idx to new intervals
        for prev_i, (prev_src_start, prev_src_end, prev_shift) in enumerate(section):
            r = None
            if source_start <= prev_src_start and source_end >= prev_src_end:
                # fully replace existing section
                r = [
                    (prev_src_start, prev_src_end, shift)
                ]
            elif source_start >= prev_src_start and source_end <= prev_src_end:
                # fully fits inside existing section
                r = [
                    (prev_src_start, source_start, prev_shift),
                    (source_start, source_end, shift),
                    (source_end, prev_src_end, prev_shift),
                ]
            elif source_start <= prev_src_start < source_end:
                # starts at the left but end overlaps prev at the end
                r = [
                    (prev_src_start, source_end, shift),
                    (source_end, prev_src_end, prev_shift),
                ]
            elif source_start <= prev_src_end < source_end:
                # from the middle of prev
                r = [
                    (prev_src_start, source_start, prev_shift),
                    (source_start, prev_src_end, shift),
                ]
            if r:
                print(f"add replacements for {prev_i=}, {r}")
                replacements[prev_i] = r

        # do section replacements
        section = [s for i, s in enumerate(section) if i not in replacements]
        for old, new_list in replacements.items():
            for new in new_list:
                assert new not in section
                print(f"add segment {new=}")
                section.append(new)
        section = sorted(section)
        if replacements:
            print(f"replacements for line {i} complete: {section=}")

    if section:
        print(f"Adding final section: {section=}")
        res.append(section)
    return res


def fertilizer(inp, part=1):
    almanac = parse_almanac(inp, part=part)

    min_loc = float("inf")
    for start, end, shift in almanac[-1]:
        min_loc = min(min_loc, start + shift)
    return min_loc


if __name__ == "__main__":
    test(fertilizer, expected=35)
    run(fertilizer)

    test(fertilizer, part=2, expected=46)
    run(fertilizer, part=2)
