def calc(noun, verb):
    with open('inputs/2.txt') as f:
        inp = [int(x) for x in f.read().split(',')]
        inp[1] = noun
        inp[2] = verb

        curr_idx = 0
        while True:
            op = inp[curr_idx]

            if op == 99:
                return inp[0]

            a_idx, b_idx, res_idx = (inp[curr_idx + x] for x in range(1, 4))
            if op == 1:
                inp[res_idx] = inp[a_idx] + inp[b_idx]
            elif op == 2:
                inp[res_idx] = inp[a_idx] * inp[b_idx]

            curr_idx += 4


def part1():
    return calc(12, 2)


def part2():
    for noun in range(100):
        for verb in range(100):
            if calc(noun, verb) == 19690720:
                return 100 * noun + verb


for res in (
    part1(),
    part2(),
):
    print(res)
