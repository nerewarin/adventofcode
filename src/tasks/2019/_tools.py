import os
import re
import math
import traceback


def get_initiator_fname():
    # define script that has called us - previous py filename from callstack
    for line in reversed(traceback.format_stack()):
        res = re.search(r'File.*"(.*)[\\\/](\d+).py', line)
        if res:
            fname = res.group(2)
            return os.path.join('inputs', '{}.txt'.format(fname))
    raise ValueError()


def get_puzzle_input(scalar_type=int, delimeter=',', multiline=False, strip=True):
    fpath = get_initiator_fname()
    print('reading {}'.format(fpath))
    with open(fpath, 'r') as f:
        if multiline:
            lines = [i for i in f.readlines()]
        else:
            lines = [i for i in f.readline().split(delimeter)]

        if strip:
            lines = [line.strip() for line in lines]

        lines = [scalar_type(symb) for symb in lines]
        return lines


def lcm(a, b):
    return a * b // math.gcd(a, b)
