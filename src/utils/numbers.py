def get_sign(number):
    if number < 0:
        return -1
    return 1


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v
