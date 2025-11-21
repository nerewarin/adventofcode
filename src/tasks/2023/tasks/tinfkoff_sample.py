"""
https://github.com/Tinkoff/career/blob/main/interview/sections/programming.md
"""
Input = [1, 2, 4, 5], [3, 3, 4], [2, 3, 4, 5, 6]

l1, l2, l3 = map(len, Input)

i = j = k = 0
while i < l1 and j < l2 and k < l3:
    x = Input[0][i]
    y = Input[1][j]
    z = Input[2][k]

    if y > x or z > x:
        i += 1
        continue

    if y < x:
        j += 1
        continue

    if z < x:
        k += 1
        continue

    print(x)
    break
