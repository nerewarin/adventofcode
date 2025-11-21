txt = open(r"G:\PycharmProjects\adventofcode\adventofcode2023\inputs\10\run")

pipeMatrix = []

for line in txt.readlines():
    line = line.replace("-", "═").replace("|", "║").replace("F", "╔").replace("J", "╝").replace("7", "╗").replace("L",
        "╚")
    row = [*line.strip()]
    row.insert(0, ".")
    row.append(".")
    pipeMatrix.append(row)

pipeMatrix.insert(0, ["."] * len(pipeMatrix[0]))
pipeMatrix.append(["."] * len(pipeMatrix[0]))


def printPipeMatrix():
    for row in range(1, len(pipeMatrix) - 1):
        r = ""
        for col in range(1, len(pipeMatrix[0]) - 1):
            char = pipeMatrix[row][col]
            r += "\033[31m×\033[0m" if char == "." else "\033[32m$\033[0m" if char == "S" else char
        print(r)


def connectsAt(pipe):
    dirs = []
    if pipe in ["║", "╚", "╝"]:
        dirs.append("bottom")
    if pipe in ["║", "╗", "╔"]:
        dirs.append("top")
    if pipe in ["═", "╚", "╔"]:
        dirs.append("left")
    if pipe in ["═", "╝", "╗"]:
        dirs.append("right")
    return dirs


def neighbors(coords):
    row = coords["row"]
    col = coords["col"]
    pipe = pipeMatrix[row][col]
    pipeConnects = ["top", "bottom", "left", "right"] if pipe == "S" else connectsAt(pipe)
    neighborList = []
    top = pipeMatrix[row - 1][col]
    bottom = pipeMatrix[row + 1][col]
    left = pipeMatrix[row][col - 1]
    right = pipeMatrix[row][col + 1]
    if "top" in connectsAt(top) and "bottom" in pipeConnects:
        neighborList.append({"row": row - 1, "col": col})
    if "bottom" in connectsAt(bottom) and "top" in pipeConnects:
        neighborList.append({"row": row + 1, "col": col})
    if "left" in connectsAt(left) and "right" in pipeConnects:
        neighborList.append({"row": row, "col": col - 1})
    if "right" in connectsAt(right) and "left" in pipeConnects:
        neighborList.append({"row": row, "col": col + 1})
    return neighborList


s = {}
mainLoop = []
for row in range(len(pipeMatrix)):
    r = []
    for col in range(len(pipeMatrix[0])):
        r.append(0)
        if pipeMatrix[row][col] == "S":
            r[col] = 1
            s.update({"row": row, "col": col})
    mainLoop.append(r)

prev = s
current = neighbors(s)[1]
mainLoop[current["row"]][current["col"]] = 1
running = False
while not (running and len(neighbors(current)) == 1):
    running = True
    nextNeighbors = neighbors(current)
    if nextNeighbors[0]["row"] == prev["row"] and nextNeighbors[0]["col"] == prev["col"]:
        prev = current
        current = nextNeighbors[1]
    else:
        prev = current
        current = nextNeighbors[0]
    mainLoop[current["row"]][current["col"]] = 1

for row in range(1, len(pipeMatrix) - 1):
    for col in range(1, len(pipeMatrix[0]) - 1):
        if mainLoop[row][col] == 0:
            pipeMatrix[row][col] = " "


def inside(row, col):
    cnt = 0
    prev = ""
    for c in range(col, len(pipeMatrix[row])):
        tile = pipeMatrix[row][c]
        if tile == "║":
            cnt += 1
        if (prev + tile == "╔╝") or (prev + tile == "╚╗"):
            cnt += 1

        if tile in "╔╝╚╗":
            prev = tile

    return cnt % 2


total = 0
for row in range(1, len(pipeMatrix) - 1):
    for col in range(1, len(pipeMatrix[0]) - 1):
        if pipeMatrix[row][col] == " ":  # not main loop
            ins = inside(row, col)
            if ins == 1:
                pipeMatrix[row][col] = "."
                total += 1

printPipeMatrix()
print(total)
