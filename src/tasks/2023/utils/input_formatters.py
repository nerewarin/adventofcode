def group_by_blank(lines: list[str]) -> list[list[str]]:
    """Groups list of strings to 2d list by blank line as separator
    """
    res = []
    group = []
    for line in lines:
        if not line:
            res.append(group)
            group = []
        else:
            group.append(line)
    else:
        res.append(group)

    return res


def cast_2d_list_elements(lst, type_=int):
    """Modifies 2d-list casting all their elements to a given type_"""
    m = [[] for _ in lst]
    for row, line in enumerate(lst):
        for size in line:
            m[row].append(type_(size))
    return m


def lines_to_tuples(lines: list[str]) -> list[tuple[str]]:
    return [
        tuple(line.split()) for line in lines
    ]
