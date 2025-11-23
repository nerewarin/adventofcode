import inspect
import pathlib
import re

_TASK_NUM_FROM_FILE_REGEXP = re.compile(r"(?:day)?(\d+)")


def _get_resources_dir():
    for frame in inspect.stack():
        fname = pathlib.Path(frame.filename)
        if not fname.name:
            continue

        res = _TASK_NUM_FROM_FILE_REGEXP.match(fname.name)
        if res:
            task_num = res.group(1)
            # Find the year folder (2019, adventofcode2022, etc.) from the file path
            # File is in src/tasks/{year}/tasks/dayX.py, inputs are in src/tasks/{year}/inputs/
            parts = fname.parts
            try:
                tasks_idx = parts.index("tasks")
                parts[tasks_idx + 1]
                # Go up from src/tasks/{year}/tasks/dayX.py to src/tasks/{year}/inputs/
                year_path = fname.parent.parent
                return year_path / "inputs" / task_num
            except (ValueError, IndexError):
                # Fallback: try relative path from file location
                return fname.parent.parent.parent / "inputs" / task_num

    raise ValueError("Could not find filename from stack matching regexp")


def _file_to_list(fname):
    lst = []
    with fname.open() as f:
        for raw in f.readlines():
            lst.append(raw.replace("\n", ""))
    return lst


def test(fn, expected, *args, test_part=None, **kwargs):
    """Checks the output of applying function to test data matches expected result"""
    root = _get_resources_dir()

    fname = "tst"
    if test_part and test_part > 1:
        fname += str(test_part)
        success_msg = f"test {fn.__name__} ({test_part=}) passed"
    else:
        success_msg = f"test {fn.__name__} passed"

    test_data = _file_to_list(root / fname)

    res = fn(test_data, *args, **kwargs)

    if res != expected:
        raise ValueError(f"fn {fn} returned wrong result: {res=} != {expected=}!")

    print(success_msg)


def run(fn, *args, **kwargs):
    """Prints the output of applying function to task data to console"""
    data = _file_to_list(_get_resources_dir() / "run")
    res = fn(data, *args, **kwargs)
    print(f"run {fn.__name__} result: {res}")
    return res
