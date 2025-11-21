import pathlib
import inspect
import re

_TASK_NUM_FROM_FILE_REGEXP = re.compile("(?:day)?(\d+)")
_INPUTS_ROOT = pathlib.Path("./../inputs").resolve()


def _get_resources_dir():
    for frame in inspect.stack():
        fname = pathlib.Path(frame.filename).name
        if not isinstance(fname, str):
            continue

        res = _TASK_NUM_FROM_FILE_REGEXP.match(fname)
        if res:
            task_num = res.group(1)
            return _INPUTS_ROOT / task_num

    raise ValueError("Could not find filename from stack matching regexp")


def _file_to_list(fname):
    lst = []
    with fname.open() as f:
        for raw in f.readlines():
            lst.append(raw.replace("\n", ""))
    return lst


def test(fn, expected, *args, **kwargs):
    """Checks the output of applying function to test data matches expected result"""
    root = _get_resources_dir()

    fname = "tst"
    test_part = kwargs.get("test_part")
    if test_part and test_part > 1:
        fname += str(test_part)

    test_data = _file_to_list(root / fname)

    res = fn(test_data, *args, **kwargs)

    if res != expected:
        raise ValueError(f"fn {fn} returned wrong result: {res=} != {expected=}!")
    print("test ok")

def run(fn, *args, **kwargs):
    """Prints the output of applying function to task data to console"""
    data = _file_to_list(_get_resources_dir() / "run")
    res = fn(data, *args, **kwargs)
    print(res)
    return res
