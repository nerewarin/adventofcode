import inspect
import pathlib
import re

from src.utils.logger import get_logger
from src.utils.profiler import timeit_deco

_TASK_NUM_FROM_FILE_REGEXP = re.compile(r"(?:day)?(\d+)")

_logger = get_logger()


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


def test(fn, expected, *args, test_part=None, test_data=None, **kwargs):
    """Checks the output of applying function to test data matches expected result"""
    func_name = fn.__name__
    _base_msg = "{func_name}{extra} in 'test' mode"
    extra_params = []

    if args:
        extra_params.append(f"{args=}")
    if kwargs:
        for k, v in kwargs.items():
            extra_params.append(f"{k}={v}")

    if extra_params:
        extra = " (" + ", ".join(extra_params) + ")"
    else:
        extra = ""
    base_msg = _base_msg.format(func_name=func_name, extra=extra)

    if test_data is None:
        root = _get_resources_dir()
        fname = "tst"
        if test_part and test_part > 1:
            fname += str(test_part)
            extra_params.append(f"{test_part=}")
        test_data = _file_to_list(root / fname)

    _logger.info(f"Running {base_msg}")
    res = fn(test_data, *args, **kwargs)

    result_msg = base_msg + " {result}"
    if res != expected:
        result = f"returned wrong result: {res=} != {expected=}!"
        msg = result_msg.format(result=result)
        raise ValueError(msg)
    else:
        result = "passed"
        msg = result_msg.format(result=result)
        _logger.warning(msg)


def run(fn, *args, **kwargs):
    """Prints the output of applying function to task data to console"""
    data = _file_to_list(_get_resources_dir() / "run")
    _logger.info(f"Running {fn.__name__} in 'run' mode")

    res = timeit_deco(fn)(data, *args, **kwargs)
    _logger.warning(f"'run' {fn.__name__} result: {res}")
    return res
