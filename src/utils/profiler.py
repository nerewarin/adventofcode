import time
from functools import wraps

from src.utils.logger import get_logger

_logger = get_logger()


def timeit_deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        _logger.info(f"{func.__name__} took {end - start:.6f}s")
        return result

    return wrapper
