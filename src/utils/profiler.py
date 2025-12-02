from functools import wraps
from timeit import Timer


def timeit_deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = Timer(lambda: func(*args, **kwargs))
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {t.timeit(number=1):.6f}s")
        return result

    return wrapper
