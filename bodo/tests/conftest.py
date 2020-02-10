# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import gc
import pytest
from numba.runtime import rtsys


# similar to Pandas
@pytest.fixture
def datapath():
    """Get the path to a test data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``bodo/tests/data``

    Returns
    -------
    path : path including ``bodo/tests/data``.

    Raises
    ------
    ValueError
        If the path doesn't exist.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            msg = "Could not find file {}."
            raise ValueError(msg.format(path))
        return path

    return deco


@pytest.fixture(scope="function")
def memory_leak_check():
    """
    A context manager fixture that makes sure there is no memory leak in the test.
    Equivalent to Numba's MemoryLeakMixin:
    https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/tests/support.py#L688
    """
    gc.collect()
    old = rtsys.get_allocation_stats()
    yield
    gc.collect()
    new = rtsys.get_allocation_stats()
    total_alloc = new.alloc - old.alloc
    total_free = new.free - old.free
    total_mi_alloc = new.mi_alloc - old.mi_alloc
    total_mi_free = new.mi_free - old.mi_free
    assert total_alloc == total_free
    assert total_mi_alloc == total_mi_free


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    Mark the first half of the tests with marker "firsthalf"
    """
    n = len(items)
    for item in items[0:n//2]:
        item.add_marker(pytest.mark.firsthalf)
