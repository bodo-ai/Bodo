"""
Tests for numba internals that require patching. These mostly consist
of tests ported from Numba and possibly modified to test inside Bodo.

Most of these tests are designed to avoid the numba pipeline and instead just
test the desired helper function.
"""

import os

import numba  # noqa TID253
import pytest
from numba.core import types  # noqa TID253


# Used for testing topological sort ordering. Copied from
# test_dataflow.py inside Numba.
def for_break(n, x):
    for i in range(n):
        n = 0
        if i == x:
            break
    else:
        n = i
    return i, n


@pytest.mark.parametrize(
    "iterative",
    [
        False,
        True,
    ],
)
def test_for_break(iterative):
    """
    Check the ordering for breaking out of a for loop. This implicitly
    checks that visited
    """
    old_use_iterative = os.environ.get("BODO_TOPO_ORDER_ITERATIVE", None)
    try:
        os.environ["BODO_TOPO_ORDER_ITERATIVE"] = "1" if iterative else "0"
        pyfunc = for_break
        cfunc = numba.jit((types.intp, types.intp))(pyfunc)
        for n, x in [(4, 2), (4, 6)]:
            assert pyfunc(n, x) == cfunc(n, x)
    finally:
        if old_use_iterative is None:
            del os.environ["BODO_TOPO_ORDER_ITERATIVE"]
        else:
            os.environ["BODO_TOPO_ORDER_ITERATIVE"] = old_use_iterative
