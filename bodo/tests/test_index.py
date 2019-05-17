"""
Tests for pd.Index functionality
"""
import pandas as pd
import bodo


def test_range_index_constructor():
    """
    Test pd.RangeIndex()
    """
    def impl1():  # single literal
        return pd.RangeIndex(10)

    pd.testing.assert_index_equal(bodo.jit(impl1)(), impl1())

    def impl2():  # two literals
        return pd.RangeIndex(3, 10)

    pd.testing.assert_index_equal(bodo.jit(impl2)(), impl2())

    def impl3():  # three literals
        return pd.RangeIndex(3, 10, 2)

    pd.testing.assert_index_equal(bodo.jit(impl3)(), impl3())

    def impl4(a):  # single arg
        return pd.RangeIndex(a)

    pd.testing.assert_index_equal(bodo.jit(impl4)(5), impl4(5))

    def impl5(a, b):  # two args
        return pd.RangeIndex(a, b)

    pd.testing.assert_index_equal(bodo.jit(impl5)(5, 10), impl5(5, 10))

    def impl6(a, b, c):  # three args
        return pd.RangeIndex(a, b, c)

    pd.testing.assert_index_equal(bodo.jit(impl6)(5, 10, 2), impl6(5, 10, 2))
