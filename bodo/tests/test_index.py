"""
Tests for pd.Index functionality
"""
import pandas as pd
import numpy as np
import bodo
import pytest


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

    def impl7(r):  # unbox
        return r._start, r._stop, r._step

    r = pd.RangeIndex(3, 10, 2)
    assert bodo.jit(impl7)(r) == impl7(r)


def test_numeric_index_constructor():
    """
    Test pd.Int64Index/UInt64Index/Float64Index objects
    """
    def impl1():  # list input
        return pd.Int64Index([10, 12])

    pd.testing.assert_index_equal(bodo.jit(impl1)(), impl1())

    def impl2():  # list input with name
        return pd.Int64Index([10, 12], name='A')

    pd.testing.assert_index_equal(bodo.jit(impl2)(), impl2())

    def impl3():  # array input
        return pd.Int64Index(np.arange(3))

    pd.testing.assert_index_equal(bodo.jit(impl3)(), impl3())

    def impl4():  # array input different type
        return pd.Int64Index(np.ones(3, dtype=np.int32))

    pd.testing.assert_index_equal(bodo.jit(impl4)(), impl4())

    def impl5():  # uint64: list input
        return pd.UInt64Index([10, 12])

    pd.testing.assert_index_equal(bodo.jit(impl5)(), impl5())

    def impl6():  # uint64: array input different type
        return pd.UInt64Index(np.ones(3, dtype=np.int32))

    pd.testing.assert_index_equal(bodo.jit(impl6)(), impl6())

    def impl7():  # float64: list input
        return pd.Float64Index([10.1, 12.1])

    pd.testing.assert_index_equal(bodo.jit(impl7)(), impl7())

    def impl8():  # float64: array input different type
        return pd.Float64Index(np.ones(3, dtype=np.int32))

    pd.testing.assert_index_equal(bodo.jit(impl8)(), impl8())


@pytest.mark.parametrize('index', [
    pd.Int64Index([10, 12]),
    pd.Float64Index([10.1, 12.1]),
    pd.UInt64Index([10, 12])
])
def test_numeric_index_box(index):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(index), impl(index))


@pytest.mark.parametrize('dti_val', [
    pd.date_range(start='2018-04-24', end='2018-04-27', periods=3),
    pd.date_range(start='2018-04-24', end='2018-04-27', periods=3, name='A'),
])
def test_datetime_index_unbox(dti_val):
    def test_impl(dti):
        return dti

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(dti_val), test_impl(dti_val))


@pytest.mark.parametrize('data', [
    [100, 110],
    np.arange(10),
    np.arange(10).view(np.dtype('datetime64[ns]')),
    pd.Series(np.arange(10)),
    pd.Series(np.arange(10).view(np.dtype('datetime64[ns]'))),
    ['2015-8-3', '1990-11-21'],  # TODO: other time formats
    pd.Series(['2015-8-3', '1990-11-21']),
    pd.DatetimeIndex(['2015-8-3', '1990-11-21']),
])
def test_datetime_index_constructor(data):
    def test_impl(d):
        return pd.DatetimeIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(data), test_impl(data))
