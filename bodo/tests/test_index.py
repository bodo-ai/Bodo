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
    pd.UInt64Index([10, 12]),
    pd.Index(['A', 'B']),
])
def test_array_index_box(index):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(index), impl(index))


@pytest.fixture(params = [
    pd.date_range(start='2018-04-24', end='2018-04-27', periods=3),
    pd.date_range(start='2018-04-24', end='2018-04-27', periods=3, name='A'),
])
def dti_val(request):
    return request.param


def test_datetime_index_unbox(dti_val):
    def test_impl(dti):
        return dti

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(dti_val), test_impl(dti_val))


@pytest.mark.parametrize('field', bodo.hiframes.pd_timestamp_ext.date_fields)
def test_datetime_field(dti_val, field):
    func_text = 'def impl(A):\n'
    func_text += '  return A.{}\n'.format(field)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['impl']

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_date(dti_val):
    def impl(A):
      return A.date

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_min(dti_val):
    def impl(A):
      return A.min()

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_max(dti_val):
    def impl(A):
      return A.max()

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_sub(dti_val):
    t = dti_val.min()  # Timestamp object
    # DatetimeIndex - Timestamp
    def impl(A, t):
      return A - t

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(dti_val, t), impl(dti_val, t))

    # Timestamp - DatetimeIndex
    def impl2(A, t):
      return t - A

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_index_equal(bodo_func(dti_val, t), impl2(dti_val, t))


@pytest.mark.parametrize('comp', ['==', '!=', '>=', '>', '<=', '<'])
def test_datetime_str_comp(dti_val, comp):
    # string literal
    func_text = 'def impl(A):\n'
    func_text += '  return A {} "2015-01-23"\n'.format(comp)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['impl']

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))

    # string passed in
    func_text = 'def impl(A, s):\n'
    func_text += '  return A {} s\n'.format(comp)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['impl']

    bodo_func = bodo.jit(impl)
    s = '2015-01-23'
    np.testing.assert_array_equal(bodo_func(dti_val, s), impl(dti_val, s))


@pytest.mark.parametrize('data', [
    [100, 110],
    np.arange(10),
    np.arange(10).view(np.dtype('datetime64[ns]')),
    pd.Series(np.arange(10)),
    pd.Series(np.arange(10).view(np.dtype('datetime64[ns]'))),
    ['2015-8-3', '1990-11-21'],  # TODO: other time formats
    ['2015-8-3', 'NaT', '', '1990-11-21'],  # NaT cases
    pd.Series(['2015-8-3', '1990-11-21']),
    pd.DatetimeIndex(['2015-8-3', '1990-11-21']),
])
def test_datetime_index_constructor(data):
    def test_impl(d):
        return pd.DatetimeIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(data), test_impl(data))


@pytest.fixture(params = [
    pd.timedelta_range(start='1D', end='3D'),
    pd.timedelta_range(start='1D', end='3D', name='A'),
])
def timedelta_index_val(request):
    return request.param


def test_timedelta_index_unbox(timedelta_index_val):
    def test_impl(timedelta_index):
        return timedelta_index

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(
        bodo_func(timedelta_index_val), test_impl(timedelta_index_val))
