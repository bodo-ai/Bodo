# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Tests for pd.Index functionality
"""
import pandas as pd
import numpy as np
import bodo
import pytest
from bodo.tests.utils import check_func, AnalysisTestPipeline


def test_range_index_constructor(memory_leak_check):
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
        return pd.RangeIndex(a, name="ABC")

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


def test_numeric_index_constructor(memory_leak_check):
    """
    Test pd.Int64Index/UInt64Index/Float64Index objects
    """

    def impl1():  # list input
        return pd.Int64Index([10, 12])

    pd.testing.assert_index_equal(bodo.jit(impl1)(), impl1())

    def impl2():  # list input with name
        return pd.Int64Index([10, 12], name="A")

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


def test_init_numeric_index_array_analysis():
    """make sure shape equivalence for init_numeric_index() is applied correctly
    """
    import numba.tests.test_array_analysis

    def impl(d):
        I = pd.Int64Index(d)
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(np.arange(10))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"]),
    ],
)
def test_array_index_box(index, memory_leak_check):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(index), impl(index))


@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"]),
        pd.RangeIndex(10),
        # pd.RangeIndex(3, 10, 2), # TODO: support
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.timedelta_range(start="1D", end="3D", name="A"),
        # TODO: PeriodIndex.values returns object array of Period objects
        # pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
    ],
)
def test_index_values(index):
    def impl(A):
        return A.values

    check_func(impl, (index,))


# Need to add the code and the check for the PeriodIndex
# pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"]),
        pd.RangeIndex(10),
        # pd.RangeIndex(3, 10, 2), # TODO: support
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.timedelta_range(start="1D", end="3D", name="A"),
    ],
)
def test_index_copy(index):
    def test_impl_copy(S):
        return S.copy()

    check_func(test_impl_copy, (index,))


@pytest.fixture(
    params=[
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
    ]
)
def dti_val(request):
    return request.param


def test_datetime_index_unbox(dti_val, memory_leak_check):
    def test_impl(dti):
        return dti

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(dti_val), test_impl(dti_val))


@pytest.mark.parametrize("field", bodo.hiframes.pd_timestamp_ext.date_fields)
def test_datetime_field(dti_val, field):
    func_text = "def impl(A):\n"
    func_text += "  return A.{}\n".format(field)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

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


def test_datetime_getitem(dti_val):
    # constant integer index
    def impl(A):
        return A[0]

    bodo_func = bodo.jit(impl)
    assert bodo_func(dti_val) == impl(dti_val)

    # non-constant integer index
    def impl2(A, i):
        return A[i]

    i = 0
    bodo_func = bodo.jit(impl2)
    assert bodo_func(dti_val, i) == impl2(dti_val, i)

    # constant slice
    def impl3(A):
        return A[:1]

    bodo_func = bodo.jit(impl3)
    pd.testing.assert_index_equal(bodo_func(dti_val), impl3(dti_val))

    # non-constant slice
    def impl4(A, i):
        return A[:i]

    i = 1
    bodo_func = bodo.jit(impl4)
    pd.testing.assert_index_equal(bodo_func(dti_val, i), impl4(dti_val, i))


@pytest.mark.parametrize("comp", ["==", "!=", ">=", ">", "<=", "<"])
def test_datetime_str_comp(dti_val, comp):
    # string literal
    func_text = "def impl(A):\n"
    func_text += '  return A {} "2015-01-23"\n'.format(comp)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))

    # string passed in
    func_text = "def impl(A, s):\n"
    func_text += "  return A {} s\n".format(comp)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    bodo_func = bodo.jit(impl)
    s = "2015-01-23"
    np.testing.assert_array_equal(bodo_func(dti_val, s), impl(dti_val, s))


@pytest.mark.parametrize(
    "data",
    [
        [100, 110],
        np.arange(10),
        np.arange(10).view(np.dtype("datetime64[ns]")),
        pd.Series(np.arange(10)),
        pd.Series(np.arange(10).view(np.dtype("datetime64[ns]"))),
        ["2015-8-3", "1990-11-21"],  # TODO: other time formats
        ["2015-8-3", "NaT", "", "1990-11-21"],  # NaT cases
        pd.Series(["2015-8-3", "1990-11-21"]),
        pd.DatetimeIndex(["2015-8-3", "1990-11-21"]),
    ],
)
def test_datetime_index_constructor(data):
    def test_impl(d):
        return pd.DatetimeIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(data), test_impl(data))


def test_init_datetime_index_array_analysis():
    """make sure shape equivalence for init_datetime_index() is applied correctly
    """
    import numba.tests.test_array_analysis

    def impl(n):
        d = pd.date_range("2017-01-03", periods=n)
        I = pd.DatetimeIndex(d)
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(10)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


def test_pd_date_range():
    def impl():
        return pd.date_range(start="2018-01-01", end="2018-01-08")

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())

    def impl2():
        return pd.date_range(start="2018-01-01", periods=8)

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_index_equal(bodo_func(), impl2())

    def impl3():
        return pd.date_range(start="2018-04-24", end="2018-04-27", periods=3)

    bodo_func = bodo.jit(impl3)
    pd.testing.assert_index_equal(bodo_func(), impl3())


@pytest.fixture(
    params=[
        pd.timedelta_range(start="1D", end="3D"),
        pd.timedelta_range(start="1D", end="3D", name="A"),
    ]
)
def timedelta_index_val(request):
    return request.param


def test_timedelta_index_unbox(timedelta_index_val, memory_leak_check):
    def test_impl(timedelta_index):
        return timedelta_index

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(
        bodo_func(timedelta_index_val), test_impl(timedelta_index_val)
    )


@pytest.mark.parametrize(
    "data",
    [
        [100, 110],
        np.arange(10),
        np.arange(10).view(np.dtype("timedelta64[ns]")),
        pd.Series(np.arange(10)),
        pd.Series(np.arange(10).view(np.dtype("timedelta64[ns]"))),
        pd.TimedeltaIndex(np.arange(10)),
    ],
)
def test_timedelta_index_constructor(data):
    def test_impl(d):
        return pd.TimedeltaIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(data), test_impl(data))


def test_init_timedelta_index_array_analysis():
    """make sure shape equivalence for init_timedelta_index() is applied correctly
    """
    import numba.tests.test_array_analysis

    def impl(d):
        I = pd.TimedeltaIndex(d)
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.TimedeltaIndex(np.arange(10)))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


@pytest.mark.parametrize("field", bodo.hiframes.pd_timestamp_ext.timedelta_fields)
def test_timedelta_field(timedelta_index_val, field):
    func_text = "def impl(A):\n"
    func_text += "  return A.{}\n".format(field)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(
        bodo_func(timedelta_index_val), impl(timedelta_index_val)
    )


@pytest.mark.parametrize(
    "period_index",
    [
        pd.PeriodIndex(year=[2015, 2016, 2018], quarter=[1, 2, 3]),
        pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
    ],
)
def test_period_index_box(period_index, memory_leak_check):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(period_index), impl(period_index))


@pytest.mark.parametrize(
    "m_ind",
    [
        pd.MultiIndex.from_arrays([[3, 4, 1, 5, -3]]),
        pd.MultiIndex.from_arrays(
            [
                ["ABCD", "V", "CAD", "", "AA"],
                [1.3, 4.1, 3.1, -1.1, -3.2],
                pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            ]
        ),
    ],
)
def test_multi_index_unbox(m_ind, memory_leak_check):
    def test_impl(m_ind):
        return m_ind

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(m_ind), test_impl(m_ind))
