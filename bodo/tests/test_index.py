# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Tests for pd.Index functionality
"""
import datetime
import operator

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import AnalysisTestPipeline, check_func
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_range_index_constructor(memory_leak_check, is_slow_run):
    """
    Test pd.RangeIndex()
    """

    def impl1():  # single literal
        return pd.RangeIndex(10)

    check_func(impl1, (), only_seq=True)

    if not is_slow_run:
        return

    def impl2():  # two literals
        return pd.RangeIndex(3, 10)

    check_func(impl2, (), only_seq=True)

    def impl3():  # three literals
        return pd.RangeIndex(3, 10, 2)

    check_func(impl3, (), only_seq=True)

    def impl4(a):  # single arg
        return pd.RangeIndex(a, name="ABC")

    check_func(impl4, (5,), only_seq=True)

    def impl5(a, b):  # two args
        return pd.RangeIndex(a, b)

    check_func(impl5, (5, 10), only_seq=True)

    def impl6(a, b, c):  # three args
        return pd.RangeIndex(a, b, c)

    check_func(impl6, (5, 10, 2), only_seq=True)

    def impl7(r):  # unbox
        return r._start, r._stop, r._step

    r = pd.RangeIndex(3, 10, 2)
    check_func(impl7, (r,), only_seq=True)


@pytest.mark.slow
def test_distributed_range_index(memory_leak_check):
    """
    Tests returning a distributed range index.
    """

    def test_impl():
        return pd.RangeIndex(0, 1, 1)

    check_func(test_impl, ())


@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 3, 4]),  # Int array
        np.ones(3, dtype=np.int64),  # Int64Index: array of int64
        np.arange(3),  # Int64Ind: array input
        pd.date_range(
            start="2018-04-24", end="2018-04-27", periods=3
        ),  # datetime range
        pd.timedelta_range(start="1D", end="3D"),  # deltatime range
        pd.date_range(start="2018-04-10", end="2018-04-27", periods=3),
        pd.date_range(
            start="2018-04-10", end="2018-04-27", periods=3
        ).to_series(),  # deltatime series
        pd.Series(["hello world", "lsskasbdf", ""] * 3),
        pytest.param(
            pd.Series([b"sdalf", b"asd", b"mnbghu"] * 3),
            id="binary_case",
        ),
    ],
)
def test_generic_index_constructor(data):
    """
    Test the pd.Index with different inputs
    """

    def impl(data):
        return pd.Index(data)

    # parallel with no dtype
    check_func(impl, (data,))


@pytest.mark.slow
def test_binary_infer(memory_leak_check):
    """tests that we can infer the type of a binary index"""

    def impl(idx):
        return idx

    check_func(
        impl,
        (
            pd.Index(
                pd.array([b"ajkshdg", b"jhasdgf", b"asdfajd", np.NaN] * 3),
                name="my_index",
            ),
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "data,dtype",
    [
        (np.ones(3, dtype=np.int32), np.float64),
        (np.arange(10), np.dtype("datetime64[ns]")),
        (
            pd.Series(["2020-9-1", "2019-10-11", "2018-1-4", "2015-8-3", "1990-11-21"]),
            np.dtype("datetime64[ns]"),
        ),
        (np.arange(10), np.dtype("timedelta64[ns]")),
        (pd.Series(np.arange(10)), np.dtype("timedelta64[ns]")),
    ],
)
def test_generic_index_constructor_with_dtype(data, dtype):
    def impl(data, dtype):
        return pd.Index(data, dtype=dtype)

    check_func(impl, (data, dtype))


@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [
        [1, 3, 4],
        ["A", "B", "C"],
        pytest.param(
            [bytes(5), b"abc", b"vhkj"],
            id="binary_case",
        ),
    ],
)
def test_generic_index_constructor_sequential(data):
    def impl(data):
        return pd.Index(data)

    check_func(impl, (data,), dist_test=False)


@pytest.mark.slow
def test_numeric_index_constructor(memory_leak_check, is_slow_run):
    """
    Test pd.Int64Index/UInt64Index/Float64Index objects
    """

    def impl1():  # list input
        return pd.Int64Index([10, 12])

    check_func(impl1, (), only_seq=True)

    if not is_slow_run:
        return

    def impl2():  # list input with name
        return pd.Int64Index([10, 12], name="A")

    check_func(impl2, (), only_seq=True)

    def impl3():  # array input
        return pd.Int64Index(np.arange(3))

    check_func(impl3, (), only_seq=True)

    def impl4():  # array input different type
        return pd.Int64Index(np.ones(3, dtype=np.int32))

    check_func(impl4, (), only_seq=True)

    def impl5():  # uint64: list input
        return pd.UInt64Index([10, 12])

    check_func(impl5, (), only_seq=True)

    def impl6():  # uint64: array input different type
        return pd.UInt64Index(np.ones(3, dtype=np.int32))

    check_func(impl6, (), only_seq=True)

    def impl7():  # float64: list input
        return pd.Float64Index([10.1, 12.1])

    check_func(impl7, (), only_seq=True)

    def impl8():  # float64: array input different type
        return pd.Float64Index(np.ones(3, dtype=np.int32))

    check_func(impl8, (), only_seq=True)


def test_init_numeric_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_numeric_index() is applied correctly"""
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"]),
        pytest.param(pd.Index([b"hkjl", bytes(2), b""]), id="binary_case"),
    ],
)
def test_array_index_box(index, memory_leak_check):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(index), impl(index))


@pytest.mark.slow
@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"] * 4),
        pd.RangeIndex(10),
        # pd.RangeIndex(3, 10, 2), # TODO: support
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.timedelta_range(start="1D", end="3D", name="A"),
        pd.CategoricalIndex(["A", "B", "A", "C", "B"]),
        # TODO: PeriodIndex.values returns object array of Period objects
        # pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
        pytest.param(pd.Index([b"hkjl", bytes(2), b""] * 3), id="binary_case"),
    ],
)
def test_index_values(index, memory_leak_check):
    def impl(A):
        return A.values

    check_func(impl, (index,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12, 11, 1, 3, 4], name="A"),
        pd.Index(["A", "B", "C", "D", "FF"], name="B"),
        pytest.param(
            pd.Index([b"A", bytes(1), b"C", b"", b"ghbjhk"], name="cur_idx"),
            id="binary_case",
        ),
        pd.RangeIndex(10, name="BB"),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=6, name="A"),
    ],
)
def test_index_slice_name(index, memory_leak_check):
    """make sure Index name is preserved properly in slicing"""

    def impl(I):
        return I[:3]

    check_func(impl, (index,), only_seq=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "index, key",
    [
        (pd.RangeIndex(3, 22, 3), 9),
        (pd.Int64Index([10, 12, 15, 18]), 15),
        (pd.Index(["A", "B", "C", "AA", "DD"]), "A"),
        pytest.param(
            pd.Index([b"asdhgk", b"", bytes(7), b"AA", b"DD"]),
            b"asdhgk",
            id="binary_case",
        ),
        (
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=6),
            pd.Timestamp("2018-04-27"),
        ),
        (pd.timedelta_range(start="1D", end="3D"), pd.Timedelta("2D")),
    ],
)
def test_index_get_loc(index, key, memory_leak_check):
    """test Index.get_loc() for various Index types"""

    def impl(A, key):
        return A.get_loc(key)

    check_func(impl, (index, key), only_seq=True)


# TODO(ehsan): add memory_leak_check when Numba's Exception memory leaks are fixed
def test_index_get_loc_error_checking():
    """Test possible errors in Index.get_loc() such as non-unique Index which is not
    supported.
    """

    def impl(A, key):
        return A.get_loc(key)

    # String idx

    # repeated value raises an error
    index = pd.Index(["A", "B", "C", "AA", "DD", "C"])
    key = "C"
    with pytest.raises(
        ValueError, match=r"Index.get_loc\(\): non-unique Index not supported yet"
    ):
        bodo.jit(impl)(index, key)
    # key not in Index
    index = pd.Index(["A", "B", "C", "AA", "DD"])
    key = "E"
    with pytest.raises(KeyError, match=r"Index.get_loc\(\): key not found"):
        bodo.jit(impl)(index, key)

    # binary idx

    # repeated value raises an error
    index = pd.Index([b"A", b"B", b"C", b"AA", b"DD", b"C"])
    key = b"C"
    with pytest.raises(
        ValueError, match=r"Index.get_loc\(\): non-unique Index not supported yet"
    ):
        bodo.jit(impl)(index, key)
    # key not in Index
    index = pd.Index([b"A", b"B", b"C", b"AA", b"DD"])
    key = b"E"
    with pytest.raises(KeyError, match=r"Index.get_loc\(\): key not found"):
        bodo.jit(impl)(index, key)

    # key not in RangeIndex
    with pytest.raises(KeyError, match=r"Index.get_loc\(\): key not found"):
        bodo.jit(impl)(pd.RangeIndex(1, 4, 2), 2)
    with pytest.raises(KeyError, match=r"Index.get_loc\(\): key not found"):
        bodo.jit(impl)(pd.RangeIndex(1, 4, 2), 11)
    with pytest.raises(KeyError, match=r"Index.get_loc\(\): key not found"):
        bodo.jit(impl)(pd.RangeIndex(3, 11, 2), 2)


# Need to add the code and the check for the PeriodIndex
# pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12, 0, 2, 1, 3, -4]),
        pd.Float64Index([10.1, 12.1, 1.2, 3.1, -1.2, -3.1, 0.0]),
        pd.UInt64Index([10, 12, 0, 1, 11, 12, 5, 3]),
        pd.Index(["A", "B", "AB", "", "CDEF", "CC", "l"]),
        pytest.param(
            pd.Index([b"asdga", b"asdga", b"", b"oihjb", bytes(2), b"CC", b"asdfl"]),
            id="binary_case",
        ),
        pd.RangeIndex(11),
        # pd.RangeIndex(3, 10, 2), # TODO: support
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.PeriodIndex(
            year=[2015, 2015, 2016, 1026, 2018, 2018, 2019],
            month=[1, 2, 3, 1, 2, 3, 4],
            freq="M",
        ),
        pd.timedelta_range(start="1D", end="15D", name="A"),
    ],
)
def test_index_copy(index, memory_leak_check):
    def test_impl_copy(S):
        return S.copy()

    def test_impl_copy_name(S):
        return S.copy(name="kjahsdfbhjsd")

    check_func(test_impl_copy, (index,))
    check_func(test_impl_copy_name, (index,))


@pytest.fixture(
    params=[
        pd.date_range(start="2018-04-14", end="2018-04-24", periods=10),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-30", periods=10, name="A"),
            marks=pytest.mark.slow,
        ),
    ]
)
def dti_val(request):
    return request.param


@pytest.mark.slow
def test_datetime_index_unbox(dti_val, memory_leak_check):
    def test_impl(dti):
        return dti

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(dti_val), test_impl(dti_val))


@pytest.mark.parametrize("field", bodo.hiframes.pd_timestamp_ext.date_fields)
def test_datetime_field(dti_val, field, memory_leak_check):
    """tests datetime index.field. This should be inlined in series pass"""

    func_text = "def impl(A):\n"
    func_text += "  return A.{}\n".format(field)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    check_func(impl, (dti_val,))


def test_datetime_date(dti_val, memory_leak_check):
    """tests datetime index.field. This should be inlined in series pass"""

    def impl(A):
        return A.date

    check_func(impl, (dti_val,))


def test_datetime_min(dti_val, memory_leak_check):
    def impl(A):
        return A.min()

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_max(dti_val, memory_leak_check):
    def impl(A):
        return A.max()

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(bodo_func(dti_val), impl(dti_val))


def test_datetime_sub(dti_val, memory_leak_check):
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


def test_datetimeindex_constant_lowering(memory_leak_check):
    dti = pd.to_datetime(
        ["1/1/2018", np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)]
    )

    def impl():
        return dti

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


@pytest.mark.parametrize(
    "comparison_impl",
    [
        pytest.param(lambda a, b: a == b, id="eq"),
        pytest.param(lambda a, b: a != b, id="ne"),
        pytest.param(lambda a, b: a < b, id="lt"),
        pytest.param(lambda a, b: a <= b, id="le"),
        pytest.param(lambda a, b: a > b, id="gt"),
        pytest.param(lambda a, b: a >= b, id="ge"),
    ],
)
def test_datetime_compare(comparison_impl, dti_val, memory_leak_check):
    t = dti_val.min()  # Timestamp object

    check_func(comparison_impl, (dti_val, t))
    check_func(comparison_impl, (t, dti_val))


def test_string_index_constant_lowering(memory_leak_check):
    si = pd.Index(["A", "BB", "ABC", "", "KG", "FF", "ABCDF"])

    def impl():
        return si

    bodo_func = bodo.jit(distributed=False)(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_string_index_constant_get_loc(memory_leak_check):
    """make sure get_loc works for constant lowered Index since it doesn't have a dict"""
    si = pd.Index(["A", "BB", "ABC", "", "KG", "FF", "ABCDF"])

    def impl():
        return si.get_loc("ABC")

    check_func(impl, (), only_seq=True)


def test_string_index_constant_contains(memory_leak_check):
    """make sure contains works for constant lowered Index since it doesn't have a dict"""
    si = pd.Index(["A", "BB", "ABC", "", "KG", "FF", "ABCDF"])

    def impl():
        return "ABC" in si

    check_func(impl, (), only_seq=True)


def test_binary_index_constant_lowering(memory_leak_check):
    si = pd.Index([b"asfd", b"bjkhj", bytes(12), b"", b"KG", b"khs", b"asdfkh"])

    def impl():
        return si

    bodo_func = bodo.jit(distributed=False)(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_int64_index_constant_lowering(memory_leak_check):
    idx = pd.Int64Index([-1, 43, 54, 65, 123])

    def impl():
        return idx

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_uint64_index_constant_lowering(memory_leak_check):
    idx = pd.UInt64Index([1, 43, 54, 65, 123])

    def impl():
        return idx

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_float64_index_constant_lowering(memory_leak_check):
    idx = pd.Float64Index([1.2, 43.4, 54.7, 65, 123])

    def impl():
        return idx

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


@pytest.mark.smoke
def test_datetime_getitem(dti_val, memory_leak_check):
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
def test_datetime_str_comp(dti_val, comp, memory_leak_check):
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


@pytest.mark.slow
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
        np.array([datetime.date(2020, 1, 1) + datetime.timedelta(i) for i in range(7)]),
    ],
)
def test_datetime_index_constructor(data, memory_leak_check):
    def test_impl(d):
        return pd.DatetimeIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(data), test_impl(data))


def test_init_datetime_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_datetime_index() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(d):
        I = pd.DatetimeIndex(d)
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.date_range("2017-01-03", periods=10).values)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


def test_pd_date_range(memory_leak_check):
    """test pd.date_range() support with various argument combinations"""
    # start/end provided (default freq="D")
    def impl():
        return pd.date_range(start="2018-01-01", end="2018-01-08")

    check_func(impl, ())

    # start/periods provided (default freq="D")
    def impl2():
        return pd.date_range(start="2018-01-01", periods=8)

    check_func(impl2, ())

    # start/end/periods provided
    def impl3():
        return pd.date_range(start="2018-04-24", end="2018-04-27", periods=3)

    check_func(impl3, ())

    # start/end/freq provided
    def impl4():
        return pd.date_range(start="2018-04-24", end="2018-04-27", freq="2D")

    check_func(impl4, ())

    # end/periods/freq provided
    def impl5():
        return pd.date_range(end="2018-04-24", periods=20, freq="2D")

    check_func(impl5, ())


@pytest.fixture(
    params=[
        pd.timedelta_range(start="1D", end="3D"),
        pytest.param(
            pd.timedelta_range(start="1D", end="3D", name="A"), marks=pytest.mark.slow
        ),
    ]
)
def timedelta_index_val(request):
    return request.param


@pytest.mark.slow
def test_timedelta_index_unbox(timedelta_index_val, memory_leak_check):
    def test_impl(timedelta_index):
        return timedelta_index

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(
        bodo_func(timedelta_index_val), test_impl(timedelta_index_val)
    )


@pytest.fixture(
    params=[
        pytest.param([100, 110]),
        pytest.param(np.arange(10), marks=pytest.mark.slow),
        pytest.param(
            np.arange(10).view(np.dtype("timedelta64[ns]")),
        ),
        pytest.param(pd.Series(np.arange(10)), marks=pytest.mark.slow),
        pytest.param(
            pd.Series(np.arange(10).view(np.dtype("timedelta64[ns]"))),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.TimedeltaIndex(np.arange(10)), marks=pytest.mark.slow),
    ],
)
def tdi_data(request):
    return request.param


def test_timedelta_index_constructor(tdi_data, memory_leak_check):
    def test_impl(d):
        return pd.TimedeltaIndex(d)

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(tdi_data), test_impl(tdi_data))


def test_timedelta_index_min(tdi_data, memory_leak_check):
    def test_impl(d):
        return pd.TimedeltaIndex(d).min()

    check_func(test_impl, (tdi_data,))


def test_timedelta_index_max(tdi_data, memory_leak_check):
    def test_impl(d):
        return pd.TimedeltaIndex(d).max()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(tdi_data) == test_impl(tdi_data)


def test_timedelta_index_constant_lowering(memory_leak_check):
    tdi = pd.TimedeltaIndex(np.arange(10))

    def impl():
        return tdi

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_init_timedelta_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_timedelta_index() is applied correctly"""
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
def test_timedelta_field(timedelta_index_val, field, memory_leak_check):
    """tests timdelta index.field. This should be inlined in series pass"""

    func_text = "def impl(A):\n"
    func_text += "  return A.{}\n".format(field)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    check_func(impl, (timedelta_index_val,))


@pytest.mark.parametrize(
    "comparison_impl",
    [
        pytest.param(lambda a, b: a == b, id="eq"),
        pytest.param(lambda a, b: a != b, id="ne"),
        pytest.param(lambda a, b: a < b, id="lt"),
        pytest.param(lambda a, b: a <= b, id="le"),
        pytest.param(lambda a, b: a > b, id="gt"),
        pytest.param(lambda a, b: a >= b, id="ge"),
    ],
)
def test_timedelta_compare(comparison_impl, timedelta_index_val, memory_leak_check):
    t = timedelta_index_val.min()  # Timedelta object

    check_func(comparison_impl, (timedelta_index_val, t))
    check_func(comparison_impl, (t, timedelta_index_val))


@pytest.mark.parametrize(
    "categorical_index",
    [
        pd.CategoricalIndex(["A", "B", "A", "CC", "B"]),
        pd.CategoricalIndex([1, 4, 5, 1, 4, 2], ordered=True),
    ],
)
def test_categorical_index_box(categorical_index, memory_leak_check):
    def impl(A):
        return A

    check_func(impl, (categorical_index,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "interval_index",
    [
        pd.IntervalIndex.from_arrays(np.arange(11), np.arange(11) + 1),
    ],
)
def test_interval_index_box(interval_index, memory_leak_check):
    def impl(A):
        return A

    check_func(impl, (interval_index,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "period_index",
    [
        pd.PeriodIndex(year=[2015, 2016, 2018], quarter=[1, 2, 3]),
        pytest.param(
            pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_period_index_box(period_index, memory_leak_check):
    def impl(A):
        return A

    pd.testing.assert_index_equal(bodo.jit(impl)(period_index), impl(period_index))


def test_periodindex_constant_lowering(memory_leak_check):
    pi = pd.PeriodIndex(year=[2015, 2016, 2018], quarter=[1, 2, 3])

    def impl():
        return pi

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


@pytest.mark.slow
@pytest.mark.parametrize(
    "m_ind",
    [
        pytest.param(
            pd.MultiIndex.from_arrays([[3, 4, 1, 5, -3]]), marks=pytest.mark.slow
        ),
        pd.MultiIndex.from_arrays(
            [
                ["ABCD", "V", "CAD", "", "AA"],
                [1.3, 4.1, 3.1, -1.1, -3.2],
                pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            ]
        ),
        # repeated names
        pytest.param(
            pd.MultiIndex.from_arrays([[1, 5, 9], [2, 1, 8]], names=["A", "A"]),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_multi_index_unbox(m_ind, memory_leak_check):
    def test_impl(m_ind):
        return m_ind

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(m_ind), test_impl(m_ind))


def test_init_string_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_binary_str_index() is applied correctly for string indexes"""
    import numba.tests.test_array_analysis

    def impl(d):
        I = bodo.hiframes.pd_index_ext.init_binary_str_index(d, "AA")
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.array(["AA", "BB", "C"]))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


def test_init_binary_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_binary_str_index() is applied for binary indexes"""
    import numba.tests.test_array_analysis

    def impl(d):
        I = bodo.hiframes.pd_index_ext.init_binary_str_index(d, "AA")
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    # this is needed, else we get issues due to numpys handling of this array
    test_func(np.array([b"AA", b"asdgas", b"sdfkah"], dtype=object))

    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("d#0")


@pytest.mark.skip("Need support for charSeq [BE-1281]")
def test_init_binary_array_bad_numpy_arr(memory_leak_check):
    a = np.array([b"AA", b"asdgas", b"sdfkah"])

    @bodo.jit
    def impl(d):
        I = pd.Index(d)
        return I

    check_func(impl, (a,))


@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"] * 4),
        pytest.param(pd.Index([b"A", b"B"] * 4), id="binary_idx"),
        pd.RangeIndex(1, 15, 2),
        pd.RangeIndex(-4, 8, 4),
        pd.RangeIndex(-10, -1, -3),
        pd.RangeIndex(-4, 8, -4),
        # Unskip after [BE-497] is resolved
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-497] is resolved
        pytest.param(
            pd.timedelta_range(start="1D", end="3D", name="A"), marks=pytest.mark.skip
        ),
    ],
)
def test_index_iter(index, memory_leak_check):
    def test_impl1(index):
        return list(index)

    def test_impl2(index):
        lst = []
        for val in index:
            lst.append(val)
        return lst

    check_func(test_impl1, (index,), dist_test=False)
    check_func(test_impl2, (index,), dist_test=False)


def test_init_range_index_array_analysis(memory_leak_check):
    """make sure shape equivalence for init_range_index() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(n):
        I = bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)
        return I

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(11)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("I#0") == eq_set._get_ind("n")


def test_map_str(memory_leak_check):
    """test string output in map"""

    def test_impl(I):
        return I.map(lambda a: str(a))

    I = pd.Int64Index([1, 211, 333, 43, 51])
    check_func(test_impl, (I,))


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.RangeIndex(11), marks=pytest.mark.slow),
        pd.Int64Index([10, 12, 1, 3, 2, 4, 5, -1]),
        pytest.param(
            pd.Float64Index([10.1, 12.1, 1.1, 2.2, -1.2, 4.1, -2.1]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["A", "BB", "ABC", "", "KG", "FF", "ABCDF"]),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.date_range("2019-01-14", periods=11), marks=pytest.mark.slow),
        # TODO(ehsan): return CategoricalIndex for CategoricalIndex.map()
        # pytest.param(pd.CategoricalIndex(["A", "B"]*3), marks=pytest.mark.slow),
        # TODO: enable when pd.Timedelta is supported (including box_if_dt64)
        # pd.timedelta_range("3D", periods=11),
    ],
)
def test_map(index, memory_leak_check):
    """test Index.map for all Index types"""

    def test_impl(I):
        return I.map(lambda a: a)

    check_func(test_impl, (index,))


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([i for i in range(100)]),
        pytest.param(pd.UInt64Index([i for i in range(100)]), marks=pytest.mark.slow),
        pytest.param(pd.Float64Index([i for i in range(100)]), marks=pytest.mark.slow),
        pd.Index([i for i in range(100, -1, -1)]),
        pd.Index([1, 4, 3, 2, 5, 6, 4]),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=10),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-22", periods=100),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(
                [
                    np.datetime64("2018-01-02"),
                    np.datetime64("2018-01-03"),
                    np.datetime64("2018-01-01"),
                    np.datetime64("2018-01-06"),
                    np.datetime64("2018-01-04"),
                    np.datetime64("2018-01-05"),
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.timedelta_range(start="1D", end="15D", freq="1D"), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.timedelta_range(start="3Y", end="1Y", periods=100),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([pd.Timedelta(1), pd.Timedelta(3), pd.Timedelta(2)]),
            marks=pytest.mark.slow,
        ),
        pd.RangeIndex(1, 100, 2),
        pytest.param(pd.RangeIndex(1, 100, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(1, 1, 1), marks=pytest.mark.slow),
    ],
)
def test_monotonicity(index, memory_leak_check):
    """
    Test is_monotonic, is_monotonic_increasing, and is_monotonic_decreasing attributes for indices
    of type Int64Index, UInt64Index, Float64Index, DatetimeIndex, TimedeltaIndex, and RangeIndex.
    """

    def f1(index):
        return index.is_monotonic

    def f2(index):
        return index.is_monotonic_increasing

    def f3(index):
        return index.is_monotonic_decreasing

    check_func(f1, (index,))
    check_func(f2, (index,))
    check_func(f3, (index,))


@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 3, 4]),  # Int array
        np.ones(3, dtype=np.int64),  # Int64Index: array of int64
        pd.date_range(
            start="2018-04-24", end="2018-04-27", periods=3
        ),  # datetime range
        pd.timedelta_range(start="1D", end="3D"),  # deltatime range
        pytest.param(
            np.array([b"adksg", b"abdsgas", b"asdga", b"lkjhi"]),
            id="binary_case",
        ),
    ],
)
def test_index_unsupported(data):
    """Test that a Bodo error is raised for unsupported
    Index methods
    """

    def test_all(idx):
        return idx.all()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_all)(idx=pd.Index(data))

    def test_any(idx):
        return idx.any()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_any)(idx=pd.Index(data))

    def test_append(idx):
        return idx.append()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_append)(idx=pd.Index(data))

    def test_argmax(idx):
        return idx.argmax()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_argmax)(idx=pd.Index(data))

    def test_argmin(idx):
        return idx.argmin()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_argmin)(idx=pd.Index(data))

    def test_argsort(idx):
        return idx.argsort()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_argsort)(idx=pd.Index(data))

    def test_asof(idx):
        return idx.asof()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_asof)(idx=pd.Index(data))

    def test_asof_locs(idx):
        return idx.asof_locs()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_asof_locs)(idx=pd.Index(data))

    def test_astype(idx):
        return idx.astype()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_astype)(idx=pd.Index(data))

    def test_delete(idx):
        return idx.delete()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_delete)(idx=pd.Index(data))

    def test_difference(idx):
        return idx.difference()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_difference)(idx=pd.Index(data))

    def test_drop(idx):
        return idx.drop()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_drop)(idx=pd.Index(data))

    def test_droplevel(idx):
        return idx.droplevel()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_droplevel)(idx=pd.Index(data))

    def test_dropna(idx):
        return idx.dropna()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_dropna)(idx=pd.Index(data))

    def test_equals(idx):
        return idx.equals()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_equals)(idx=pd.Index(data))

    def test_factorize(idx):
        return idx.factorize()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_factorize)(idx=pd.Index(data))

    def test_fillna(idx):
        return idx.fillna()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_fillna)(idx=pd.Index(data))

    def test_format(idx):
        return idx.format()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_format)(idx=pd.Index(data))

    def test_get_indexer(idx):
        return idx.get_indexer()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_indexer)(idx=pd.Index(data))

    def test_get_indexer_for(idx):
        return idx.get_indexer_for()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_indexer_for)(idx=pd.Index(data))

    def test_get_indexer_non_unique(idx):
        return idx.get_indexer_non_unique()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_indexer_non_unique)(idx=pd.Index(data))

    def test_get_level_values(idx):
        return idx.get_level_values()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_level_values)(idx=pd.Index(data))

    def test_get_slice_bound(idx):
        return idx.get_slice_bound()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_slice_bound)(idx=pd.Index(data))

    def test_get_value(idx):
        return idx.get_value()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_get_value)(idx=pd.Index(data))

    def test_groupby(idx):
        return idx.groupby()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_groupby)(idx=pd.Index(data))

    def test_holds_integer(idx):
        return idx.holds_integer()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_holds_integer)(idx=pd.Index(data))

    def test_identical(idx):
        return idx.identical()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_identical)(idx=pd.Index(data))

    def test_insert(idx):
        return idx.insert()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_insert)(idx=pd.Index(data))

    def test_intersection(idx):
        return idx.intersection()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_intersection)(idx=pd.Index(data))

    def test_is_(idx):
        return idx.is_()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_)(idx=pd.Index(data))

    def test_is_boolean(idx):
        return idx.is_boolean()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_boolean)(idx=pd.Index(data))

    def test_is_categorical(idx):
        return idx.is_categorical()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_categorical)(idx=pd.Index(data))

    def test_is_floating(idx):
        return idx.is_floating()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_floating)(idx=pd.Index(data))

    def test_is_integer(idx):
        return idx.is_integer()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_integer)(idx=pd.Index(data))

    def test_is_interval(idx):
        return idx.is_interval()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_interval)(idx=pd.Index(data))

    def test_is_mixed(idx):
        return idx.is_mixed()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_mixed)(idx=pd.Index(data))

    def test_is_numeric(idx):
        return idx.is_numeric()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_numeric)(idx=pd.Index(data))

    def test_is_object(idx):
        return idx.is_object()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_object)(idx=pd.Index(data))

    def test_is_type_compatible(idx):
        return idx.is_type_compatible()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_type_compatible)(idx=pd.Index(data))

    def test_isin(idx):
        return idx.isin()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_isin)(idx=pd.Index(data))

    def test_item(idx):
        return idx.item()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_item)(idx=pd.Index(data))

    def test_join(idx):
        return idx.join()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_join)(idx=pd.Index(data))

    def test_memory_usage(idx):
        return idx.memory_usage()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_memory_usage)(idx=pd.Index(data))

    def test_nunique(idx):
        return idx.nunique()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_nunique)(idx=pd.Index(data))

    def test_putmask(idx):
        return idx.putmask()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_putmask)(idx=pd.Index(data))

    def test_ravel(idx):
        return idx.ravel()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_ravel)(idx=pd.Index(data))

    def test_reindex(idx):
        return idx.reindex()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_reindex)(idx=pd.Index(data))

    def test_repeat(idx):
        return idx.repeat()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_repeat)(idx=pd.Index(data))

    def test_searchsorted(idx):
        return idx.searchsorted()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_searchsorted)(idx=pd.Index(data))

    def test_set_names(idx):
        return idx.set_names()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_set_names)(idx=pd.Index(data))

    def test_set_value(idx):
        return idx.set_value()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_set_value)(idx=pd.Index(data))

    def test_shift(idx):
        return idx.shift()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_shift)(idx=pd.Index(data))

    def test_slice_indexer(idx):
        return idx.slice_indexer()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_slice_indexer)(idx=pd.Index(data))

    def test_slice_locs(idx):
        return idx.slice_locs()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_slice_locs)(idx=pd.Index(data))

    def test_sort(idx):
        return idx.sort()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_sort)(idx=pd.Index(data))

    def test_sort_values(idx):
        return idx.sort_values()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_sort_values)(idx=pd.Index(data))

    def test_sortlevel(idx):
        return idx.sortlevel()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_sortlevel)(idx=pd.Index(data))

    def test_str(idx):
        return idx.str()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_str)(idx=pd.Index(data))

    def test_symmetric_difference(idx):
        return idx.symmetric_difference()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_symmetric_difference)(idx=pd.Index(data))

    def test_to_flat_index(idx):
        return idx.to_flat_index()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_flat_index)(idx=pd.Index(data))

    def test_to_frame(idx):
        return idx.to_frame()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_frame)(idx=pd.Index(data))

    def test_to_list(idx):
        return idx.to_list()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_list)(idx=pd.Index(data))

    def test_to_native_types(idx):
        return idx.to_native_types()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_native_types)(idx=pd.Index(data))

    def test_to_numpy(idx):
        return idx.to_numpy()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_numpy)(idx=pd.Index(data))

    def test_to_series(idx):
        return idx.to_series()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_series)(idx=pd.Index(data))

    def test_tolist(idx):
        return idx.tolist()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_tolist)(idx=pd.Index(data))

    def test_transpose(idx):
        return idx.transpose()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_transpose)(idx=pd.Index(data))

    def test_union(idx):
        return idx.union()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_union)(idx=pd.Index(data))

    def test_unique(idx):
        return idx.unique()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_unique)(idx=pd.Index(data))

    def test_value_counts(idx):
        return idx.value_counts()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_value_counts)(idx=pd.Index(data))

    def test_view(idx):
        return idx.view()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_view)(idx=pd.Index(data))

    def test_where(idx):
        return idx.where()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_where)(idx=pd.Index(data))


def test_heter_index_binop():
    """test binary operations on heterogeneous Index values"""
    # TODO(ehsan): fix Numba bugs for passing list literal to pd.Index
    def impl1():
        A = pd.Index(("A", 2))
        return A == 2

    def impl2():
        A = pd.Index(("A", 2))
        return A == pd.Index(("A", 3))

    def impl3():
        A = pd.Index(("A", 2))
        return "A" == A

    check_func(impl1, (), dist_test=False)
    check_func(impl2, (), dist_test=False)
    check_func(impl3, (), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
def test_index_cmp_ops(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Index([4, 6, 7, 1])
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, 2))
    check_func(test_impl, (2, S))
    S = pd.RangeIndex(12)
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, 2))
    check_func(test_impl, (2, S))


@pytest.mark.parametrize(
    "index",
    [
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.timedelta_range(start="1D", end="3D", name="A"),
        pd.Index(["A", "BB", "ABC", "", "KG", "FF", "ABCDF"]),
        pytest.param(
            pd.Index([b"sdhfa", b"asdf", bytes(5), b"", b"A", b"abjfdsb", b"ABCDF"]),
            id="binary_case",
        ),
        pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
        pd.RangeIndex(10),
    ],
)
def test_index_nbytes(index, memory_leak_check):
    """Test nbytes computation with different Index types"""

    def impl(idx):
        return idx.nbytes

    # RangeIndexType has three int64 values, so the total is 24 bytes on
    # each rank.
    if isinstance(index, pd.RangeIndex):
        py_out = 24
        check_func(impl, (index,), py_output=py_out, dist_test=False)
        check_func(impl, (index,), py_output=py_out * bodo.get_size(), only_1D=True)
    # String/BinaryIndex has three 3 underlying arrays (data, offsets, null_bit_map)

    # In the String example: 15 characters,
    # data=15 characters * 1 byte +
    # offset=(7 elements + 1 extra) * 8 bytes +
    # null_bit_map= 1 byte
    # Total = 80 bytes for sequential case
    # + 9 (For each rank we get an extra 8 bytes for offset
    # and 1 byte for null_bit_map)
    elif isinstance(index[0], str):
        py_out = 80 + (bodo.get_size() - 1) * 9
        check_func(impl, (index,), py_output=py_out, only_1D=True)

    # In the Binary example:
    # data = 27 bytes
    # offset=(7 elements + 1 extra) * 8 bytes +
    # null_bit_map= 1 byte
    # Total = 92 bytes for sequential case
    # + 9 (For each rank we get an extra 8 bytes for offset
    elif isinstance(index[0], bytes):
        py_out = 92 + (bodo.get_size() - 1) * 9
        check_func(impl, (index,), py_output=py_out, only_1D=True)

    # PeriodIndex example:
    # data = 24 bytes
    # null_bit_map= 1 byte per rank
    # Total = 24 + num_ranks bytes
    elif isinstance(index, pd.PeriodIndex):
        py_out = 24 + bodo.get_size()
        check_func(impl, (index,), py_output=py_out, only_1D=True)
    else:
        check_func(impl, (index,))


def test_boolean_index(memory_leak_check):
    def impl1(arr):
        idx = pd.Index(arr)
        return idx

    arr = pd.array([True, False] * 10)
    check_func(impl1, (arr,))

    n_arr = pd.array([True, False, None, True, None] * 3)
    check_func(impl1, (n_arr,))

    def impl2(idx, data):
        return pd.Series(data, index=idx)

    idx = pd.Index(arr)
    data = np.arange(20)
    check_func(
        impl2,
        (
            data,
            idx,
        ),
    )


def test_datetime_max_all_na():
    """tests index.max() properly returns NA for all NA datetime index"""
    idx = pd.DatetimeIndex([None, None, None] * 5)

    # Series return is to avoid scalar NaT type boxing issue, see BE-860
    def impl(I):
        return pd.Series([I.max()])

    check_func(impl, (idx,))


def test_datetime_min_all_na():
    """tests index.min() properly returns NA for all NA datetime index"""
    idx = pd.DatetimeIndex([None, None, None] * 5)

    # Series return is to avoid scalar NaT type boxing issue, see BE-860
    def impl(I):
        return pd.Series([I.min()])

    check_func(impl, (idx,))


def test_timedelta_min_all_na():
    """tests index.max() properly returns NA for all NA timedelta index"""
    idx = pd.TimedeltaIndex([None, None, None] * 5)

    def impl(I):
        return I.min()

    check_func(impl, (idx,))


def test_timedelta_max_all_na():
    """tests index.min() properly returns NA for all NA timedelta index"""
    idx = pd.TimedeltaIndex([None, None, None] * 5)

    def impl(I):
        return I.max()

    check_func(impl, (idx,))


@pytest.mark.parametrize(
    "idx,new_name",
    [
        pytest.param(pd.RangeIndex(start=0, stop=5), "a", id="RangeIndexType"),
        pytest.param(pd.Index([1, 2, 3, 4, 5]), "a", id="NumericIndexType"),
        pytest.param(pd.Index(["a", "b", "c", "d", "e"]), "a", id="StringIndexType"),
        pytest.param(
            pd.Index([b"a", b"b", b"c", b"d", b"e"]), "a", id="BinaryIndexType"
        ),
        pytest.param(
            pd.PeriodIndex(
                year=[2000, 2001, 2002, 2003, 2004], quarter=[1, 2, 3, 2, 1]
            ),
            "a",
            id="PeriodIndexType",
        ),
        pytest.param(
            pd.DatetimeIndex(
                ["2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"]
            ),
            "a",
            id="DatetimeIndexType",
        ),
        pytest.param(
            pd.TimedeltaIndex(["1 days", "2 days", "3 days", "4 days", "5 days"]),
            "a",
            id="TimedeltaIndexType",
        ),
        pytest.param(pd.interval_range(start=0, end=5), "a", id="IntervalIndexType"),
        pytest.param(
            pd.CategoricalIndex(["a", "a", "b", "c", "d"]),
            "a",
            id="CategoricalIndexType",
        ),
    ],
)
def test_index_rename(idx, new_name):
    """tests index.rename() for all types of supported indexes"""

    def impl(I, name):
        return I.rename(name)

    check_func(impl, (idx, new_name))


def test_index_rename_dist_bug(memory_leak_check):
    """tests index.rename() for distribution match between input and output [BE-2285]"""

    @bodo.jit(distributed=["I"], returns_maybe_distributed=False)
    def f(I):
        return I.rename("B")

    I = pd.Index(["A", "B", "C", "D"], name="A")
    with pytest.raises(BodoError, match=r"has distributed flag in function"):
        f(I)


def test_index_rename_heter():
    """tests the special case for heterogeneous indices"""
    df = pd.DataFrame(
        {"key": ["bar", "bat", "baz", "fii", "foo"], "value": [1, 2, 3, 4, 5]}
    )

    def impl(df):
        def row_index_renamer(row):
            return row.index.rename("a").name

        return df.apply(row_index_renamer, axis=1)

    check_func(impl, (df,))


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(pd.RangeIndex(start=0, stop=5), id="RangeIndexType"),
        pytest.param(
            pd.Index([1, 2, None, 4, 5]), id="NumericIndexType", marks=pytest.mark.slow
        ),
        pytest.param(
            pd.Index(["a", "b", None, "d", "e"] * 2),
            id="StringIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([b"a", b"b", None, b"d", b"e"] * 2),
            id="BinaryIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["a", "a", None, "c", "d"]),
            id="CategoricalIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.PeriodIndex(
                [
                    pd.Period("2001-01", freq="M"),
                    pd.Period("2002-01", freq="M"),
                    None,
                    pd.Period("2004-01", freq="M"),
                    pd.Period("2005-01", freq="M"),
                ],
            ),
            id="PeriodIndexType",
        ),
        pytest.param(
            pd.DatetimeIndex(
                ["2000-01-01", "2001-01-01", None, "2003-01-01", "2004-01-01"]
            ),
            id="DatetimeIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.TimedeltaIndex(["1 days", "2 days", None, "4 days", "5 days"]),
            id="TimedeltaIndexType",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(lambda I: I.isna(), id="isna"),
        pytest.param(lambda I: I.notna(), id="notna"),
        pytest.param(lambda I: I.isnull(), id="isnull"),
        pytest.param(lambda I: I.notnull(), id="notnull"),
    ],
)
def test_index_check_na(idx, impl):
    check_func(impl, (idx,))


def test_period_index_lower_check_na():
    idx = pd.PeriodIndex(
        [
            pd.Period("2001-01", freq="M"),
            pd.Period("2002-01", freq="M"),
            None,
            pd.Period("2004-01", freq="M"),
            pd.Period("2005-01", freq="M"),
        ],
    )

    def impl():
        return idx.isna()

    check_func(impl, (), dist_test=False)


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(pd.RangeIndex(start=0, stop=5), id="RangeIndexType"),
        pytest.param(
            pd.Index([5, 1, 1, 2, 2, 3, 4] * 2),
            id="NumericIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["e", "a", "a", "b", "b", "c", "d"] * 2),
            id="StringIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([b"e", b"a", b"a", b"b", b"b", b"c", b"d"] * 2),
            id="BinaryIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["e", "a", "a", "b", "b", "c", "d"] * 2),
            id="CategoricalIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.PeriodIndex(
                [
                    pd.Period("2001-01", freq="M"),
                    pd.Period("2001-01", freq="M"),
                    pd.Period("2002-01", freq="M"),
                    pd.Period("2002-01", freq="M"),
                    pd.Period("2003-01", freq="M"),
                    pd.Period("2004-01", freq="M"),
                    pd.Period("2005-01", freq="M"),
                ]
                * 2,
            ),
            id="PeriodIndexType",
        ),
        pytest.param(
            pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2000-01-01",
                    "2001-01-01",
                    "2001-01-01",
                    "2002-01-01",
                    "2003-01-01",
                    "2004-01-01",
                ]
                * 2
            ),
            id="DatetimeIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.TimedeltaIndex(
                ["4 days", "1 days", "1 days", "2 days", "2 days", "3 days", "5 days"]
            ),
            id="TimedeltaIndexType",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_drop_duplicates(idx):
    """tests index.drop_duplicates() for all types of supported indexes"""

    def impl(I):
        return I.drop_duplicates()

    check_func(impl, (idx,), sort_output=True)


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(pd.RangeIndex(start=0, stop=5), id="RangeIndexType"),
        pytest.param(
            pd.Index([5, 1, 1, 2, 2, 3, 4] * 2),
            id="NumericIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["e", "a", "a", "b", "b", "c", "d"] * 2),
            id="StringIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([b"e", b"a", b"a", b"b", b"b", b"c", b"d"] * 2),
            id="BinaryIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["e", "a", "a", "b", "b", "c", "d"] * 2),
            id="CategoricalIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.PeriodIndex(
                [
                    pd.Period("2001-01", freq="M"),
                    pd.Period("2001-01", freq="M"),
                    pd.Period("2002-01", freq="M"),
                    pd.Period("2002-01", freq="M"),
                    pd.Period("2003-01", freq="M"),
                    pd.Period("2004-01", freq="M"),
                    pd.Period("2005-01", freq="M"),
                ]
                * 2,
            ),
            id="PeriodIndexType",
        ),
        pytest.param(
            pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2000-01-01",
                    "2001-01-01",
                    "2001-01-01",
                    "2002-01-01",
                    "2003-01-01",
                    "2004-01-01",
                ]
                * 2
            ),
            id="DatetimeIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.TimedeltaIndex(
                ["4 days", "1 days", "1 days", "2 days", "2 days", "3 days", "5 days"]
            ),
            id="TimedeltaIndexType",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([1, 2, np.nan, 1, 4, np.nan, 5, 2, 6]),
            id="NumericNAIndex",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["a", "b", None, "a", "d", None, "e", "b", "f"] * 10),
            id="StringNAIndex",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index([b"e", b"a", pd.NA, b"b", b"b", pd.NA, b"d"] * 2),
            id="BinaryNAIndex",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_duplicated(idx, memory_leak_check):
    """
    Tests Index.duplicated() for all types of supported indices
    """

    def test_impl(I):
        return I.duplicated()

    check_func(test_impl, (idx,))
