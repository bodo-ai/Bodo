# Copyright (C) 2022 Bodo Inc. All rights reserved.
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


@pytest.mark.parametrize(
    "arr",
    [
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ]
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="UInt8",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="UInt16",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="UInt32",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="UInt64",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="Int8",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="Int16",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="Int32",
        ),
        pd.array(
            [
                0,
                1,
                3,
                None,
                2,
                7,
                13,
                None,
                12,
                None,
                11,
                22,
                None,
                None,
                43,
                None,
                42,
            ],
            dtype="Int64",
        ),
        pd.array([True, True, True, True, False, False, None, True]),
    ],
)
def test_numeric_null_constructor(arr):
    def impl(arr):
        return pd.Index(arr)

    check_func(impl, (arr,), dist_test=False)


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
        pd.Index([10, 12], dtype="Int64"),
        pd.Index([10.1, 12.1], dtype="float64"),
        pd.Index([10, 12], dtype="UInt64"),
        pd.Index(["A", "B"]),
        pytest.param(pd.Index([b"hkjl", bytes(2), b""]), id="binary_case"),
    ],
)
def test_array_index_box(index, memory_leak_check):
    def impl(A):
        return A

    bodo_out = bodo.jit(impl)(index)
    # convert ArrowStringArray to regular Numpy
    if bodo_out.dtype == pd.StringDtype("pyarrow"):
        bodo_out = pd.Index(bodo_out.to_numpy())
    pd.testing.assert_index_equal(bodo_out, index)


@pytest.mark.slow
@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 12], dtype="Int64"),
        pd.Index([10.1, 12.1], dtype="float64"),
        pd.Index([10, 12], dtype="UInt64"),
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


@pytest.mark.parametrize(
    "args",
    [
        (pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 5, 1]), pd.Index([1, 5, 2, 1, 3])),
        pytest.param(
            (pd.Index([1, 2, 3, 4, 5]), pd.Index([6, 7, 8, 9, 10])),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), pd.Index([6, 7, 8, 9, 10])),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index([1, 2, 3, 4, 5]), pd.Index([1, 2, 3, 4, 5])),
            marks=pytest.mark.slow,
        ),
        (pd.Index([1.0, 1.5, 1.2, 1.7, 2.9, 1.2]), pd.Index([1.0, 1.1, 1.2, 1.3])),
        (pd.Index(list("ABCDEFGHIJKLMNOPQRSTUVWX")), pd.Index(list("AEIOUY"))),  # FAILS
        (
            pd.date_range("2018-01-01", "2018-12-01", freq="M"),
            pd.date_range("2018-06-01", "2019-06-01", freq="M"),
        ),
        (
            pd.TimedeltaIndex(
                ["1 days", "2 days", "3 days", "4 days", "5 days", "6 days"]
            ),
            pd.TimedeltaIndex(
                [
                    "6 seconds",
                    "6 minutes",
                    "6 hours",
                    "6 days",
                    "6 milliseconds",
                    "6 nanoseconds",
                ]
            ),
        ),
        (
            pd.Index([b"a", b"l", b"p", b"h", b"a"]),
            pd.Index([b"p", b"l", b"e", b"a", b"sconesAreDelicious", b"e"]),
        ),
        (pd.RangeIndex(0, 10, 1), pd.RangeIndex(0, 5, 1)),
        (pd.RangeIndex(0, 10, 1), pd.RangeIndex(0, 30, 5)),
        pytest.param(
            (pd.RangeIndex(100, -1, -25), pd.RangeIndex(-10, 30, 5)),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.RangeIndex(0, 100, 15), pd.Index([2, 4, 8, 16, 64, 15])),
            marks=pytest.mark.slow,
        ),
        (pd.Index([1, 2, 3, 4, 5]), pd.Series([2, 4, 6, 8, 10])),
        (pd.Index([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])),
        pytest.param(
            (pd.Index(list(range(-5, 100000))), pd.Index(list(range(100005)))),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_set_operations(args):
    def impl1(I, J):
        return I.union(J)

    def impl2(I, J):
        return I.intersection(J)

    def impl3(I, J):
        return I.difference(J)

    def impl4(I, J):
        return I.symmetric_difference(J)

    I, J = args
    # Descending RangeIndex distributed set methods not supported yet [BE-2944]
    # BinaryIndex distributed set methods not supported yet [BE-3005]
    dist_test = not (
        (isinstance(I, pd.RangeIndex) and (I.step < 0))
        or (isinstance(J, pd.RangeIndex) and (J.step < 0))
        or isinstance(I[0], bytes)
        or isinstance(J[0], bytes)
    )

    # Bodo diverges from the Pandas API for union by returning in a possibly different
    # order and always removing duplicates.
    check_func(
        impl1,
        (I, J),
        py_output=I.append(pd.Index(J)).unique(),
        sort_output=True,
        dist_test=dist_test,
    )

    # Bodo diverges from the Pandas API for intersection by returning in a possibly
    # different order, defaulting sort to None, and always converting RangeIndex
    # to NumericIndex.
    check_func(
        impl2,
        (I, J),
        py_output=I.intersection(J, sort=None),
        sort_output=True,
        dist_test=dist_test,
        only_1D=True,
    )

    # Bodo diverges from the Pandas API for difference and symmetric_difference
    # by returning in a possibly different order, and always converting RangeIndex
    # to NumericIndex.
    check_func(impl3, (I, J), sort_output=True, dist_test=dist_test)
    check_func(impl3, (pd.Index(J), I), sort_output=True, dist_test=dist_test)
    check_func(impl4, (I, J), sort_output=True, dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Index([True, False, True, True, False]),
        pytest.param(pd.Index(pd.array([6, 7, 8, 9, 10])), marks=pytest.mark.slow),
        pytest.param(pd.Index(pd.array([6, 7, None, 9, None])), marks=pytest.mark.slow),
        pd.Index(["A", "B", "C", "D", "E"]),
        pd.Index([b"a", b"e", b"i", b"o", b"u"]),
        pd.RangeIndex(0, 100, 15),
        pd.date_range("2018-01-01", "2018-01-10"),
        pytest.param(pd.timedelta_range("1D", "7D"), marks=pytest.mark.slow),
        pd.CategoricalIndex(list("abcaacab")),
        pytest.param(
            pd.CategoricalIndex([1, 2, 3, 1, 1, 2, 3, 1]), marks=pytest.mark.slow
        ),
        pd.interval_range(0, 5),
        pd.period_range("2018", "2019", freq="M"),
        pd.MultiIndex.from_product([["A", "B", "C"], [1, 2, 3]]),
    ],
)
def test_index_is_methods(index):
    def impl(I):
        return (
            I.is_numeric(),
            I.is_integer(),
            I.is_floating(),
            I.is_boolean(),
            I.is_categorical(),
            I.is_interval(),
            I.is_object(),
        )

    check_func(impl, (index,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 12, 11, 1, 3, 4], dtype="Int64", name="A"),
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


@pytest.mark.parametrize(
    "args",
    [
        (pd.Index([1, 2, 3, 4, 5]), pd.RangeIndex(5, 0, -1)),
        (
            pd.Index([0.1, -0.3, 0.1, 0.7, -0.3], name="buzz"),
            pd.Index([b"A", b"L", b"P", b"H", b"A"]),
        ),
        (pd.Index([1, 5, 4, 2, 1, 5], name="bar"), pd.Index(list("aeiouy"))),
        (pd.date_range("2018-01-01", "2018-12-01", periods=6), pd.RangeIndex(0, 6, 1)),
        (
            pd.TimedeltaIndex(["1 days", "7 days", "3 days", "4 hours", "2 days"]),
            pd.Index(list("98765")),
        ),
        (pd.CategoricalIndex(list("abcaacab")), pd.RangeIndex(8, 65, 8)),
        (
            pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3], name="fizz"),
            pd.CategoricalIndex(list("aaeaaeieaa"), name="vowels"),
        ),
    ],
)
def test_index_to_series(args):
    def impl1(index):
        return index.to_series()

    def impl2(index):
        return index.to_series(name="foo")

    def impl3(index, other):
        return index.to_series(index=other)

    def impl4(index, other):
        return index.to_series(name=4, index=other)

    index, other = args

    # Descending RangeIndex distributed to_series not supported yet [BE-2944]
    dist_test_A = isinstance(index, pd.RangeIndex) and index.step < 0
    dist_test_B = isinstance(index, pd.RangeIndex) and index.step < 0

    check_func(impl1, (index,), dist_test=dist_test_A)
    check_func(impl2, (index,), dist_test=dist_test_A)
    check_func(impl3, (index, other), dist_test=dist_test_A)
    check_func(impl4, (index, other), dist_test=dist_test_A)
    if not isinstance(other, pd.MultiIndex):
        check_func(impl3, (other, index), dist_test=dist_test_B)

    # Passing in other iterables for index not supported for Timedelta or Binary.
    # Lists/tuples not supported for distributed tests.
    if not isinstance(other[0], (bytes, pd._libs.tslibs.timedeltas.Timedelta)):
        check_func(impl3, (index, list(other)), dist_test=False)
        check_func(impl3, (index, pd.Series(other)), dist_test=dist_test_A)
        check_func(impl3, (index, other.to_numpy()), dist_test=dist_test_A)
    if not isinstance(index[0], (bytes, pd._libs.tslibs.timedeltas.Timedelta)):
        check_func(impl3, (other, tuple(index)), dist_test=False)
        if not isinstance(other, pd.MultiIndex):
            check_func(impl4, (other, pd.Series(index)), dist_test=dist_test_B)
            check_func(impl4, (other, index.to_numpy()), dist_test=dist_test_B)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([1, 5, 4, 2, 1, 5], name="bar"),
        pd.Index([0.1, -0.3, 0.1, 0.7, -0.3], name="buzz"),
        pd.Index(list("aeiouy")),
        pd.Index([b"A", b"L", b"P", b"H", b"A"]),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(5, 0, -1),
        pd.RangeIndex(15, 100, 15, name="R"),
        pd.date_range("2018-01-01", "2018-12-01", periods=6),
        pd.TimedeltaIndex(["1 days", "7 days", "3 days", "4 hours", "2 days"]),
        pd.CategoricalIndex(list("abcaacab")),
        pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3], name="fizz"),
        pd.CategoricalIndex(list("aaeaaeieaa"), name="vowels"),
        # TODO: support cases where column names of the original MultiIndex
        # have heterogeneous types (i.e. name = [42, "Z"])
        pd.MultiIndex.from_product([[1, 2, 3], ["A", "B"]]),
        pd.MultiIndex.from_product([[1, 2, 3], ["A", "B"]], names=["x", "y"]),
        pd.MultiIndex.from_arrays(
            [[1, 5, 2, 1, 0], [1, 5, 2, 1, 3], ["A", "A", "B", "A", "B"]],
            names=[6, 7, 8],
        ),
    ],
)
def test_index_to_frame(index):
    def impl1(index):
        return index.to_frame()

    def impl2(index):
        return index.to_frame(index=False)

    def impl3(index):
        return index.to_frame(name="foo")

    def impl4(index):
        return index.to_frame(name=4, index=False)

    def impl5(index):
        return index.to_frame(name=["X", "Y"])

    def impl6(index):
        return index.to_frame(name=[64, 42], index=False)

    def impl7(index):
        return index.to_frame(name=("q", 9, "s"))

    def impl8(index):
        return index.to_frame(name=[7, "r", 13], index=False)

    dist_test = not (isinstance(index, pd.RangeIndex) and index.step < 0)
    check_func(impl1, (index,), dist_test=dist_test)
    check_func(impl2, (index,), dist_test=dist_test)
    if not isinstance(index, pd.MultiIndex):
        check_func(impl3, (index,), dist_test=dist_test)
        check_func(impl4, (index,), dist_test=dist_test)
    else:
        if len(index.levels) == 2:
            check_func(impl5, (index,), dist_test=dist_test)
            check_func(impl6, (index,), dist_test=dist_test)
        elif len(index.levels) == 3:
            check_func(impl7, (index,), dist_test=dist_test)
            check_func(impl8, (index,), dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([0.1, -0.3, 0.1, 0.7, -0.3]),
        pd.Index([True, False, False, False, False, True]),
        pd.Index(pd.array([1, 2, 3, None, 5, None, None])),
        pd.Index(list("aeiouy")),
        pd.Index([b"A", b"L", b"P", b"H", b"A"]),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(5, 0, -1),
        pd.RangeIndex(15, 100, 15),
        pd.date_range("2018-01-01", "2018-12-01", periods=6),
        pd.TimedeltaIndex(["1 days", "7 days", "3 days", "4 hours", "2 days"]),
        pd.CategoricalIndex(list("abcaacab")),
        pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3]),
        pd.CategoricalIndex(list("aaeaaeieaa")),
        pd.interval_range(0, 5),
    ],
)
def test_index_to_numpy(index):
    def impl1(index):
        return index.to_numpy()

    # Copy = True guarantees no alias. Copy = False is not tested because
    # it does not guarantee anything.
    def impl2(index):
        A = index.to_numpy(copy=True)
        B = index.to_numpy(copy=True)
        B[0] = B[1]
        return [A[0], A[1]]

    # RangeIndex distributed to_numpy not supported yet [BE-2944]
    dist_test = not isinstance(index, pd.RangeIndex)
    check_func(impl1, (index,), dist_test=dist_test)

    # IntervalArray getitem not supported yet (Related to [BE-2814])
    # np.ndarray of strings has issues with setitem (see [BE-2981])
    if not isinstance(index, pd.IntervalIndex) and not isinstance(index[0], str):
        check_func(impl2, (index,), dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([0.1, -0.3, 0.1, 0.7, -0.3]),
        pd.Index([True, False, False, False, False, True]),
        pd.Index(list("aeiouy")),
        pd.Index([b"A", b"L", b"P", b"H", b"A"]),
        pd.RangeIndex(0, 6, 1),
        pd.RangeIndex(5, 0, -1),
        pd.RangeIndex(15, 100, 15),
        pd.date_range("2018-01-01", "2018-12-01", periods=6),
        pd.TimedeltaIndex(["1 days", "7 days", "3 days", "4 hours", "2 days"]),
        pd.CategoricalIndex(list("abcaacab")),
        pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3]),
        pd.CategoricalIndex(list("aaeaaeieaa")),
    ],
)
def test_index_to_list(index):
    def impl(index):
        return index.to_list()

    # RangeIndex distributed to_list not supported yet [BE-2944]
    dist_test = not isinstance(index, pd.RangeIndex)
    check_func(impl, (index,), dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([13, -21, 0, 1, 1, 34, -2, 3, 55, 5, 88, -8], name="F"),
        pd.Index(
            pd.array(
                [
                    0,
                    1,
                    3,
                    None,
                    2,
                    7,
                    13,
                    None,
                    12,
                    None,
                    11,
                    22,
                    None,
                    23,
                    9,
                    24,
                    8,
                    None,
                    43,
                    None,
                    42,
                ]
            )
        ),
        pd.Index(
            [
                0.005780190596061163,
                0.23944886595871595,
                0.3463438965210339,
                0.4604841251734305,
                0.6825334098773845,
            ]
        ),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.Index([True, True, True, True, False, False, True]),
            marks=pytest.mark.skip,
        ),
        pd.Index(["a", "A", "alpha", "AAA", "a", "aaa", "alphabet", ""]),
        pd.Index(
            [
                b"c",
                b"cookie",
                b"CCC",
                b"cookies",
                b"COOKIE",
                b"cookie",
                b"CoOkIe",
                b"ccc",
            ]
        ),
        pd.RangeIndex(0, 10, 1, name="R"),
        pd.RangeIndex(0, 10, 3),
        pytest.param(pd.RangeIndex(0, 10, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, 1), marks=pytest.mark.slow),
        pd.RangeIndex(10, 0, 3),
        pytest.param(pd.RangeIndex(10, 0, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, -1), marks=pytest.mark.slow),
        pd.RangeIndex(0, 10, -3),
        pytest.param(pd.RangeIndex(0, 10, -30), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -30), marks=pytest.mark.slow),
        pd.TimedeltaIndex(
            [
                "2 days",
                "2 hours",
                "2 minutes",
                "2 seconds",
                "10 minutes",
                "10 days",
                "10 minutes",
                "5 seconds",
            ]
        ),
        pd.date_range(start="2018-01-10", end="2019-01-10", periods=13),
        pd.DatetimeIndex(
            pd.array(
                [
                    pd.Timestamp("2004-03-14"),
                    pd.Timestamp("2007-02-27"),
                    pd.Timestamp("2004-03-03"),
                    pd.Timestamp("2007-01-27"),
                    pd.Timestamp("2006-01-27"),
                ]
            )
        ),
        pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3, 1, 5, 2, 5, 1]),
        pd.CategoricalIndex(pd.array([None, 15, 14, 14, 14, 11, 12, 10, 15, None])),
        pytest.param(
            pd.CategoricalIndex(
                [0.5, 0.18, 0.5, 0.18, 0.07, 0.46, 0.46, 0.5, 0.72, 0.5]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["A", "B", "c", "a", "a", "C", "A", "B"]),
            marks=pytest.mark.slow,
        ),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.CategoricalIndex([True, False, False, False, False]),
            marks=pytest.mark.skip,
        ),
    ],
)
def test_index_sort_values(index):
    def impl1(index):
        return index.sort_values(ascending=True, na_position="first")

    def impl2(index):
        return index.sort_values(ascending=False, na_position="first")

    def impl3(index):
        return index.sort_values(ascending=True, na_position="last")

    def impl4(index):
        return index.sort_values(ascending=False, na_position="last")

    # Verifying that sorting does not alter the original index
    def impl5(index):
        index2 = index.sort_values()
        return list(index + index2)

    # RangeIndex distributed sort_values not supported yet [BE-2944] & [BE-3008]
    dist_test = not (isinstance(index, pd.RangeIndex))
    check_func(impl1, (index,), dist_test=dist_test)
    check_func(impl2, (index,), dist_test=dist_test)
    # If the Index is numerical, test alternative placement of nulls and
    # verify nondestructiveness
    if isinstance(index, pd.core.indexes.numeric.Int64Index):
        check_func(impl3, (index,), dist_test=dist_test)
        check_func(impl4, (index,), dist_test=dist_test)
        check_func(impl5, (index,), dist_test=False)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([13, -21, 0, 1, 1, 34, -2, 3, 55, 5, 88, -8]),
        pd.Index(
            pd.array(
                [
                    0,
                    1,
                    3,
                    None,
                    2,
                    7,
                    13,
                    None,
                    12,
                    None,
                    11,
                    22,
                    None,
                    23,
                    9,
                    24,
                    8,
                    None,
                    43,
                    None,
                    42,
                ]
            )
        ),
        pd.Index(
            [
                0.005780190596061163,
                0.23944886595871595,
                0.3463438965210339,
                0.4604841251734305,
                0.6825334098773845,
            ]
        ),
        pd.Index([True, True, True, True, False, False, True]),
        pd.Index(["a", "A", "alpha", "AAA", "a", "aaa", "alphabet", ""]),
        pd.Index(
            [
                b"c",
                b"cookie",
                b"CCC",
                b"cookies",
                b"COOKIE",
                b"cookie",
                b"CoOkIe",
                b"ccc",
            ]
        ),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(0, 10, 3),
        pytest.param(pd.RangeIndex(0, 10, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, 1), marks=pytest.mark.slow),
        pd.RangeIndex(10, 0, 3),
        pytest.param(pd.RangeIndex(10, 0, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, -1), marks=pytest.mark.slow),
        pd.RangeIndex(0, 10, -3),
        pytest.param(pd.RangeIndex(0, 10, -30), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -30), marks=pytest.mark.slow),
        pd.TimedeltaIndex(
            [
                "2 days",
                "2 hours",
                "2 minutes",
                "2 seconds",
                "10 minutes",
                "10 days",
                "10 minutes",
                "5 seconds",
            ]
        ),
        pd.date_range(start="2018-01-10", end="2019-01-10", periods=13),
        pd.DatetimeIndex(
            pd.array(
                [
                    pd.Timestamp("2004-03-15"),
                    pd.Timestamp("2004-03-14"),
                    pd.Timestamp("2007-02-27"),
                    pd.Timestamp("2004-03-03"),
                    pd.Timestamp("2007-01-27"),
                ]
            )
        ),
        pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3, 1, 5, 2, 5, 1]),
        pytest.param(
            pd.CategoricalIndex(pd.array([None, 15, 14, 14, 14, 11, 12, 10, 15, None])),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(
                [0.5, 0.18, 0.5, 0.18, 0.07, 0.46, 0.46, 0.5, 0.72, 0.5]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["A", "B", "c", "a", "a", "C", "A", "B"]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex([True, False, False, False, False]),
            marks=pytest.mark.slow,
        ),
        pd.PeriodIndex(
            year=[2000, 2002, 2001, 2001, 2001, 2004], quarter=[1, 1, 1, 3, 2, 1]
        ),
        pytest.param(
            pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_argsort(index):
    def impl(index):
        return index.argsort()

    # RangeIndex distributed argsort not supported yet [BE-2944]
    dist_test = not isinstance(index, pd.RangeIndex)
    check_func(impl, (index,), dist_test=dist_test)


@pytest.mark.slow
@pytest.mark.parametrize(
    "index, key",
    [
        (pd.RangeIndex(3, 22, 3), 9),
        (pd.Index([10, 12, 15, 18], dtype="Int64"), 15),
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


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 5, 2, 1, 0, 3, 1, 2, 5, 1]),
        pd.Index(pd.array([1, None, None, 5, None, 2, 1, 0, 3, 1, 2, None, 1, None])),
        pd.Index(
            [
                0.7573456092417888,
                0.21589999967248008,
                0.8671567646182514,
                0.383775019426454,
                0.21589999967248008,
                0.5937443415123859,
                0.5583538962837552,
                0.5583538962837552,
                0.6448399071221529,
                0.383775019426454,
            ]
        ),
        pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1], ordered=True),
        pytest.param(
            pd.CategoricalIndex([True, True, False, True, False], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex([1.0, 1.1, 1.2, 1.8, 1.5], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["A", "B", "C", "A", "A", "C", "A", "B"], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(
                [b"a", b"", bytes(4), b"a", b"A", b"Z", b"j", b"a"], ordered=True
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(
                [
                    pd.Timestamp("2018-03-01"),
                    pd.Timestamp("2018-06-01"),
                    pd.Timestamp("2018-03-01"),
                    pd.Timestamp("2018-01-01"),
                    pd.Timestamp("2018-03-01"),
                ],
                ordered=True,
            ),
            marks=pytest.mark.slow,
        ),
        pd.RangeIndex(start=0, stop=10, step=1),
        pd.RangeIndex(start=-3, stop=4, step=1),
        pytest.param(pd.RangeIndex(start=0, stop=10, step=2), marks=pytest.mark.slow),
        pytest.param(
            pd.RangeIndex(start=100, stop=-100, step=-5), marks=pytest.mark.slow
        ),
        pytest.param(pd.RangeIndex(start=100, stop=0, step=-7), marks=pytest.mark.slow),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.TimedeltaIndex(
            ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
        ),
        pd.PeriodIndex(
            year=[2000, 2000, 2001, 2001, 2000, 2004], quarter=[2, 2, 2, 3, 1, 1]
        ),
        pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
    ],
)
def test_index_argminmax(index, memory_leak_check):
    def impl1(index):
        return index.argmin()

    def impl2(index):
        return index.argmax()

    # TODO: support Index.argmin() / Index.argmax() 1D_Var
    check_func(impl1, (index,), only_seq=True)
    check_func(impl2, (index,), only_seq=True)
    check_func(impl1, (index,), only_1D=True)
    check_func(impl2, (index,), only_1D=True)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 5, 2, 1, 0]),
        pd.Index([0.5, -13.0, -0.5, 2.75, 0.5, 64.1]),
        pd.Index(pd.array([None, 1, None, 5, 2, 1, 3, None, None])),
        pd.Index([True, True, True, False, True]),
        pd.Index(pd.array([True, True, True, False, True])),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(-100, 100, 7),
        pd.RangeIndex(100, -100, -9),
        pd.RangeIndex(0, 10, -1),
        pd.CategoricalIndex([1, 5, 1, 1, 2, 1, 5, 2, 1, 3], ordered=True),
        pd.CategoricalIndex(list("ZEBRALIBRARY"), ordered=True),
    ],
)
def test_numeric_range_min_max(index):
    def impl1(I):
        return I.min()

    def impl2(I):
        return I.max()

    # Descending RangeIndex distributed min/max not supported yet [BE-2944]
    dist_test = not (isinstance(index, pd.RangeIndex) and (index.step < 0))
    check_func(impl1, (index,), dist_test=dist_test)
    check_func(impl2, (index,), dist_test=dist_test)


@pytest.mark.parametrize(
    "args",
    [
        (pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9]), 1),
        (pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9]), 5),
        pytest.param(
            (pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9]), 8), marks=pytest.mark.slow
        ),
        pytest.param(
            (pd.Index([1, 5, 2, 1, 0, 3, 1, 2, 5, 1]), 1), marks=pytest.mark.slow
        ),
        pytest.param(
            (pd.Index([1, 5, 2, 1, 0, 3, 1, 2, 5, 1]), 5), marks=pytest.mark.slow
        ),
        pytest.param(
            (pd.Index([1, 5, 2, 1, 0, 3, 1, 2, 5, 1]), 8), marks=pytest.mark.slow
        ),
        pytest.param(
            (
                pd.Index(
                    [
                        0.7573456092417888,
                        0.21589999967248008,
                        0.8671567646182514,
                        0.383775019426454,
                        0.21589999967248008,
                        0.5937443415123859,
                        0.5583538962837552,
                        0.5583538962837552,
                        0.6448399071221529,
                        0.383775019426454,
                    ]
                ),
                0.21589999967248008,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Index(
                    [
                        0.7573456092417888,
                        0.21589999967248008,
                        0.8671567646182514,
                        0.383775019426454,
                        0.21589999967248008,
                        0.5937443415123859,
                        0.5583538962837552,
                        0.5583538962837552,
                        0.6448399071221529,
                        0.383775019426454,
                    ]
                ),
                0.314159265359,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index([True, True, True, True, False, True, False, False]), True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index([True, True, True, True, True]), False), marks=pytest.mark.slow
        ),
        (pd.Index(["A", "L", "P", "H", "A", "B", "E", "T"]), "A"),
        (pd.Index(["A", "L", "P", "H", "A", "B", "E", "T"]), "a"),
        pytest.param(
            (pd.Index(["A", "L", "P", "H", "A", "B", "E", "T"]), "B"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index(["A", "L", "P", "H", "A", "B", "E", "T"]), "C"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Index(["A", "L", "P", "H", "A", "B", "E", "T"]), "AL"),
            marks=pytest.mark.slow,
        ),
        (pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]), 1),
        (pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]), 4),
        # Unskip after [BE-2811] resolved
        pytest.param(
            (pd.CategoricalIndex([True, True, True, True, True]), True),
            marks=(pytest.mark.slow, pytest.mark.skip),
        ),
        # Unskip after [BE-2811] resolved
        pytest.param(
            (pd.CategoricalIndex([True, True, True, True, True]), False),
            marks=(pytest.mark.slow, pytest.mark.skip),
        ),
        pytest.param(
            (pd.CategoricalIndex([1.0, 1.1, 1.2, 1.8, 1.5, 1.1]), 1.5),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.CategoricalIndex([1.0, 1.1, 1.2, 1.8, 1.5, 1.1]), 1.6),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.CategoricalIndex(["A", "B", "C", "A", "A", "C", "A", "B"]), "A"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.CategoricalIndex(["A", "B", "C", "A", "A", "C", "A", "B"]), "D"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.CategoricalIndex(
                    [
                        pd.Timestamp("2018-01-01"),
                        pd.Timestamp("2018-02-01"),
                        pd.Timestamp("2018-04-01"),
                        pd.Timestamp("2018-05-01"),
                        pd.Timestamp("2018-06-01"),
                    ]
                ),
                pd.Timestamp("2018-02-01"),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.CategoricalIndex(
                    [
                        pd.Timestamp("2018-01-01"),
                        pd.Timestamp("2018-02-01"),
                        pd.Timestamp("2018-04-01"),
                        pd.Timestamp("2018-05-01"),
                        pd.Timestamp("2018-06-01"),
                    ]
                ),
                pd.Timestamp("2018-03-01"),
            ),
            marks=pytest.mark.slow,
        ),
        (pd.RangeIndex(start=0, stop=10, step=2), 4),
        (pd.RangeIndex(start=0, stop=10, step=2), 5),
        (
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            pd.Timestamp("2018-04-24"),
        ),
        (
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            pd.Timestamp("2018-04-26"),
        ),
        (
            pd.TimedeltaIndex(
                ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
            ),
            pd.Timedelta("1 days"),
        ),
        (
            pd.TimedeltaIndex(
                ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
            ),
            pd.Timedelta("4 days"),
        ),
        (
            pd.Index([b"asdga", b"asdga", b"", b"oihjb", bytes(2), b"CC", b"asdfl"]),
            b"CC",
        ),
        (
            pd.Index([b"asdga", b"asdga", b"", b"oihjb", bytes(2), b"CC", b"asdfl"]),
            b"asdgA",
        ),
    ],
)
def test_index_contains(args, memory_leak_check):
    def impl(idx, elem):
        return elem in idx

    idx, elem = args
    check_func(impl, (idx, elem), dist_test=False)


@pytest.mark.parametrize(
    "args",
    [
        (0, 1, 0),
        (0, 100, 0),
        (0, 100, 1),
        (0, 100, -1),
        (0, 100, 10),
        (99, -1, -1),
        (0, 100, 5),
        (15, 210, 27),
    ],
)
def test_range_index_malformed(args):
    # Compile time check: step passed in is zero
    def impl1(start, stop, step):
        return pd.RangeIndex(start, stop, step)

    # Compile time check: step sometimes zero
    def impl2(start, stop, step=0):
        return pd.RangeIndex(start, stop, step)

    # Compile time check: step always zero
    def impl3(start, stop):
        return pd.RangeIndex(start, stop, 0)

    # Runtime check: loop that un-obviously proceeds until step=0
    def impl4(start, stop, step):
        step = abs(step) + 4
        while True:
            if step % 2 == 0:
                step //= 2
            else:
                step = 3 * step + 1
            if step < 2:
                step -= 1
                break
        return pd.RangeIndex(start, stop, step)

    start, stop, step = args
    if step == 0:
        with pytest.raises(BodoError, match="Step must not be zero"):
            bodo.jit(impl1)(start=start, stop=stop, step=step)
        with pytest.raises(BodoError, match="Step must not be zero"):
            bodo.jit(impl2)(start=start, stop=stop, step=step)
    else:
        check_func(impl1, (start, stop, step), dist_test=False)
        with pytest.raises(BodoError, match="Step must not be zero"):
            bodo.jit(impl2)(start=start, stop=stop)
    with pytest.raises(BodoError, match="Step must not be zero"):
        bodo.jit(impl3)(start=start, stop=stop)
    with pytest.raises(ValueError, match="Step must not be zero"):
        bodo.jit(impl4)(start=start, stop=stop, step=step)


# Need to add the code and the check for the PeriodIndex
# pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 12, 0, 2, 1, 3, -4], dtype="Int64"),
        pd.Index([10.1, 12.1, 1.2, 3.1, -1.2, -3.1, 0.0], dtype="float64"),
        pd.Index([10, 12, 0, 1, 11, 12, 5, 3], dtype="UInt64"),
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
    pd.testing.assert_index_equal(pd.Index(bodo_func().values.to_numpy()), impl())


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

    bodo_out = bodo.jit(test_impl)(m_ind)
    bodo_out = pd.MultiIndex(
        levels=[
            v.values.to_numpy()
            if isinstance(v.values, pd.arrays.ArrowStringArray)
            else v
            for v in bodo_out.levels
        ],
        codes=bodo_out.codes,
        names=bodo_out.names,
    )
    pd.testing.assert_index_equal(bodo_out, m_ind)


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
        pd.Index([10, 12], dtype="Int64"),
        pd.Index([10.1, 12.1], dtype="float64"),
        pd.Index([10, 12], dtype="UInt64"),
        pd.Index(["A", "B"] * 4),
        pytest.param(pd.Index([b"A", b"B"] * 4), id="binary_idx"),
        pd.RangeIndex(1, 15, 2),
        pd.RangeIndex(-4, 8, 4),
        pd.RangeIndex(-10, -1, -3),
        pd.RangeIndex(-4, 8, -4),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        ),
        pytest.param(pd.timedelta_range(start="1D", end="6D", name="A")),
        # Note: only functional for homogenous Index types
        pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]),
        pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
        pd.CategoricalIndex([1.1, 2.2, 3.3, 1.13, 2.2, 4.05, 3.3]),
        pd.CategoricalIndex(["A", "a", "E", "e", "I", "i"]),
        pd.CategoricalIndex(
            [
                "blue",
                "green",
                "yellow",
                "purple",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "green",
                "blue",
                "yellow",
                "yellow",
                "purple",
                "orange",
                "purple",
                "purple",
                "red",
                "orange",
                "purple",
                "red",
                "yellow",
                "green",
                "orange",
                "blue",
                "purple",
                "blue",
                "orange",
                "red",
                "blue",
            ]
        ),
        pd.CategoricalIndex([True, False, False, True, False, True, True, True]),
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


@pytest.mark.parametrize(
    "args",
    [
        (pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9]), pd.Series([0, 1, 2, 3])),
        (
            pd.Index(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]),
            pd.Series(["A", "E", "I", "O", "U"]),
        ),
        (
            pd.Index([b"alpha", b"beta", b"gamma", b"delta", b"epsilon"]),
            pd.Series([b"alphabet", b"beta", b"ALPHA", b"delta", b"beta"]),
        ),
        (
            pd.Index([1.1, 2.2, 1.3, 1.4, 1.1, 1.2, 1.3, 2.4]),
            pd.Series([1.1, 1.2, 1.3, 1.5]),
        ),
        # Unskip after [BE-2811] resolved
        pytest.param(
            (
                pd.Index([True, True, True, True, False, True, False, False]),
                pd.Series([True]),
            ),
            marks=(pytest.mark.slow, pytest.mark.skip),
        ),
        (pd.RangeIndex(start=0, stop=100, step=10), pd.Series([0, 25, 50, 75])),
        (
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            pd.date_range(start="2018-04-25", end="2018-04-28", periods=5),
        ),
        (
            pd.TimedeltaIndex(
                ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
            ),
            pd.TimedeltaIndex(
                ["4 days", "2 days", "5 days", "2 days", "3 hours", "2 seconds"]
            ),
        ),
    ],
)
def test_index_isin(args, memory_leak_check):
    def impl(idx, elems):
        return idx.isin(elems)

    idx, elems = args
    # RangeIndex distributed isin not supported yet [BE-2944]
    dist_test = not isinstance(idx, pd.RangeIndex)
    check_func(impl, (idx, elems), dist_test=dist_test)
    # TODO: fix casting certain types of series to list or set
    if (
        not isinstance(idx, (pd.TimedeltaIndex, pd.DatetimeIndex))
        and idx.dtype != "object"
    ):
        check_func(impl, (idx, list(elems)), dist_test=dist_test)
        check_func(impl, (idx, set(elems)), dist_test=dist_test)
    check_func(impl, (idx, pd.Index(elems)), dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 12], dtype="Int64"),
        pd.Index([10.1, 12.1], dtype="float64"),
        pd.Index([10, 12], dtype="UInt64"),
        pd.Index(["A", "B"] * 4),
        pytest.param(pd.Index([b"A", b"B"] * 4), id="binary_idx"),
        pd.RangeIndex(1, 15, 2),
        pd.RangeIndex(-4, 8, 4),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        ),
        pytest.param(
            pd.timedelta_range(start="1D", end="3D", name="A"),
        ),
        pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]),
        pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
        pd.CategoricalIndex([1.1, 2.2, 3.3, 1.13, 2.2, 4.05, 3.3]),
        pd.CategoricalIndex(["A", "a", "E", "e", "I", "i"]),
        pd.CategoricalIndex(
            [
                "blue",
                "green",
                "yellow",
                "purple",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "green",
                "blue",
                "yellow",
                "yellow",
                "purple",
                "orange",
                "purple",
                "purple",
                "red",
                "orange",
                "purple",
                "red",
                "yellow",
                "green",
                "orange",
                "blue",
                "purple",
                "blue",
                "orange",
                "red",
                "blue",
            ]
        ),
    ],
)
def test_index_getitem(index, memory_leak_check):
    def test_impl_1(index):
        return index[0]

    def test_impl_2(index):
        return index[len(index) - 1]

    def test_impl_3(index):
        return index[:2]

    def test_impl_4(index):
        return index[1::2]

    # Unskip after [BE-2829] resolved
    def test_impl_5(index):
        return index[pd.Index([i % 2 == 0 for i in range(len(index))])]

    check_func(test_impl_1, (index,), dist_test=False)
    check_func(test_impl_2, (index,), dist_test=False)
    check_func(test_impl_3, (index,), dist_test=False)
    check_func(test_impl_4, (index,), dist_test=False)
    # Unskip after [BE-2829] resolved
    # check_func(test_impl_5, (index,), dist_test=False)


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

    I = pd.Index([1, 211, 333, 43, 51], dtype="Int64")
    check_func(test_impl, (I,))


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.RangeIndex(11), marks=pytest.mark.slow),
        # TODO: revert when [BE-2672] is resolved
        # pd.Index([10, 12, 1, 3, 2, 4, 5, -1], dtype="Int64"),
        pytest.param(
            pd.Index([10.1, 12.1, 1.1, 2.2, -1.2, 4.1, -2.1], dtype="float64"),
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

    check_func(test_impl, (index,), check_dtype=False)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([i for i in range(100)]),
        pytest.param(
            pd.Index([i for i in range(100)], dtype="UInt64"), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.Index([i for i in range(100)], dtype="float64"), marks=pytest.mark.slow
        ),
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
            pd.timedelta_range(start="3D", end="1D", periods=100),
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


def test_range_index_dce(memory_leak_check):
    """
    Tests a use case from BodoSQL where a RangeIndex is
    generated but cannot have its distribution determined
    because the uses have all been optimized out.

    The solution tests that the range index, whose name may
    be needed, can be properly eliminated.
    """

    def impl(df):
        df1 = df
        S0 = df1["A"]
        S1 = df1["B"]
        df2 = pd.DataFrame(
            {"EXPR$0": S0.max(), "__bodo_dummy__1": S1.max()},
            index=pd.RangeIndex(0, 1, 1, name="my_copied_name"),
        )
        df3 = df2[((df2["__bodo_dummy__1"] != 200))]
        df4 = df3.loc[
            :,
            [
                "EXPR$0",
            ],
        ]
        return df4

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)})
    check_func(impl, (df,))


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1]),
        pd.Index(
            [
                0.7573456092417888,
                0.21589999967248008,
                0.8671567646182514,
                0.383775019426454,
                0.21589999967248008,
                0.5937443415123859,
                0.5583538962837552,
                0.5583538962837552,
                0.6448399071221529,
                0.383775019426454,
            ]
        ),
        pd.Index(pd.array([1, 1, 2, 1, None, None, 1, 2, 3, 2, 1, 1, None])),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.Index([True, True, True, True, False, True, False, False]),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.Index([False, False, False, False, False]),
            marks=pytest.mark.skip,
        ),
        pd.Index(["A", "B"] * 4),
        pytest.param(
            pd.Index(
                [
                    "blue",
                    "green",
                    "yellow",
                    "purple",
                    "red",
                    "green",
                    "blue",
                    "purple",
                    "yellow",
                    "green",
                    "blue",
                    "yellow",
                    "yellow",
                    "purple",
                    "orange",
                    "purple",
                    "purple",
                    "red",
                    "orange",
                    "purple",
                    "red",
                    "yellow",
                    "green",
                    "orange",
                    "blue",
                    "purple",
                    "blue",
                    "orange",
                    "red",
                    "blue",
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pd.Index(["alpha", "Alpha", "alpha", "ALPHA", "alphonse", "a", "A"]),
        pd.CategoricalIndex(["A", "B", "C", "A", "A", "C", "A", "B"]),
        pytest.param(
            pd.CategoricalIndex([1, 5, 1, 2, 0, 1, 1, 1, 0, 5, 9]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex([0.0, 0.25, 0.5, 0.75, 0.25, 0.25, 0.75, 0.25]),
            marks=pytest.mark.slow,
        ),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.CategoricalIndex([True, True, True, True, False, True, False, False]),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2811] is resolved (boolean index issues)
        pytest.param(
            pd.CategoricalIndex([True, True, True, True, True]),
            marks=pytest.mark.skip,
        ),
        pd.RangeIndex(start=0, stop=7, step=1),
        pd.RangeIndex(start=0, stop=10, step=2),
        pd.RangeIndex(start=100, stop=0, step=-10),
        # Unskip after [BE-2812] is resolved (PeriodIndex.unique())
        pytest.param(
            pd.PeriodIndex(
                year=[2000, 2000, 2001, 2001, 2001], quarter=[1, 1, 1, 3, 2]
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2811] is resolved (PeriodIndex.unique())
        pytest.param(
            pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
            marks=pytest.mark.skip,
        ),
        pd.interval_range(0, 5),
        pd.IntervalIndex.from_tuples([(0, 1), (3, 4), (2, 3), (1, 2)]),
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(1.0, 1.5), (1.5, 3.0), (3.0, 4.0), (4.0, 10.1)]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(0, 1), (3, 4), (2, 3), (0, 1), (0, 1), (1, 2), (2, 3)]
            ),
            marks=pytest.mark.slow,
        ),
        # Unskip after [BE-2813] is resolved (intervals of datetime values)
        pytest.param(
            pd.interval_range(
                start=pd.Timestamp("2005-01-01"), end=pd.Timestamp("2005-01-31")
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2813] is resolved (intervals with shared start-value)
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(1, 2), (2, 3), (3, 5), (1, 2), (2, 5), (1, 5)]
            ),
            marks=pytest.mark.skip,
        ),
        pytest.param(
            pd.IntervalIndex.from_tuples([(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.IntervalIndex.from_tuples([(1, 5), (2, 5), (3, 7), (4, 7), (5, 7)]),
            marks=pytest.mark.slow,
        ),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
        pd.TimedeltaIndex(
            ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
        ),
        pd.Index([b"asdga", b"asdga", b"", b"oihjb", bytes(2), b"CC", b"asdfl"]),
        pytest.param(
            pd.Index(
                [
                    b"alpha",
                    b"beta",
                    b"gamma",
                    b"delta",
                    b"ALPHA",
                    b"a",
                    b"A",
                    b"A",
                    b"beta",
                    b"ALPHA",
                ]
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_unique(idx):
    def impl(idx):
        return idx.unique()

    # Descending RangeIndex distributed unique not supported yet [BE-2944]
    # IntervalIndex distributed unique not supported yet (related to [BE-2813])
    dist_test = not isinstance(idx, pd.IntervalIndex) and not (
        isinstance(idx, pd.RangeIndex) and idx.step < 0
    )
    check_func(impl, (idx,), dist_test=dist_test, sort_output=True)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1, 3]),
                pd.array([True, False] * 5),
                pd.Series([-1, -2, -3, -4, -5] * 2),
            ),
        ),
        pytest.param(
            (
                pd.Index([1.0, 2.0, 3.0, 2.0, 1.0]),
                pd.array([True, True, False, False, False]),
                pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
            ),
        ),
        # Unskip after [BE-2811] Resolved
        pytest.param(
            (
                pd.Index([bool((i % 3 == 0) or (i % 5 == 0)) for i in range(20)]),
                pd.array([bool((i % 4 == 0) or (i % 7 == 0)) for i in range(20)]),
                pd.Series([bool((i % 6 == 0) or (i % 9 == 0)) for i in range(20)]),
            ),
            marks=pytest.mark.skip,
        ),
        pytest.param(
            (
                pd.Index(list("ABCDEFGHIJ")),
                pd.array([True, False, True, True, False] * 2),
                pd.Series(list("abcdefghij")),
            ),
        ),
        pytest.param(
            (
                pd.Index(
                    [
                        bytes(chr(i + k) + chr(j), "utf-8")
                        for k in range(2)
                        for i in range(65, 69)
                        for j in range(66, 68)
                    ]
                ),
                pd.array(
                    [
                        bool((i % 10 == 0) or (i % 7 == 0) or (i % 13 == 0))
                        for i in range(16)
                    ]
                ),
                pd.Series([bytes(chr(k), "utf-8") for k in range(49, 57)] * 2),
            ),
        ),
        pytest.param(
            (
                pd.Index(pd.date_range("2018-01-01", "2019-01-01", periods=12)),
                pd.array([True, False, False, True, True, False] * 2),
                pd.Series([pd.Timestamp(f"200{i}-01-01") for i in range(6)] * 2),
            ),
        ),
        pytest.param(
            (
                pd.TimedeltaIndex(
                    [
                        f"{amt} {unit}"
                        for amt in range(1, 4)
                        for unit in ["minutes", "hours"]
                    ]
                ),
                pd.array([True, True, False, True, False, True]),
                pd.Series(
                    pd.TimedeltaIndex(
                        [
                            f"{amt} {unit}"
                            for amt in range(1, 3)
                            for unit in ["seconds", "minutes", "hours"]
                        ]
                    )
                ),
            ),
        ),
        pytest.param(
            (
                pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3, 1, 5, 2, 5, 1]),
                pd.array([False, True, True, True, False] * 3),
                pd.Series([0, 1, 2, 3, 5] * 3),
            ),
        ),
        pytest.param(
            (
                pd.CategoricalIndex(list("abcaacab")),
                pd.array([False, True, False, True] * 2),
                pd.Series(list("cbcbbcbc")),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.CategoricalIndex(list("abcaacab")),
                pd.array([False, False, True, True] * 2),
                pd.Series(list("cabcabca")),
            ),
            marks=pytest.mark.slow,
        ),
        # Unskip after [BE-2910] is resolved
        pytest.param(
            (
                pd.CategoricalIndex([1, 5, 2, 1, 0, 1, 5, 2, 1, 3, 1, 5, 2, 5, 1]),
                pd.array([False, True, True, True, False] * 3),
                pd.Series(list(range(12))),
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2910] is resolved
        pytest.param(
            (
                pd.CategoricalIndex(list("abcaacab")),
                pd.array([False, True, False, True] * 2),
                pd.Series(pd.CategoricalIndex(list("cacacaca"))),
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2910] is resolved
        pytest.param(
            (
                pd.CategoricalIndex(list("abcaacab")),
                pd.array([False, True, False, True] * 2),
                pd.Series(pd.CategoricalIndex(list("ABCABCAB"))),
            ),
            marks=pytest.mark.skip,
        ),
        pytest.param(
            (
                pd.RangeIndex(0, 10, 1),
                pd.array([bool((i % 3 == 0) or (i % 4 == 0)) for i in range(10)]),
                pd.Series(list(range(0, 20, 2))),
            ),
        ),
        pytest.param(
            (
                pd.RangeIndex(0, 20, 2),
                pd.array([bool((i % 3 == 0) or (i % 4 == 0)) for i in range(10)]),
                pd.Series(list(range(-1, -11, -1))),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.RangeIndex(0, -100, -15),
                pd.array([True, True, False, False, True, True, False]),
                pd.Series(list(range(0, 100, 15))),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.RangeIndex(100, -101, -25),
                pd.array([True, False, True] * 3),
                pd.Series([42, 72, -16] * 3),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Index([1, 2, 3, 4, 5]),
                pd.array([True, False, False, True, True]),
                pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Index([0.1, 0.2, 0.3, 0.4, 0.5]),
                pd.array([False, True, True, False, True]),
                pd.Series([1, 2, 3, 4, 5]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Index([1, 2, 3, 4, 5]),
                pd.array([True, False, True, False, True]),
                np.array([10, 20, 30, 40, 50]),
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_where_putmask(args):
    def impl1(idx, con):
        return idx.where(con)

    def impl2(idx, con, oth):
        return idx.where(con, oth)

    def impl3(idx, con, oth):
        return idx.putmask(con, oth)

    idx, con, oth = args
    # Non-simple RangeIndex distributed nunique not supported yet [BE-2944]
    dist_test = not (
        isinstance(idx, pd.RangeIndex) and (idx.start != 0 or idx.step != 1)
    )
    check_func(impl1, (idx, con), dist_test=dist_test)
    if isinstance(
        idx,
        (
            pd.core.indexes.numeric.Int64Index,
            pd.core.indexes.numeric.Float64Index,
            pd.RangeIndex,
        ),
    ):
        check_func(impl2, (idx, con, np.nan), dist_test=dist_test)
    check_func(impl2, (idx, con, oth), dist_test=dist_test)
    check_func(impl2, (idx, con, oth[0]), dist_test=dist_test)
    check_func(impl3, (idx, con, oth), dist_test=dist_test)
    check_func(impl3, (idx, con, oth[0]), dist_test=dist_test)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1]),
        pytest.param(
            pd.Index(
                [
                    0.7573456092417888,
                    0.21589999967248008,
                    0.8671567646182514,
                    0.383775019426454,
                    0.21589999967248008,
                    0.5937443415123859,
                    0.5583538962837552,
                    0.5583538962837552,
                    0.6448399071221529,
                    0.383775019426454,
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pd.Index(pd.array([1, 1, 2, 1, None, None, 1, 2, 3, 2, 1, 1, None])),
        pd.Index([True, True, True, True, False, True, False, False]),
        pytest.param(
            pd.Index([False, False, False, False, False]), marks=pytest.mark.slow
        ),
        pd.Index(["A", "B"] * 4),
        pytest.param(
            pd.Index(
                [
                    "blue",
                    "green",
                    "yellow",
                    "purple",
                    "red",
                    "green",
                    "blue",
                    "purple",
                    "yellow",
                    "green",
                    "blue",
                    "yellow",
                    "yellow",
                    "purple",
                    "orange",
                    "purple",
                    "purple",
                    "red",
                    "orange",
                    "purple",
                    "red",
                    "yellow",
                    "green",
                    "orange",
                    "blue",
                    "purple",
                    "blue",
                    "orange",
                    "red",
                    "blue",
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["alpha", "Alpha", "alpha", "ALPHA", "alphonse", "a", "A"]),
            marks=pytest.mark.slow,
        ),
        pd.CategoricalIndex(["A", "B", "C", "A", "A", "C", "A", "B"]),
        pytest.param(
            pd.CategoricalIndex([1, 5, 1, 2, 0, 1, 1, 1, 0, 5, 9]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex([0.0, 0.25, 0.5, 0.75, 0.25, 0.25, 0.75, 0.25]),
            marks=pytest.mark.slow,
        ),
        pd.CategoricalIndex([True, True, True, True, False, True, False, False]),
        pytest.param(
            pd.CategoricalIndex([True, True, True, True, True]), marks=pytest.mark.slow
        ),
        pd.RangeIndex(start=0, stop=100, step=1),
        pd.RangeIndex(start=0, stop=10, step=2),
        pytest.param(
            pd.RangeIndex(start=100, stop=0, step=-10), marks=pytest.mark.slow
        ),
        pd.PeriodIndex(
            year=[2000, 2000, 2001, 2001, 2001, 2001], quarter=[1, 1, 1, 3, 3, 2]
        ),
        pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.interval_range(0, 5),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples([(0, 1), (3, 4), (2, 3), (1, 2), (-1, 8)]),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(1.0, 1.5), (1.5, 3.0), (3.0, 4.0), (4.0, 10.1), (10.1, 11.0)]
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(0, 1), (3, 4), (2, 3), (0, 1), (0, 1), (1, 2), (2, 3)]
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.interval_range(
                start=pd.Timestamp("2005-01-01"), end=pd.Timestamp("2005-01-31")
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples(
                [(1, 2), (2, 3), (3, 5), (1, 2), (2, 5), (1, 5)]
            ),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples([(1, 5), (2, 6), (3, 7), (4, 8), (5, 9)]),
            marks=pytest.mark.skip,
        ),
        # Unskip after [BE-2814] is resolved
        pytest.param(
            pd.IntervalIndex.from_tuples([(1, 5), (5, 6), (2, 5), (3, 7), (4, 6)]),
            marks=pytest.mark.skip,
        ),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=8),
        pd.TimedeltaIndex(
            ["1 days", "2 days", "3 days", "2 days", "3 hours", "2 minutes"]
        ),
        pd.Index([b"asdga", b"asdga", b"", b"oihjb", bytes(2), b"CC", b"asdfl"]),
        pytest.param(
            pd.Index(
                [
                    b"alpha",
                    b"beta",
                    b"gamma",
                    b"delta",
                    b"ALPHA",
                    b"a",
                    b"A",
                    b"A",
                    b"beta",
                    b"ALPHA",
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(
                pd.array(
                    [
                        4,
                        "a",
                        "b",
                        "c",
                        1,
                        2,
                        False,
                        0.5,
                        3,
                        "a",
                        "2",
                        97,
                        "97",
                        3,
                        0.5,
                        0.51,
                        3,
                        3,
                        "1 days",
                        "B",
                        "C",
                        2,
                        "CCC",
                        True,
                        False,
                    ]
                )
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_index_nunique(idx):
    def impl(idx, dropna):
        return idx.nunique(dropna=dropna)

    # Non-simple RangeIndex distributed nunique not supported yet [BE-2944]
    dist_test = not (
        isinstance(idx, pd.RangeIndex) and (idx.start != 0 or idx.step != 1)
    )
    check_func(impl, (idx, True), dist_test=dist_test)
    check_func(impl, (idx, False), dist_test=dist_test)


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

    def test_append(idx):
        return idx.append()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_append)(idx=pd.Index(data))

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

    def test_is_(idx):
        return idx.is_()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_)(idx=pd.Index(data))

    def test_is_mixed(idx):
        return idx.is_mixed()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_mixed)(idx=pd.Index(data))

    def test_is_type_compatible(idx):
        return idx.is_type_compatible()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_is_type_compatible)(idx=pd.Index(data))

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

    def test_ravel(idx):
        return idx.ravel()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_ravel)(idx=pd.Index(data))

    def test_reindex(idx):
        return idx.reindex()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_reindex)(idx=pd.Index(data))

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

    def test_sortlevel(idx):
        return idx.sortlevel()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_sortlevel)(idx=pd.Index(data))

    def test_str(idx):
        return idx.str()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_str)(idx=pd.Index(data))

    def test_to_flat_index(idx):
        return idx.to_flat_index()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_flat_index)(idx=pd.Index(data))

    def test_to_native_types(idx):
        return idx.to_native_types()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_to_native_types)(idx=pd.Index(data))

    def test_transpose(idx):
        return idx.transpose()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_transpose)(idx=pd.Index(data))

    def test_value_counts(idx):
        return idx.value_counts()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_value_counts)(idx=pd.Index(data))

    def test_view(idx):
        return idx.view()

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_view)(idx=pd.Index(data))


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


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 3, 0, 3, 2]),
        pd.Index([-2, -1, 0, 0, 0, 1, 2]),
        pd.Index([0, 0, 0, 0, 0]),
        pd.Index([5, 4, 3, 2, 1]),
        pd.Index(pd.array([5, 4, None, 2, 1, None, None])),
        pd.Index(pd.array([5, 0, None, 2, 1, 0, None])),
        pd.Index(pd.array([0, 0, None, 0, 0, 0, None])),
        pytest.param(pd.Index([0, -4, 2, -1, 3, 0, 0]), marks=pytest.mark.slow),
        pytest.param(pd.Index([0, 1, 2, 3, 4, 5]), marks=pytest.mark.slow),
        pytest.param(pd.Index([-2, -1]), marks=pytest.mark.slow),
        pytest.param(pd.Index([-2, -1, 0]), marks=pytest.mark.slow),
        pytest.param(pd.Index([-2, -1, 0, 1]), marks=pytest.mark.slow),
        pytest.param(pd.Index([2, 1, 0]), marks=pytest.mark.slow),
        pytest.param(pd.Index([2, 1, 0, 1, 2]), marks=pytest.mark.slow),
        pd.Index([True, True, True, True, True]),
        pd.Index([True, False, True, True, True]),
        pd.Index([False, False, False, False, False]),
        pd.Index([]),
        pd.Index(["", "", "", "A", "B", "C"]),
        pd.Index(["alpha", "beta", "gamma", "delta"]),
        pd.Index(["", ""]),
        pytest.param(
            pd.Index(["cab", "ab", "b", "", "b", "ba", "bac"]), marks=pytest.mark.slow
        ),
        pd.Index([b"", b"", b"", b"A", b"B", b"C"]),
        pd.Index([b"pi", b"theta", b"mu", b"kappa"]),
        pd.Index([b"", b""]),
        pytest.param(
            pd.Index([b"cab", b"ab", b"b", b"", b"b", b"ba", b"bac"]),
            marks=pytest.mark.slow,
        ),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(-5, 5, 1),
        pd.RangeIndex(-5, 5, 2),
        pd.RangeIndex(-5, 0, 1),
        pytest.param(pd.RangeIndex(0, 10, 3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, 1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, 3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, 50), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, -3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(0, 10, -30), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -3), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(10, 0, -30), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(5, -5, -1), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(5, -5, -2), marks=pytest.mark.slow),
        pytest.param(pd.RangeIndex(5, 0, -1), marks=pytest.mark.slow),
    ],
)
def test_index_any_all(index):
    def impl1(index):
        return index.any()

    def impl2(index):
        return index.all()

    # Non-simple RangeIndex distributed any/all not supported yet [BE-2944]
    dist_test = not (
        isinstance(index, pd.RangeIndex) and (index.start != 0 or index.step != 1)
    )

    # Bodo diverges from the Pandas API for String & Binary Index by returning
    # a boolean instead of a string.
    if not isinstance(index.dtype, pd.Int64Dtype):
        boolean_any = any(list(index))
        boolean_all = all(list(index))
        check_func(impl1, (index,), dist_test=dist_test, py_output=boolean_any)
        check_func(impl2, (index,), dist_test=dist_test, py_output=boolean_all)
        check_func(impl1, (index,), dist_test=dist_test, py_output=boolean_any)
        check_func(impl2, (index,), dist_test=dist_test, py_output=boolean_all)
    else:
        check_func(impl1, (index,), dist_test=dist_test)
        check_func(impl2, (index,), dist_test=dist_test)
        check_func(impl1, (index,), dist_test=dist_test)
        check_func(impl2, (index,), dist_test=dist_test)


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
        pd.Index([10, 12], dtype="Int64"),
        pd.Index([10.1, 12.1], dtype="float64"),
        pd.Index([10, 12], dtype="UInt64"),
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

    # Nullable int index example:
    # data = 16 bytes
    # null_bit_map= 1 byte if 1 rank, 2 bytes if > 1 rank
    # Total = 16 + num_ranks bytes
    elif isinstance(index, pd.Index) and str(index.dtype) in ["Int64", "UInt64"]:
        py_out = 16 + (1 if bodo.get_size() == 1 else 2)
        check_func(impl, (index,), py_output=py_out, only_1D=True)

    else:
        check_func(impl, (index,))


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 5, 2, 1, 0], name="I1"),
        pd.Index(pd.array([2, 4, 6, 8, 10], dtype="int8")),
        pd.Index([1, 4, 9, 16, 32], dtype="uint16", name="I3"),
        pd.Index([True, False, False, False, True]),
        pd.Index([0.5, 0.9, 1.3, 0.5, 0.5], name="I5"),
        pd.Index(pd.array([1, 2, None, 3, None, None, 4])),
        pd.Index([], name="I7"),
        pd.Index(list("sternrents")),
        pd.Index([b"A", b"B", b"C", b"D", b"E"], name="I9"),
        pd.date_range("2018-01-01", "2018-01-06"),
        pd.date_range("2018-01-01", "2017-01-06", name="I11"),
        pd.timedelta_range("1D", "7D"),
        pd.timedelta_range("7D", "1D", name="I13"),
        pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]),
        pd.CategoricalIndex(list("AAEAAEIOUOIEA"), ordered=True, name="I15"),
        pd.CategoricalIndex([], name="I16"),
        pd.RangeIndex(0, 26, 5, name="I17"),
        pd.RangeIndex(30, -1, -10),
        pd.RangeIndex(0, 10, -1, name="I19"),
        pd.MultiIndex.from_product([[1, 2, 3], [4, 5, 6]], names=("I20A", "I20B")),
        pd.MultiIndex.from_arrays([[1, 2, 1, 2, 1, 2], list("AAABBB"), list("CECEDE")]),
        pd.MultiIndex.from_product([pd.array([], dtype="int64")]),
    ],
)
def test_index_simple_attributes(index):
    """Tests simple Index attributes"""

    def impl1(idx):
        return idx.T

    # Return several attributes at once in a tuple to reduce the number of
    # functions that need to be compiled and run with check_func
    def impl2(idx):
        return (
            idx.size,
            idx.shape,
            idx.ndim,
            idx.nlevels,
            idx.empty,
            idx.is_all_dates,
            idx.inferred_type,
            idx.name,
            idx.names,
        )

    def impl3(idx):
        return idx.dtype

    # Distributed descending RangeIndex not supported [BE-2944]
    dist_test = not (isinstance(index, pd.RangeIndex) and index.step < 0)

    # Cannot return or boolean index (see [BE-2811])
    if index.inferred_type != "boolean":
        # 1D_Var transpose banned
        check_func(impl1, (index,), only_seq=True)
        if dist_test:
            check_func(impl1, (index,), only_1D=True)

    check_func(impl2, (index,), dist_test=dist_test)

    # Bodo diverges from the Pandas API by always returning the numpy dtype in
    # ambiguous cases.
    if isinstance(index.dtype, np.dtype):
        check_func(impl3, (index,), dist_test=dist_test)


@pytest.mark.parametrize(
    "index",
    [
        pd.interval_range(0, 5, name="I1"),
        pd.interval_range(0.5, 5.5),
        pd.interval_range(0, 5, closed="left", name="I3"),
        pd.interval_range(0, 5, closed="both"),
        pd.interval_range(0, 5, closed="neither", name="I5"),
    ],
)
def test_interval_index_simple_attributes(index):
    """Tests simple IntervalIndex attributes"""

    def impl1(idx):
        return idx.T

    # Return several attributes at once in a tuple to reduce the number of
    # functions that need to be compiled and run with check_func
    def impl2(idx):
        return (
            idx.size,
            idx.shape,
            idx.ndim,
            idx.nlevels,
            idx.empty,
            idx.is_all_dates,
            idx.inferred_type,
            idx.name,
            idx.names,
        )

    # Currently unable to return an IntervalIndex where closed != right
    if index.closed == "right":
        # 1D_Var transpose banned
        check_func(impl1, (index,), only_seq=True)
        check_func(impl1, (index,), only_1D=True)

    check_func(
        impl2,
        (index,),
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.period_range("2018-01-01", "2018-01-06", freq="D", name="P1"),
        pd.period_range("2018-01-01", "2018-06-01", freq="M"),
        pd.period_range("2018-01-01", "2017-06-01", freq="Y", name="P3"),
    ],
)
def test_period_index_simple_attributes(index):
    """Tests simple PeriodIndex attributes"""

    def impl1(idx):
        return idx.T

    # Return several attributes at once in a tuple to reduce the number of
    # functions that need to be compiled and run with check_func
    def impl2(idx):
        return (
            idx.size,
            idx.shape,
            idx.ndim,
            idx.nlevels,
            idx.empty,
            idx.is_all_dates,
            idx.inferred_type,
            idx.name,
            idx.names,
        )

    # 1D_Var transpose banned
    check_func(impl1, (index,), only_seq=True)
    check_func(impl1, (index,), only_1D=True)

    check_func(
        impl2,
        (index,),
    )


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
    "index",
    [
        pd.Index([1, 2, 3, 4, 5]),
        pd.Index([-5.0, -6.1, -7.2, 3.8, -6.1], name="f"),
        pd.date_range("2018-01-01", "2018-06-01", freq="M"),
        pd.timedelta_range("1D", "7D"),
        pd.Index(["A", "E", "I", "O", "U", "A"]),
        pd.CategoricalIndex([1, 1, 2, 1, 1, 2, 3, 2, 1]),
        pd.CategoricalIndex(list("abcaacab")),
        pd.RangeIndex(0, 6, 1),
        pd.RangeIndex(5, -1, -1),
        pd.RangeIndex(0, 10, 2),
        pd.RangeIndex(-3, 30, 7),
        # Unskip after [BE-3016] resolved
        pytest.param(pd.Index([b"a", b"b", b"c", b"a", b"b"]), marks=pytest.mark.skip),
        # Unskip after [BE-711] has made progress including (at the very least):
        # - alloc_type for IntervalArray
        # - isna for IntervalArray
        # - IntervalArray getitem (integer scalar)
        # - IntervalArray setitem (slice)
        pytest.param(pd.interval_range(0, 5), marks=pytest.mark.skip),
    ],
)
def test_index_repeat(index):
    def impl(index, n):
        return index.repeat(n)

    dist_test = not (isinstance(index, pd.RangeIndex) and index.step < 0)
    check_func(impl, (index, 1), dist_test=dist_test)
    check_func(impl, (index, 3), dist_test=dist_test)
    check_func(impl, (index, 0), dist_test=dist_test)
    check_func(
        impl, (index, np.array([i % 3 for i in range(len(index))])), dist_test=dist_test
    )


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
