# Copyright (C) 2019 Bodo Inc. All rights reserved.
import datetime
import operator
import os
import random
import unittest
from dataclasses import dataclass
from decimal import Decimal

import numba
import numba.np.ufunc_db
import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    _get_dist_arg,
    _test_equal,
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    get_start_end,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call_assign


@dataclass
class SeriesReplace:
    series: pd.Series
    to_replace: (int, str)
    value: (int, str)


@dataclass
class WhereNullable:
    series: pd.Series
    cond: pd.array
    other: pd.Series

    def __iter__(self):
        return iter([self.series, self.cond, self.other])


_cov_corr_series = [
    (pd.Series(x), pd.Series(y))
    for x, y in [
        ([np.nan, -2.0, 3.0, 9.1], [np.nan, -2.0, 3.0, 5.0]),
        # TODO(quasilyte): more intricate data for complex-typed series.
        # Some arguments make assert_almost_equal fail.
        # Functions that yield mismaching results: _column_corr_impl and _column_cov_impl.
        (
            [complex(-2.0, 1.0), complex(3.0, 1.0)],
            [complex(-3.0, 1.0), complex(2.0, 1.0)],
        ),
        ([complex(-2.0, 1.0), complex(3.0, 1.0)], [1.0, -2.0]),
        ([1.0, -4.5], [complex(-4.5, 1.0), complex(3.0, 1.0)]),
    ]
]


GLOBAL_VAL = 2


# TODO: integer Null and other Nulls
# TODO: list of datetime.datetime, categorical, timedelta, ...
@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [
        555,
        [2, 3, 5],
        [2.1, 3.2, 5.4],
        [True, False, True],
        ["A", "C", "AB"],
        np.array([2, 3, 5]),
        pd.Series([2, 5, 6]),
        pd.Series([2.1, 5.3, 6.1], name="C"),
        pd.Series(["A", "B", "CC"]),
        pd.Series(["A", "B", "CC"], name="A"),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.Int64Index([10, 12, 13]),
        pd.Int64Index([10, 12, 14], name="A"),
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        [2, 3, 5],
        [2.1, 3.2, 5.4],
        ["A", "C", "AB"],
        np.array([2, 3, 5]),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.Int64Index([10, 12, 13]),
        pd.Int64Index([10, 12, 14], name="A"),
        pd.RangeIndex(1, 4, 1),
        None,
    ],
)
@pytest.mark.parametrize("name", [None, "ABC"])
def test_series_constructor(data, index, name, memory_leak_check):
    # set Series index to avoid implicit alignment in Pandas case
    if isinstance(data, pd.Series) and index is not None:
        data.index = index

    # bypass literal as data and index = None
    if isinstance(data, int) and index is None:
        return

    def impl(d, i, n):
        return pd.Series(d, i, name=n)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(data, index, name), impl(data, index, name), check_dtype=False
    )


@pytest.mark.slow
def test_series_constructor2(memory_leak_check):
    def impl(d, i, n):
        return pd.Series(d, i, name=n)

    bodo_func = bodo.jit(impl)
    data1 = pd.Series(["A", "B", "CC"], name="A")
    pd.testing.assert_series_equal(
        bodo_func(data1, None, None), impl(data1, None, None), check_dtype=False
    )

    data2 = pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A")
    pd.testing.assert_series_equal(
        bodo_func(data2, None, None), impl(data2, None, None), check_dtype=False
    )


@pytest.mark.slow
def test_series_constructor_dtype1(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype=np.int32)

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


@pytest.mark.slow
def test_series_constructor_dtype2(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype="int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


@pytest.mark.slow
def test_series_constructor_init_str(memory_leak_check):
    """test passing "str" to Series constructor to populate a Series"""

    def impl(n):
        I = np.arange(n)
        return pd.Series(index=I, data=np.nan, dtype="str")

    check_func(impl, (111,))


@pytest.mark.slow
def test_series_constructor_int_arr(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype="Int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))
    check_func(impl, (np.array([1, 4, 1, np.nan, 0], dtype=np.float32),))


@pytest.mark.slow
def test_series_constructor_range(memory_leak_check):
    def impl(start, stop, step):
        return pd.Series(range(10))

    check_func(impl, (0, 10, 1))
    check_func(impl, (10, -1, -3))


@pytest.mark.slow
def test_series_cov_ddof(memory_leak_check):
    def test_impl(s1, s2, ddof=1):
        return s1.cov(s2, ddof=ddof)

    s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
    s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
    check_func(test_impl, (s1, s2, 0))
    check_func(test_impl, (s1, s2, 1))
    check_func(test_impl, (s1, s2, 2))
    check_func(test_impl, (s1, s2, 3))
    check_func(test_impl, (s1, s2, 4))
    check_func(test_impl, (pd.Series([], dtype=float), pd.Series([], dtype=float)))


# using length of 5 arrays to enable testing on 3 ranks (2, 2, 1 distribution)
# zero length chunks on any rank can cause issues, TODO: fix
# TODO: other possible Series types like Categorical, dt64, td64, ...
@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    Decimal("1.6"),
                    Decimal("-0.2"),
                    Decimal("44.2"),
                    np.nan,
                    Decimal("0"),
                ]
                * 2
            ),
            id="series_val0",
        ),
        pytest.param(
            pd.Series([1, 8, 4, 11, -3]), marks=pytest.mark.slow, id="series_val1"
        ),
        pytest.param(
            pd.Series([True, False, False, True, True]),
            marks=pytest.mark.slow,
            id="series_val2",
        ),  # bool array without NA
        pytest.param(
            pd.Series([True, False, False, np.nan, True] * 2), id="series_val3"
        ),  # bool array with NA
        pytest.param(
            pd.Series([1, 8, 4, 0, 3], dtype=np.uint8),
            marks=pytest.mark.slow,
            id="series_val4",
        ),
        pytest.param(pd.Series([1, 8, 4, 10, 3], dtype="Int32"), id="series_val5"),
        pytest.param(
            pd.Series([1, 8, 4, -1, 2], name="ACD"),
            marks=pytest.mark.slow,
            id="series_val6",
        ),
        pytest.param(
            pd.Series([1, 8, 4, 1, -3], [3, 7, 9, 2, 1]),
            marks=pytest.mark.slow,
            id="series_val7",
        ),
        pytest.param(
            pd.Series(
                [1.1, np.nan, 4.2, 3.1, -3.5], [3, 7, 9, 2, 1], name="series_val8"
            ),
        ),
        pytest.param(
            pd.Series([1, 2, 3, -1, 6], ["A", "BA", "", "DD", "GGG"]), id="series_val9"
        ),
        pytest.param(
            pd.Series(["A", "B", "CDD", "AA", "GGG"]),
            marks=pytest.mark.slow,
            id="series_val10",
        ),  # TODO: string with Null (np.testing fails)
        pytest.param(
            pd.Series(["A", "B", "CG", "ACDE", "C"], [4, 7, 0, 1, -2]),
            id="series_val11",
        ),
        pytest.param(
            pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
            id="series_val12",
        ),
        pytest.param(
            pd.Series(
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date
            ),
            id="series_val13",
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.timedelta(3, 3, 3),
                    datetime.timedelta(2, 2, 2),
                    datetime.timedelta(1, 1, 1),
                    None,
                    datetime.timedelta(5, 5, 5),
                ]
            ),
            id="series_val14",
        ),
        pytest.param(
            pd.Series(
                [3, 5, 1, -1, 2],
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            ),
            marks=pytest.mark.slow,
            id="series_val15",
        ),
        pytest.param(
            pd.Series([["a", "bc"], ["a"], ["aaa", "b", "cc"], None, ["xx", "yy"]]),
            id="series_val16",
        ),
        pytest.param(
            pd.Series([[1, 2], [3], [5, 4, 6], None, [-1, 3, 4]]),
            id="series_val17",
        ),
        pytest.param(
            pd.Series(["AA", "BB", "", "AA", None, "AA"] * 2, dtype="category"),
            id="series_val18",
        ),
        pytest.param(
            pd.Series(pd.Categorical([1, 2, 5, None, 2] * 2, ordered=True)),
            id="series_val19",
        ),
        pytest.param(
            pd.Series(pd.date_range(start="1/1/2018", end="1/10/2018", periods=9))
            .append(pd.Series([None]))
            .astype("category"),
            id="series_val20",
        ),
        pytest.param(
            pd.Series(pd.timedelta_range(start="1 day", periods=9))
            .append(pd.Series([None]))
            .astype(pd.CategoricalDtype(ordered=True)),
            id="series_val21",
        ),
    ]
)
def series_val(request):
    return request.param


def test_series_fillna_series_val(series_val):
    def impl(S):
        val = S.iat[0]
        return S.fillna(val)

    if isinstance(series_val.iat[0], list):
        message = '"value" parameter cannot be a list'
        with pytest.raises(BodoError, match=message):
            bodo.jit(impl)(series_val)
    else:
        # TODO: Set dist_test=True once distributed getitem is supported
        # for Nullable and Categorical
        check_func(impl, (series_val,), dist_test=False, check_dtype=False)


def series_replace_impl(series, to_replace, value):
    return series.replace(to_replace, value)


def test_replace_series_val(series_val):
    """Run series.replace on the types in the series_val fixture. Catch
    expected failures from lack of coverage.
    """
    series = series_val.dropna()
    to_replace = series.iat[0]
    value = series.iat[1]

    message = ""
    if any(
        isinstance(x, (datetime.date, pd.Timedelta, pd.Timestamp))
        for x in [to_replace, value]
    ):
        # TODO: [BE-469]
        message = "Not supported for types"
    elif any(isinstance(x, pd.Categorical) for x in [to_replace, value]) or any(
        isinstance(x, list) for x in series
    ):
        message = "only support with Scalar"

    if message:
        with pytest.raises(BodoError, match=message):
            bodo.jit(series_replace_impl)(series, to_replace, value)
    else:
        check_func(series_replace_impl, (series, to_replace, value))


def test_series_replace_bitwidth(memory_leak_check):
    """Checks that series.replace succeeds on integers with
    different bitwidths."""

    def impl(S):
        return S.replace(np.int8(3), np.int64(24))

    S = pd.Series([1, 2, 3, 4, 5] * 4, dtype=np.int16)
    # Bodo dtype won't match because pandas casts everything to int64
    check_func(impl, (S,), check_dtype=False)


def test_series_float_literal(memory_leak_check):
    """Checks that series.replace with an integer and a float
    that can never be equal will return a copy."""
    # Tests for [BE-468]
    def impl(S):
        return S.replace(np.inf, np.nan)

    S = pd.Series([1, 2, 3, 4, 5] * 4, dtype=np.int16)
    check_func(impl, (S,))


def test_series_str_literal(memory_leak_check):
    """Checks that series.replace works with str literals"""

    def impl(S):
        return S.replace("a", "A")

    S = pd.Series(["a", "BB", "32e", "Ewrew"] * 4)
    check_func(impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "series_replace",
    [
        # Int
        pytest.param(
            SeriesReplace(series=pd.Series([1, 8, 4, 11, -3]), to_replace=1, value=10),
        ),
        # Float
        pytest.param(
            SeriesReplace(
                series=pd.Series([1.1, 8.2, 4.3, 11.4, -3.5]),
                to_replace=11.4,
                value=4.11,
            ),
        ),
        # String
        pytest.param(
            SeriesReplace(
                series=pd.Series(["A", "B", "CG", "ACDE"] * 4),
                to_replace="A",
                value="B",
            ),
        ),
        # Bool
        pytest.param(
            SeriesReplace(
                series=pd.Series([False, True, True, False, False]),
                to_replace=True,
                value=False,
            ),
        ),
        # List of strings
        pytest.param(
            SeriesReplace(
                pd.Series(["abc", "def"] * 4),
                to_replace=["abc"],
                value=["ghi"],
            ),
        ),
        # pd.Categorical pass
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=5,
                value=15,
            ),
        ),
        # to_replace=dictionary success
        pytest.param(
            SeriesReplace(
                pd.Series([1, 2, 3, 4] * 4),
                to_replace={1: 10, 2: 20, 3: 30},
                value=None,
            ),
        ),
    ],
)
def test_replace_types_supported(series_replace):
    """Run series.replace on particular types that all pass."""
    series = series_replace.series
    to_replace = series_replace.to_replace
    value = series_replace.value
    check_func(series_replace_impl, (series, to_replace, value))


@pytest.mark.slow
@pytest.mark.parametrize(
    "series_replace",
    [
        # Timestamp
        pytest.param(
            SeriesReplace(
                series=pd.Series(
                    pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)
                ),
                to_replace=pd.Timestamp("2018-04-24 00:00:00"),
                value=pd.Timestamp("2020-01-01 01:01:01"),
            ),
        ),
        # List of list of ints
        pytest.param(
            SeriesReplace(
                pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]]),
                to_replace=[3],
                value=[1],
            ),
        ),
        # pd.Categorical expected fail to_replace
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=pd.Categorical(5),
                value=pd.Categorical(15),
            ),
        ),
        # pd.Categorical expected fail value
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=5,
                value=pd.Categorical(15),
            ),
        ),
    ],
)
def test_replace_types_unsupported(series_replace):
    """Run series.replace on particular types that all fail."""
    series = series_replace.series
    to_replace = series_replace.to_replace
    value = series_replace.value

    if any(isinstance(x, pd.Timestamp) for x in [to_replace, value]):
        message = "Not supported for types"
    elif any(isinstance(x, (pd.Categorical)) for x in [to_replace, value]) or any(
        isinstance(x, list) for x in series
    ):
        message = "only support with Scalar"
    elif (
        (isinstance(to_replace, int) and isinstance(value, float))
        or isinstance(to_replace, bool)
        and (isinstance(value, (int, float)) and not isinstance(value, bool))
    ):
        message = "cannot replace type"
    elif series.dtype is np.dtype("int64") and isinstance(to_replace, float):
        message = "'to_replace' type must match series type"

    with pytest.raises(BodoError, match=message):
        bodo.jit(series_replace_impl)(series, to_replace, value)


def test_replace_float_int_scalar_scalar():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = 1
    value = 2.0
    check_func(series_replace_impl, (series, to_replace, value))


def test_replace_float_int_list_scalar():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = [1, 3, 6]
    value = 4.0
    check_func(series_replace_impl, (series, to_replace, value))


def test_replace_float_int_list_list():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = [1, 3]
    value = [2.0, 4.0]
    check_func(series_replace_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_string_int():
    series = pd.Series(["AZ", "BY", "CX"] * 4)
    to_replace = 1
    value = "DW"
    check_func(series_replace_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_inf_nan():
    series = pd.Series([0, 1, 2, 3] * 4)
    to_replace = [np.inf, -np.inf]
    value = np.nan
    check_func(series_replace_impl, (series, to_replace, value))


def test_series_concat(series_val, memory_leak_check):
    """test of concatenation of series.
    We convert to dataframe in order to reset the index.
    """

    def f(S1, S2):
        return pd.concat([S1, S2])

    S1 = series_val.copy()
    S2 = series_val.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"A": S2.values})
    if isinstance(series_val.values[0], list):
        check_func(
            f,
            (df1, df2),
            sort_output=True,
            reset_index=True,
            convert_columns_to_pandas=True,
        )
    else:
        check_func(f, (df1, df2), sort_output=True, reset_index=True)


def test_series_between(memory_leak_check):
    def impl_inclusive(S):
        return S.between(1, 4)

    def impl(S):
        return S.between(1, 4, inclusive=False)

    S = pd.Series([2, 0, 4, 8, np.nan])
    check_func(impl, (S,))
    check_func(impl_inclusive, (S,))


def test_datetime_series_between(memory_leak_check):
    def impl(S):
        lower = pd.Timestamp(year=2020, month=10, day=5)
        upper = pd.Timestamp(year=2020, month=10, day=20)
        return S.between(lower, upper)

    datatime_arr = [datetime.datetime(year=2020, month=10, day=x) for x in range(1, 32)]
    S = pd.Series(datatime_arr)
    check_func(impl, (S,))

    datatime_arr_nan = [datetime.datetime(year=2020, month=10, day=1), np.nan]
    S_with_nan = pd.Series(datatime_arr)
    check_func(impl, (S_with_nan,))


def test_series_concat_categorical(memory_leak_check):
    """test of concatenation of categorical series.
    TODO: refactor concat tests
    """

    def f(S1, S2):
        return pd.concat([S1, S2])

    S = pd.Series(["AA", "BB", "", "AA", None, "AA"], dtype="category")
    check_func(f, (S, S), sort_output=True, reset_index=True)


#   The code that we want to have.
#    bodo_f = bodo.jit(f)
#    pd.testing.assert_series_equal(
#        bodo_f(S1, S2), f(S1, S2), check_dtype=False, check_index_type=False
#    )
#    check_func(f, (S1, S2), reset_index=True)


# TODO: readd memory leak check when PDCategorical type constant lowering
# no longer leaks memory
def test_dataframe_concat(series_val):
    """This is actually a dataframe test that adds empty
    column when missing
    """
    # Pandas converts Integer arrays to int object arrays when adding an all NaN
    # chunk, which we cannot handle in our parallel testing.
    if isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def f(df1, df2):
        return pd.concat([df1, df2])

    S1 = series_val.copy()
    S2 = series_val.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"B": S2.values})
    # Categorical data seems to revert to object arrays when concat in Python
    # As a result we will compute the output directly and do the comparison
    # We use the actual series_val dtype to capture ordered info.
    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # Pandas 1.2.2 causes the conversion to obj to alter the representation
        # on dt64 and td64
        py_output = f(df1, df2)
        if series_val.dtype.categories.dtype in [
            np.dtype("datetime64[ns]"),
            np.dtype("timedelta64[ns]"),
        ]:
            py_output = py_output.astype(series_val.dtype.categories.dtype)
        py_output = py_output.astype(series_val.dtype)
    else:
        py_output = None
    if isinstance(series_val.values[0], list):
        check_func(
            f,
            (df1, df2),
            sort_output=True,
            reset_index=True,
            convert_columns_to_pandas=True,
            py_output=py_output,
        )
    else:
        check_func(
            f, (df1, df2), sort_output=True, reset_index=True, py_output=py_output
        )


# TODO: readd memory leak check when PDCategorical type constant lowering
# no longer leaks memory
@pytest.mark.slow
def test_dataframe_concat_cat_dynamic():
    """This is actually a dataframe test that adds empty
    column when missing if categories change dynamically
    """

    def f(df1, df2, to_replace, value):
        df1.replace(to_replace, value)
        df2.replace(to_replace, value)
        return pd.concat([df1, df2])

    S = pd.Series(["AA", "BB", "", "AA", None, "AA"], dtype="category")
    S1 = S.copy()
    S2 = S.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"B": S2.values})
    to_replace = "AA"
    value = "AAAA"
    # Categorical data seems to revert to object arrays when concat in Python
    # As a result we will compute the output directly and do the comparison
    py_output = f(df1, df2, to_replace, value).astype("category")
    check_func(
        f,
        (df1, df2, to_replace, value),
        sort_output=True,
        reset_index=True,
        py_output=py_output,
    )


# TODO (Nick): Readd the memory leak check when constant lower leak is fixed.
# Categorical input has a constant lowering step so it leaks memory.
def test_series_concat_convert_to_nullable():
    """make sure numpy integer/bool arrays are converted to nullable integer arrays in
    concatenation properly
    """

    def impl1(S1, S2):
        return pd.concat([S1, S2])

    # Integer case
    S1 = pd.Series([3, 2, 1, -4, None, 11, 21, 31, None] * 2, dtype="Int64")
    S2 = pd.Series(np.arange(11) * 2, dtype="int32")
    check_func(impl1, (S1, S2), sort_output=True, reset_index=True)

    # calling pd.Series inside the function to force values to be Numpy bool since Bodo
    # assumes nullable for bool input arrays during unboxing by default
    def impl2(S1, S2):
        return pd.concat([S1, pd.Series(S2)])

    # Boolean case
    S1 = pd.Series(
        [True, False, False, True, None, False, False, True, None] * 2, dtype="boolean"
    )
    S2 = pd.Series([True, False, False, True, True, False, False], dtype="bool").values
    check_func(impl2, (S1, S2), sort_output=True, reset_index=True)


# TODO: timedelta, period, tuple, etc.
@pytest.fixture(
    params=[
        pytest.param(pd.Series([1, 8, 4, 11, -3]), marks=pytest.mark.slow),
        pd.Series([1.1, np.nan, 4.1, 1.4, -2.1]),
        pytest.param(
            pd.Series([1, 8, 4, 10, 3], dtype=np.uint8), marks=pytest.mark.slow
        ),
        pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32"),
        pytest.param(
            pd.Series([1, 8, 4, -1, 2], [3, 7, 9, 2, 1], name="AAC"),
            marks=pytest.mark.slow,
        ),
        pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
    ]
)
def numeric_series_val(request):
    return request.param


def test_series_notna(series_val, memory_leak_check):
    def test_impl(S):
        return S.notna()

    check_func(test_impl, (series_val,))


def test_series_notnull(series_val, memory_leak_check):
    def test_impl(S):
        return S.notnull()

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_box(series_val, memory_leak_check):
    # unbox and box
    def impl(S):
        return S

    check_func(impl, (series_val,))


@pytest.mark.slow
def test_series_index(series_val, memory_leak_check):
    def test_impl(S):
        return S.index

    check_func(test_impl, (series_val,))


def test_series_index_none(memory_leak_check):
    def test_impl():
        S = pd.Series([1, 4, 8])
        return S.index

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_series_values(series_val, memory_leak_check):
    def test_impl(S):
        return S.values

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_dtype(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.dtype

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_shape(series_val, memory_leak_check):
    def test_impl(S):
        return S.shape

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_ndim(series_val, memory_leak_check):
    def test_impl(S):
        return S.ndim

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_size(series_val, memory_leak_check):
    def test_impl(S):
        return S.size

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_T(series_val, memory_leak_check):
    def test_impl(S):
        return S.T

    check_func(test_impl, (series_val,))


def test_series_hasnans(series_val, memory_leak_check):
    def test_impl(S):
        return S.hasnans

    check_func(test_impl, (series_val,))


def test_series_empty(series_val, memory_leak_check):
    def test_impl(S):
        return S.empty

    check_func(test_impl, (series_val,))


def test_series_dtypes(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.dtypes

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_name(series_val, memory_leak_check):
    def test_impl(S):
        return S.name

    check_func(test_impl, (series_val,))


def test_series_astype_numeric(numeric_series_val, memory_leak_check):
    # datetime can't be converted to float
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(S):
        return S.astype(np.float64)

    check_func(test_impl, (numeric_series_val,))


# TODO: add memory_leak_check
def test_series_astype_str(series_val):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # XXX str(float) not consistent with Python yet
    if series_val.dtype == np.float64:
        return

    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    # XXX Pandas 1.2. Null value in categorical input gets
    # set as a null value and not "nan". It is unclear if
    # this is unique to certain types. np.nan is still
    # converted to "nan"
    if isinstance(series_val.dtype, pd.CategoricalDtype):
        return

    def test_impl(S):
        return S.astype(str)

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_int_astype_str_long(memory_leak_check):
    """test a Series with a lot of int/NA values to make sure string array capacity is
    adjusted properly.
    """

    def impl(S):
        return S.astype(str)

    S = pd.Series([11111, 22, None, -3222222, None, 445] * 1000, dtype="Int64")
    check_func(impl, (S,))


def test_series_astype_int_arr(numeric_series_val, memory_leak_check):
    # only integers can be converted safely
    if not pd.api.types.is_integer_dtype(numeric_series_val):
        return

    def test_impl(S):
        return S.astype("Int64")

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, np.nan, 3.0, 2.0], dtype="float32"),
        pytest.param(
            pd.Series([1.0, np.nan, 3.0, 2.0], dtype="float64"), marks=pytest.mark.slow
        ),
    ],
)
def test_series_astype_float_to_int_arr(S, memory_leak_check):
    """Test converting float data to nullable int array"""
    # TODO: support converting string to int

    def test_impl(S):
        return S.astype("Int64")

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(
            pd.Series([True, False, False, True, True]), marks=pytest.mark.slow
        ),
        pd.Series([True, False, False, np.nan, True]),
    ],
)
def test_series_astype_bool_arr(S, memory_leak_check):
    # TODO: int, Int

    def test_impl(S):
        return S.astype("float32")

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(["a", "b", "aa", "bb", "A", "a", "BB"]),
        pd.Series([1, 2, 41, 2, 1, 4, 2, 1, 1, 25, 5, 3]),
    ],
)
def test_series_drop_duplicates(S, memory_leak_check):
    def test_impl(S):
        return S.drop_duplicates()

    check_func(test_impl, (S,), sort_output=True)


@pytest.mark.parametrize(
    "S", [pd.Series(["BB", "C", "A", None, "A", "BBB", None, "C", "BB", "A"])]
)
# TODO: add memory_leak_check
def test_series_astype_cat(S):
    def test_impl(S, ctype):
        return S.astype(ctype)

    def test_impl2(S):
        return S.astype("category")

    # full dtype values
    ctype = pd.CategoricalDtype(S.dropna().unique())
    check_func(test_impl, (S, ctype))
    # partial dtype values to force NA
    new_cats = ctype.categories.tolist()[:-1]
    np.random.shuffle(new_cats)
    ctype = pd.CategoricalDtype(new_cats)
    check_func(test_impl, (S, ctype))
    # dtype not specified
    check_func(test_impl2, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_box(S, memory_leak_check):
    def test_impl(S):
        return S

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_comp(S, memory_leak_check):
    def test_impl(S):
        return S == "BB"

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(["BB", "C", "A", None, "A", "BBB", None, "C", "BB", "A"]).astype(
            "category"
        )
    ],
)
def test_series_cat_codes(S, memory_leak_check):
    def test_impl(S):
        return S.cat.codes

    check_func(test_impl, (S,))


def test_series_copy(series_val, memory_leak_check):
    # TODO: test deep/shallow cases properly
    def test_deep(S):
        return S.copy()

    def test_shallow(S):
        return S.copy(deep=False)

    check_func(test_deep, (series_val,))
    check_func(test_shallow, (series_val,))
    check_func(test_deep, (series_val.values,))


def test_series_to_list(series_val, memory_leak_check):
    """Test Series.to_list(): non-float NAs throw a runtime error since can't be
    represented in lists.
    """
    # TODO: [BE-498] Correctly convert nan
    def impl(S):
        return S.to_list()

    if series_val.hasnans and not isinstance(series_val.iat[0], np.floating):
        message = "Not supported for NA values"
        with pytest.raises(ValueError, match=message):
            bodo.jit(impl)(series_val)
    else:
        check_func(impl, (series_val,), only_seq=True)


def test_series_to_numpy(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.to_numpy()

    check_func(test_impl, (numeric_series_val,))


# TODO: add memory_leak_check (it leaks with Decimal array)
@pytest.mark.smoke
def test_series_iat_getitem(series_val):
    def test_impl(S):
        return S.iat[2]

    bodo_func = bodo.jit(test_impl)
    bodo_out = bodo_func(series_val)
    py_out = test_impl(series_val)
    _test_equal(bodo_out, py_out)
    # fix distributed
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_iat_getitem_datetime(memory_leak_check):
    """Test to check that series.iat properly converts
    unboxes datetime.date, dt64, and td64"""

    def test_impl(S):
        return S.iat[2]

    date_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17").date)
    datetime_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17"))
    timedelta_series = pd.Series(
        np.append(
            [datetime.timedelta(days=5, seconds=4, weeks=4)] * 2,
            [datetime.timedelta(seconds=202, hours=5)] * 2,
        )
    )
    # TODO: Ensure parallel implementation works
    check_func(test_impl, (date_series,), dist_test=False)
    check_func(test_impl, (datetime_series,), dist_test=False)
    check_func(test_impl, (timedelta_series,), dist_test=False)


@pytest.mark.smoke
def test_series_iat_setitem(series_val, memory_leak_check):

    val = series_val.iat[0]

    def test_impl(S, val):
        S.iat[2] = val
        return S

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        return

    # TODO: Test distributed implementation
    check_func(test_impl, (series_val, val), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_series_iat_setitem_datetime(memory_leak_check):
    """
    Test that series.iat supports datetime.date, datetime.datetime, and datetime.timedelta
    scalar values.
    """

    def test_impl(S, val):
        S.iat[2] = val
        return S

    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    val1 = datetime.datetime(2011, 11, 4, 0, 0)
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    val2 = datetime.date(2002, 12, 26)
    S3 = pd.Series(
        [
            datetime.timedelta(3, 3, 3),
            datetime.timedelta(2, 2, 2),
            datetime.timedelta(1, 1, 1),
            np.nan,
            datetime.timedelta(5, 5, 5),
        ]
    )
    val3 = datetime.timedelta(days=64, seconds=29156, microseconds=10)
    check_func(test_impl, (S1, val1), copy_input=True, dist_test=False)
    check_func(test_impl, (S2, val2), copy_input=True, dist_test=False)
    check_func(test_impl, (S3, val3), copy_input=True, dist_test=False)


@pytest.mark.smoke
def test_series_iloc_getitem_int(series_val):
    def test_impl(S):
        return S.iloc[2]

    bodo_func = bodo.jit(test_impl)
    bodo_out = bodo_func(series_val)
    py_out = test_impl(series_val)
    _test_equal(bodo_out, py_out)
    # fix distributed
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_iloc_getitem_datetime(memory_leak_check):
    """Test to check that series.iloc properly converts
    unboxes datetime.date, dt64, and td64"""

    def test_impl(S):
        return S.iloc[2]

    date_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17").date)
    datetime_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17"))
    timedelta_series = pd.Series(
        np.append(
            [datetime.timedelta(days=5, seconds=4, weeks=4)] * 2,
            [datetime.timedelta(seconds=202, hours=5)] * 2,
        )
    )
    # TODO: Ensure parallel implementation works
    check_func(test_impl, (date_series,), dist_test=False)
    check_func(test_impl, (datetime_series,), dist_test=False)
    check_func(test_impl, (timedelta_series,), dist_test=False)


def test_series_iloc_getitem_slice(series_val, memory_leak_check):
    def test_impl(S):
        return S.iloc[1:4]

    # fix distributed
    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_iloc_getitem_array_int(series_val, memory_leak_check):
    def test_impl(S):
        return S.iloc[[1, 3]]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_iloc_getitem_array_bool(series_val, memory_leak_check):
    """Tests that getitem with Series.iloc works with a Boolean List index"""

    def test_impl(S):
        return S.iloc[[True, True, False, True, False]]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_loc_getitem_array_bool(series_val, memory_leak_check):
    """Tests that getitem with Series.loc works with a Boolean List index"""

    def test_impl(S):
        return S.loc[[True, True, False, True, False]]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


@pytest.mark.slow
def test_series_loc_getitem_int_range(memory_leak_check):
    def test_impl(S):
        return S.loc[2]

    S = pd.Series(
        data=[1, 12, 1421, -241, 4214], index=pd.RangeIndex(start=0, stop=5, step=1)
    )
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_loc_setitem_array_bool(series_val, memory_leak_check):
    """Tests that setitem with Series.loc works with a Boolean List index"""

    def test_impl(S, val):
        S.loc[[True, True, False, True, False]] = val
        return S

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    err_msg = "Series setitem not supported for Series with immutable array type .*"

    # Test with an array
    val = series_val.iloc[0:3].values.copy()
    if isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )
    # Test with a list
    val = list(val)
    if isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )
    # Test with a scalar
    val = val[0]
    if isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )


def test_series_diff(numeric_series_val, memory_leak_check):
    """test Series.diff()"""
    # # Pandas as of 1.2.2 is buggy for uint8 and produces wrong results
    if numeric_series_val.dtype == np.uint8:
        return

    def impl(S):
        return S.diff()

    # TODO: Support nullable arrays
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        with pytest.raises(
            BodoError, match="Series.diff.* column input type .* not supported"
        ):
            bodo.jit(impl)(numeric_series_val)
    else:
        check_func(impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_iloc_setitem_datetime_scalar(memory_leak_check):
    """
    Test that series.iloc supports datetime.date, datetime.datetime, and datetime.timedelta
    scalar values.
    """

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    val1 = datetime.datetime(2011, 11, 4, 0, 0)
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    val2 = datetime.date(2002, 12, 26)
    S3 = pd.Series(
        [
            datetime.timedelta(3, 3, 3),
            datetime.timedelta(2, 2, 2),
            datetime.timedelta(1, 1, 1),
            np.nan,
            datetime.timedelta(5, 5, 5),
        ]
    )
    val3 = datetime.timedelta(days=64, seconds=29156, microseconds=10)

    arr_idx = np.array([True, True, False, True, False])

    for idx in (1, slice(1, 4), arr_idx, list(arr_idx)):
        check_func(test_impl, (S1, idx, val1), copy_input=True, dist_test=False)
        check_func(test_impl, (S2, idx, val2), copy_input=True, dist_test=False)
        check_func(test_impl, (S3, idx, val3), copy_input=True, dist_test=False)


@pytest.mark.smoke
def test_series_iloc_setitem_int(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc with int idx.
    """

    val = series_val.iat[0]

    def test_impl(S, val):
        S.iloc[2] = val
        # print(S) TODO: fix crash
        return S

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        return

    # TODO: Test distributed implementation
    check_func(test_impl, (series_val, val), copy_input=True, dist_test=False)


def test_series_iloc_setitem_list_bool(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc and Series with bool arr/list idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    # test Series.values since it is used instead of iloc sometimes
    def test_impl2(S, idx, val):
        S.values[idx] = val
        return S

    idx = np.array([True, True, False, True, False] + [False] * (len(series_val) - 5))
    # value is array
    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, idx, val)
        check_func(test_impl2, (series_val, idx, val), copy_input=True, dist_test=False)

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="only setitem with scalar index is currently supported for list arrays",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    else:
        # TODO: Test distributed implementation
        check_func(test_impl, (series_val, idx, val), copy_input=True, dist_test=False)
        check_func(test_impl2, (series_val, idx, val), copy_input=True, dist_test=False)

    val = series_val.dropna().iloc[0:3].to_list()

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="StringArray setitem with index .* and value .* not supported.",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="only setitem with scalar index is currently supported for list arrays",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    else:
        # TODO: Test distributed implementation
        # Pandas promotes to float64. This is most likely a bug and Bodo
        # keeps the same, smaller type.
        if series_val.dtype == np.uint8:
            check_dtype = False
        else:
            check_dtype = True
        check_func(
            test_impl,
            (series_val, idx, val),
            copy_input=True,
            dist_test=False,
            check_dtype=check_dtype,
        )
        # setitem dt64/td64 array with list Timestamp/Timedelta values not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, idx, val),
                copy_input=True,
                dist_test=False,
                check_dtype=check_dtype,
            )


def test_series_iloc_setitem_scalar(series_val, memory_leak_check):
    """
    Tests that series.iloc setitem with array/index/slice
    properly supports a Scalar RHS
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support setitem array idx, scalar value for Categorical arrays
        return

    val = series_val.iat[0]

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    arr_idx = np.array(
        [True, True, False, True, False] + [False] * (len(series_val) - 5)
    )

    for idx in (slice(1, 4), arr_idx, list(arr_idx)):
        # string setitem not supported yet
        if isinstance(series_val.iat[0], str) and not isinstance(
            series_val.dtype, pd.CategoricalDtype
        ):
            with pytest.raises(
                BodoError, match="Series string setitem not supported yet"
            ):
                bodo.jit(test_impl)(series_val, idx, val)
            return

        # not supported for list(string) and array(item)
        elif isinstance(series_val.values[0], list):
            with pytest.raises(
                BodoError,
                match="Series setitem not supported for Series with immutable array type .*",
            ):
                bodo.jit(test_impl)(series_val, idx, val)
            return
        else:
            # TODO: Test distributed implementation
            check_func(
                test_impl, (series_val, idx, val), copy_input=True, dist_test=False
            )


def test_series_iloc_setitem_slice(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc and Series.values with slice idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, val):
        S.iloc[1:4] = val
        return S

    # test Series.values since it is used instead of iloc sometimes
    def test_impl2(S, val):
        S.values[1:4] = val
        return S

    # value is array
    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        check_func(
            test_impl2,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )

    # value is a list

    val = series_val.dropna().iloc[0:3].to_list()

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list Timestamp/Timedelta values not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_iloc_setitem_list_int(series_val, idx, memory_leak_check):
    """
    Test setitem for Series.iloc and Series.values with list/array
    of ints idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, val, idx):
        S.iloc[idx] = val
        return S

    # test Series.values which may be used as alternative to Series.iloc for setitem
    # tests underlying array setitem essentially
    def test_impl2(S, val, idx):
        S.values[idx] = val
        return S

    # value is an array
    val = series_val.iloc[0:2].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val, idx),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list(int) idx not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val, idx),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )

    val = series_val.dropna().iloc[0:2].to_list()
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val, idx),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list(int) idx not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val, idx),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )


####### getitem tests ###############


# TODO: add memory_leak_check
@pytest.mark.smoke
def test_series_getitem_int(series_val):
    # timedelta setitem not supported yet
    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    def test_impl(S):
        return S[2]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val)
    else:
        bodo_out = bodo_func(series_val)
        py_out = test_impl(series_val)
        _test_equal(bodo_out, py_out)


def test_series_getitem_slice(series_val, memory_leak_check):
    def test_impl(S):
        return S[1:4]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_getitem_list_int(series_val, idx, memory_leak_check):
    def test_impl(S, idx):
        return S[idx]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, idx)
    else:
        pd.testing.assert_series_equal(
            bodo_func(series_val, idx), test_impl(series_val, idx), check_dtype=False
        )


def test_series_getitem_array_bool(series_val, memory_leak_check):
    def test_impl(S):
        return S[[True, True, False, True, False]]

    def test_impl2(S, cond):
        # using .values to test boolean_array
        return S[cond.values]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )
    cond = pd.Series([True, True, False, True, False])
    bodo_func = bodo.jit(test_impl2)
    pd.testing.assert_series_equal(
        bodo_func(series_val, cond), test_impl2(series_val, cond), check_dtype=False
    )


############### setitem tests #################


@pytest.mark.smoke
def test_series_setitem_int(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return
    val = series_val.iat[0]

    def test_impl(S, val):
        S[2] = val
        return S

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, val)
    else:
        check_func(test_impl, (series_val, val), dist_test=False, copy_input=True)


def test_series_setitem_slice(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return

    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    def test_impl(S, val):
        S[1:4] = val
        return S

    check_func(test_impl, (series_val, val), dist_test=False, copy_input=True)


@pytest.mark.parametrize("idx", [[1, 4], np.array([1, 4]), pd.Series([1, 4])])
@pytest.mark.parametrize("list_val_arg", [True, False])
def test_series_setitem_list_int(series_val, idx, list_val_arg, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return
    val = series_val.iloc[0:2].values.copy()  # values to avoid alignment
    if list_val_arg:
        val = list(val)

    def test_impl(S, val, idx):
        S[idx] = val
        return S

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, val, idx)
    else:
        # Pandas coerces Series type to set values, so avoid low precision
        # TODO: warn or error?
        if list_val_arg and (
            series_val.dtype
            in (
                np.int8,
                np.uint8,
                np.int16,
                np.uint16,
                np.int32,
                np.uint32,
            )
        ):
            return
        check_func(test_impl, (series_val, val, idx), dist_test=False, copy_input=True)


############################ Series.loc indexing ##########################


def test_series_loc_setitem_bool(memory_leak_check):
    """test Series.loc[bool_arr] setitem"""

    def impl(S, idx, val):
        S.loc[idx] = val
        return S

    S = pd.Series(["A", "AB", None, "ABC", "AB", "D", ""])
    check_func(impl, (S, S == "AB", "ABC"), copy_input=True)
    S = pd.Series([4.8, 1.1, 2.2, np.nan, 3.3, 4.4])
    check_func(impl, (S, S < 4.0, 1.8), copy_input=True)
    S = pd.Series([4, 1, 2, None, 3, 4], dtype="Int64")
    check_func(impl, (S, S < 4, -2), copy_input=True)


############################ binary ops #############################


def test_series_operations(memory_leak_check, is_slow_run):
    def f_rpow(S1, S2):
        return S1.rpow(S2)

    def f_rsub(S1, S2):
        return S1.rsub(S2)

    def f_rmul(S1, S2):
        return S1.rmul(S2)

    def f_radd(S1, S2):
        return S1.radd(S2)

    def f_rdiv(S1, S2):
        return S1.rdiv(S2)

    def f_truediv(S1, S2):
        return S1.truediv(S2)

    def f_rtruediv(S1, S2):
        return S1.rtruediv(S2)

    def f_floordiv(S1, S2):
        return S1.floordiv(S2)

    def f_rfloordiv(S1, S2):
        return S1.rfloordiv(S2)

    def f_mod(S1, S2):
        return S1.mod(S2)

    def f_rmod(S1, S2):
        return S1.rmod(S2)

    S1 = pd.Series([2, 3, 4])
    S2 = pd.Series([6, 7, 8])
    check_func(f_rpow, (S1, S2))
    if not is_slow_run:
        return
    check_func(f_rsub, (S1, S2))
    check_func(f_rmul, (S1, S2))
    check_func(f_radd, (S1, S2))
    check_func(f_rdiv, (S1, S2))
    check_func(f_truediv, (S1, S2))
    check_func(f_rtruediv, (S1, S2))
    check_func(f_floordiv, (S1, S2))
    check_func(f_rfloordiv, (S1, S2))
    check_func(f_mod, (S1, S2))
    check_func(f_rmod, (S1, S2))


def test_series_add(memory_leak_check):
    def test_impl(s):
        return s.add(s, None)

    s = pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32")
    check_func(test_impl, (s,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "lt",
        "gt",
        "le",
        "ge",
        "ne",
        "eq",
    ],
)
@pytest.mark.parametrize("fill", [None, True])
def test_series_explicit_binary_op(numeric_series_val, op, fill, memory_leak_check):
    # dt64 not supported here
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return
    # XXX ne operator is buggy in Pandas and doesn't set NaNs in output
    # when both inputs are NaNs
    if op == "ne" and numeric_series_val.hasnans:
        return
    # Numba returns float32 for truediv but Numpy returns float64
    if op == "truediv" and numeric_series_val.dtype == np.uint8:
        return
    # Pandas 1.2.0 converts floordiv and truediv to Float64 when input is
    # nullable integer
    # TODO: Support FloatingArray
    if op in ("truediv", "floordiv") and isinstance(
        numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype
    ):
        check_dtype = False
    else:
        check_dtype = True
    if op == "pow" and numeric_series_val.dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        # negative numbers not supported in integer pow
        numeric_series_val = numeric_series_val.abs()

    func_text = "def test_impl(S, other, fill_val):\n"
    func_text += "  return S.{}(other, fill_value=fill_val)\n".format(op)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    if fill is not None:
        fill = numeric_series_val.iloc[0]
    check_func(
        test_impl,
        (numeric_series_val, numeric_series_val, fill),
        check_dtype=check_dtype,
    )


@pytest.mark.parametrize("fill", [None, 1.6])
def test_series_explicit_binary_op_nan(fill, memory_leak_check):
    # test nan conditions (both nan, left nan, right nan)
    def test_impl(S, other, fill_val):
        return S.add(other, fill_value=fill_val)

    L1 = pd.Series([1.0, np.nan, 2.3, np.nan])
    L2 = pd.Series([1.0, np.nan, np.nan, 1.1], name="ABC")
    check_func(test_impl, (L1, L2, fill))


def test_series_explicit_binary_op_nullable_int_bool(memory_leak_check):
    """Make comparison operation on nullable int array returns a BooleanArray
    (Pandas 1.0)
    """

    def test_impl(S, other):
        return S.lt(other)

    S1 = pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32")
    S2 = pd.Series([1, -1, 3, 11, 7], [3, 7, 9, 2, 1], dtype="Int32")
    check_func(test_impl, (S1, S2))
    check_func(test_impl, (S1, 5))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_binary_ops)
def test_series_binary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, 2))
    check_func(test_impl, (2, S))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_inplace_binary_ops)
def test_series_inplace_binary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  S {} other\n".format(op_str)
    func_text += "  return S\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(S.copy(), S.copy()), test_impl(S.copy(), S.copy())
    )
    pd.testing.assert_series_equal(bodo_func(S.copy(), 2), test_impl(S.copy(), 2))
    # XXX: A**=S doesn't work in Pandas for some reason
    if op != operator.ipow:
        np.testing.assert_array_equal(
            bodo_func(S.values.copy(), S.copy()), test_impl(S.values.copy(), S.copy())
        )


@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_unary_ops)
def test_series_unary_op(op, memory_leak_check):
    # TODO: fix operator.pos
    if op == operator.pos:
        return

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S):\n"
    func_text += "  return {} S\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


def test_series_ufunc(memory_leak_check):
    ufunc = np.negative

    def test_impl(S):
        return ufunc(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    # avoiding isnat since only supported for datetime/timedelta
    "ufunc",
    [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 1 and f != np.isnat],
)
def test_series_unary_ufunc(ufunc, memory_leak_check):
    def test_impl(S):
        return ufunc(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


def test_series_unary_ufunc_np_call(memory_leak_check):
    # a ufunc called explicitly, since the above test sets module name as
    # 'ufunc' instead of 'numpy'
    def test_impl(S):
        return np.negative(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ufunc", [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 2 and f.nout == 1]
)
def test_series_binary_ufunc(ufunc, memory_leak_check):
    def test_impl(S1, S2):
        return ufunc(S1, S2)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    A = np.array([1, 3, 7, 11])
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, A))
    check_func(test_impl, (A, S))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        # dtype="boolean" makes these nullable Boolean arrays
        pd.Series(
            [True, False, False, True, True, True, False, False], dtype="boolean"
        ),
        pd.Series(
            [True, False, np.nan, True, True, False, True, False], dtype="boolean"
        ),
    ],
)
def test_series_bool_cmp_op(S, op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))
    check_func(test_impl, (S, True))
    check_func(test_impl, (True, S))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        pd.Series(
            [True, False, False, True, True, True, False, False], dtype="boolean"
        ),
        pd.Series(
            [True, False, np.nan, True, True, False, True, False], dtype="boolean"
        ),
    ],
)
def test_series_bool_vals_cmp_op(S, op, memory_leak_check):
    """The comparison operation for missing value can be somewhat complicated:
    ---In Python/C++/Fortran/Javascript, the comparison NaN == NaN returns false.
    ---In Pandas the nan or None or missing values are all considered in the same
    way and the result is a missing value in output.
         -
    The following is badly treated in pandas 1.1.0 currently:
    pd.Series([True, False, np.nan, True, True, False, True, False]) .
    Instead of [True, True, <NA>, True, ..., True], pandas returns
    [True, True, False, True, ..., True].
    Adding the ', dtype="boolean"' resolves the issue.
    """
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S.values {} other.values\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))


def test_series_str_add(memory_leak_check):
    """Test addition for string Series"""

    def test_impl(S, other):
        return S + other

    S = pd.Series(["AA", "D", None, "녁은", "AA🐍"] * 2, [3, 5, 0, 7, -1] * 2)
    S2 = pd.Series(["C🐍", "녁은", "D", None, "AA"] * 2, [3, 5, 0, 7, -1] * 2)
    check_func(test_impl, (S, S2))
    check_func(test_impl, (S, "CC"))
    check_func(test_impl, ("CC", S))


def test_series_str_cmp(memory_leak_check):
    """Test basic comparison for string Series (#1381)"""

    def test_impl(S):
        return S == "A"

    S = pd.Series(pd.array(["AA", "D", None, "녁은", "AA🐍"] * 2), [3, 5, 0, 7, -1] * 2)
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S1,S2,fill,raises",
    [
        # float64 input
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0]),
            None,
            False,
        ),
        # index, name
        (
            pd.Series([1.0, 2.0, 3.0, 4.0], [3, 5, 0, 7], name="ABC"),
            pd.Series([6.0, 21.0, 3.6, 5.0], [3, 5, 0, 7]),
            None,
            False,
        ),
        # combine float64/32
        (
            pd.Series([1, 4, 5], dtype="float64"),
            pd.Series([3, 1, 2], dtype="float32"),
            None,
            False,
        ),
        # raise on size mismatch
        (pd.Series([1, 2, 3]), pd.Series([6.0, 21.0, 3.0, 5.0]), None, True),
        (pd.Series([6.0, 21.0, 3.0, 5.0]), pd.Series([1, 2, 3]), None, True),
        # integer case
        (pd.Series([1, 2, 3, 4, 5]), pd.Series([6, 21, 3, 5]), 16, False),
        # different types
        (
            pd.Series([6.1, 21.2, 3.3, 5.4, 6.7]),
            pd.Series([1, 2, 3, 4, 5]),
            None,
            False,
        ),
        # same len integer
        (pd.Series([1, 2, 3, 4, 5]), pd.Series([6, 21, 17, -5, 4]), None, False),
        # same len
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0, 0.0]),
            None,
            False,
        ),
        # fill value
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0]),
            1237.56,
            False,
        ),
        # fill value same len
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0, 0.0]),
            1237.56,
            False,
        ),
    ],
)
def test_series_combine(S1, S2, fill, raises, memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(S2, lambda a, b: 2 * a + b, fill_val)

    bodo_func = bodo.jit(test_impl)
    if raises:
        with pytest.raises(AssertionError):
            bodo_func(S1, S2, fill)
    else:
        # TODO: fix 1D_Var chunk size mismatch on inputs with different sizes
        pd.testing.assert_series_equal(bodo_func(S1, S2, fill), test_impl(S1, S2, fill))


@pytest.mark.slow
def test_series_combine_kws(memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b, fill_value=fill_val)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    fill = 1237.56
    check_func(test_impl, (S1, S2, fill))


@pytest.mark.slow
def test_series_combine_kws_int(memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b, fill_value=fill_val)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    fill = 2
    check_func(test_impl, (S1, S2, fill))


def test_series_combine_no_fill(memory_leak_check):
    def test_impl(S1, S2):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b)

    S1 = pd.Series([1, 2, 3, 4, 5])
    S2 = pd.Series([6, 21, 3, 5, 0])
    check_func(test_impl, (S1, S2))


def g(a, b):
    return 2 * a + b


@pytest.mark.slow
def test_series_combine_global_func(memory_leak_check):
    def test_impl(S1, S2):
        return S1.combine(other=S2, func=g)

    S1 = pd.Series([1, 2, 3, 4, 5])
    S2 = pd.Series([6, 21, 3, 5, 0])
    check_func(test_impl, (S1, S2))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_apply(S, memory_leak_check):
    def test_impl(S):
        return S.apply(lambda a: 2 * a)

    check_func(test_impl, (S,))


def test_series_apply_kw(memory_leak_check):
    def test_impl(S):
        return S.apply(func=lambda a: 2 * a)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_pipe():
    """
    Test Series.pipe()
    """

    def impl1(S):
        return S.pipe(lambda S: S.sum())

    # test *args, **kwargs
    def impl2(S, a, b):
        return S.pipe(lambda S, a, b: S.sum() + a + b, a, b=b)

    S = pd.Series([1, 2, 3, 4, 5, 6])
    check_func(impl1, (S,))
    check_func(impl2, (S, 1, 2))


def test_series_heter_constructor(memory_leak_check):
    """
    test creating Series with heterogeneous values
    """

    def impl1():
        return pd.Series([1, "A"], ["B", "C"])

    check_func(impl1, (), dist_test=False)


def test_series_apply_df_output(memory_leak_check):
    """test Series.apply() with dataframe output"""

    def impl1(S):
        return S.apply(lambda a: pd.Series([a, "AA"]))

    def impl2(S):
        def g(a):
            # TODO: support assert in UDFs properly
            # assert a > 0.0
            if a > 3:
                return pd.Series([a, 2 * a], ["A", "B"])
            return pd.Series([a, 3 * a], ["A", "B"])

        return S.apply(g)

    def impl3(S):
        return S.apply(lambda a: pd.Series((str(a), str(a + 1.2))))

    def impl4(S):
        return S.apply(lambda a: pd.Series((a, a + 1.2)))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(impl1, (S,))
    check_func(impl2, (S,))
    _check_IR_no_const_arr(impl3, (S,))
    _check_IR_no_const_arr(impl4, (S,))


def _check_IR_no_const_arr(test_impl, args):
    """makes sure there is no const array call left in the IR after optimization"""
    bodo_func = numba.njit(pipeline_class=SeriesOptTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    # make sure there is no const call in IR
    for block in fir.blocks.values():
        for stmt in block.body:
            assert not (
                is_call_assign(stmt)
                and (
                    guard(find_callname, fir, stmt.value)
                    in (
                        ("str_arr_from_sequence", "bodo.libs.str_arr_ext"),
                        ("asarray", "numpy"),
                    )
                )
            )


@pytest.mark.slow
def test_series_apply_extra_arg(memory_leak_check):
    def test_impl(S, D):
        return S.apply(lambda a, d: a not in d, args=(D,))

    d = ("A", "B")
    S = pd.Series(["A", "C", "FF", "AA", "CC", "B", "DD", "ABC", "A"])
    check_func(test_impl, (S, d))


@pytest.mark.slow
def test_series_apply_kwargs(memory_leak_check):
    def test_impl(S, D):
        return S.apply(lambda a, d: a not in d, d=D)

    d = ("A", "B")
    S = pd.Series(["A", "C", "FF", "AA", "CC", "B", "DD", "ABC", "A"])
    check_func(test_impl, (S, d))


def test_series_apply_args_and_kwargs(memory_leak_check):
    def test_impl(S, b, d):
        return S.apply(lambda x, c=1, a=2: x == a + c, a=b, args=(d,))

    n = 121
    S = pd.Series(np.arange(n))
    check_func(test_impl, (S, 3, 2))


@pytest.mark.slow
def test_series_apply_supported_types(series_val, memory_leak_check):
    """ Test Series.apply with all Bodo supported Types """

    def test_impl(S, val):
        return S.apply(lambda a, val: a if a != val else None, val=val)

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-130] support apply for pd.Categorical
        return
    # Bodo gives different value for NaN from Pandas.(e.g. NaN vs. False for boolean)
    S = series_val.dropna()
    # Increase data size to pass testing with 3 ranks.
    S = S.repeat(3)
    # TODO: Fails with array of lists. Test when equality operation support list comparison.
    if isinstance(S.values[0], list):
        return
    val = S.iloc[0]
    # Disable check_dtype since Pandas return uint8 as int64
    check_func(test_impl, (S, val), check_dtype=False)


@pytest.mark.slow
def test_series_apply_args(memory_leak_check):
    """ Test Series.apply with unsupported and wrong arguments """

    def test_convert_dtype_false(S):
        return S.apply(lambda a: a, convert_dtype=False)

    def test_convert_dtype_true(S):
        return S.apply(lambda a: a, convert_dtype=True)

    def test_np_func(S):
        return S.apply(np.abs)

    def test_wrong_func(S):
        return S.apply("XX")

    def test_wrong_arg(S):
        return S.apply(lambda x: x, axis=1)

    S = pd.Series([2, 1, 3])
    with pytest.raises(
        BodoError, match="Series.apply.* only supports default value True"
    ):
        bodo.jit(test_convert_dtype_false)(S)

    bodo.jit(test_convert_dtype_true)(S)

    with pytest.raises(
        BodoError, match="Series.apply.* not support built-in functions yet"
    ):
        bodo.jit(test_np_func)(S)

    with pytest.raises(
        BodoError, match="Series.apply.*: user-defined function not supported"
    ):
        bodo.jit(test_wrong_func)(S)

    with pytest.raises(
        BodoError,
        match=r"Series.apply.*: user-defined function not supported: got an unexpected keyword argument",
    ):
        bodo.jit(test_wrong_arg)(S)


# TODO: Add memory leak check once constant lowering memory leak is fixed
@pytest.mark.slow
def test_series_map_supported_types(series_val):
    """ Test Series.map with all Bodo supported Types """

    def test_impl(S):
        return S.map(lambda a: a)  # if not pd.isna(a) else None)

    # nan vs. 0
    S = series_val.dropna()
    # Increase data size to pass testing with 3 ranks
    S = S.repeat(3)
    # Disable check_dtype since Pandas return int64 vs. Int64
    check_func(test_impl, (S,), check_dtype=False)


@pytest.mark.slow
def test_series_map_args(memory_leak_check):
    """ Test Series.map with unsupported and wrong arguments """

    def test_na_action_ignore(S):
        return S.map(lambda a: a, na_action="ignore")

    def test_na_action_none(S):
        return S.map(lambda a: a, na_action=None)

    def test_np_func(S):
        return S.map(np.abs)

    def test_wrong_func(S):
        return S.map("XX")

    S = pd.Series([2, 1, 3])
    with pytest.raises(BodoError, match="Series.map.* only supports default value"):
        bodo.jit(test_na_action_ignore)(S)

    bodo.jit(test_na_action_none)(S)

    with pytest.raises(
        BodoError, match="Series.map.* not support built-in functions yet"
    ):
        bodo.jit(test_np_func)(S)

    with pytest.raises(
        BodoError, match="Series.map.*: user-defined function not supported"
    ):
        bodo.jit(test_wrong_func)(S)


# TODO: add memory_leak_check after fix its failure with Categorical
@pytest.mark.slow
def test_series_groupby_supported_types(series_val):
    """ Test Series.groupby with all Bodo supported Types """

    def test_impl(S):
        return S.groupby(level=0).max()

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        series_val = series_val.cat.as_ordered()

    # TODO: [BE-346] Gives nan with array of lists.
    # series_val16: bodo_output: [nan, nan, nan, nan, nan]
    if isinstance(series_val.values[0], list):
        return

    check_func(
        test_impl,
        (series_val,),
        check_names=False,
        sort_output=True,
        reset_index=True,
        check_categorical=False,  # Bodo keeps categorical type, Pandas changes series type
        check_dtype=False,
    )


@pytest.mark.slow
def test_series_groupby_by_arg_supported_types(series_val, memory_leak_check):
    """ Test Series.groupby by argument with all Bodo supported Types """

    def test_impl_by(S, byS):
        return S.groupby(byS).mean()

    # TODO: [BE-347]
    if series_val.dtype == np.bool_ or is_bool_object_series(series_val):
        return

    if isinstance(series_val.values[0], list):
        return

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # matches length for both series_val and by argument
    if len(series_val) > 5:
        series_val = series_val[:5]

    S = pd.Series([390.0, 350.0, 30.0, 20.0, 5.5])
    check_func(
        test_impl_by,
        (S, series_val.values),
        check_names=False,
        sort_output=True,
        reset_index=True,
    )

    # TODO: [BE-347] support boolean arrays for `by` argument
    # def test_impl_by_types(S):
    #    return S.groupby(S>100).mean()
    # S = pd.Series([390., 350., 30., 20.])
    # check_func(test_impl_by_types, (S, ))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_map(S, memory_leak_check):
    def test_impl(S):
        return S.map(lambda a: 2 * a)

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S,d",
    [
        (pd.Series([1.0, 2.0, np.nan, 4.0, 1.0]), {1.0: "A", 4.0: "DD"}),
        (
            pd.Series(
                ["AA", "B", "ABC", "D", None, "AA", "B"],
                [3, 1, 0, 2, 4, -1, -2],
                name="ABC",
            ),
            {"AA": 1, "B": 2},
        ),
        (
            pd.Series(
                ["AA", "B", "ABC", "D", None, "AA", "B"],
                [3, 1, 0, 2, 4, -1, -2],
                name="ABC",
            ),
            {"AA": "ABC", "B": "DGE"},
        ),
    ],
)
def test_series_map_dict_arg(S, d, memory_leak_check):
    """test passing dict mapper to Series.map()"""

    def test_impl(S, d):
        return S.map(d)

    check_func(test_impl, (S, d), check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
        pd.Series([-1, 11, 2, 3, 5]),
    ],
)
def test_series_map_none(S, memory_leak_check):
    """Test returning None from UDF"""

    def test_impl(S):
        return S.map(lambda a: 2 * a if a > 2 else None)

    check_func(test_impl, (S,), check_dtype=False)


def test_series_map_none_str(memory_leak_check):
    """Test returning None from UDF with string output"""

    def test_impl(S):
        return S.map(lambda a: a + "2" if not pd.isna(a) else None)

    S = pd.Series(["AA", "B", np.nan, "D", "CDE"] * 4)
    check_func(test_impl, (S,), check_dtype=False, only_1DVar=True)


def test_series_map_none_timestamp(memory_leak_check):
    """Test returning Optional(timestamp) from UDF"""

    def impl(S):
        return S.map(
            lambda a: a + datetime.timedelta(days=1) if a.year < 2019 else None
        )

    S = pd.Series(pd.date_range(start="2018-04-24", end="2019-04-29", periods=5))
    check_func(impl, (S,))


def test_series_map_isna_check(memory_leak_check):
    """Test checking for NA input values in UDF"""

    def impl1(S):
        return S.map(
            lambda a: pd.Timestamp("2019-01-01")
            if pd.isna(a)
            else a + datetime.timedelta(days=1)
        )

    def impl2(S):
        return S.map(
            lambda a: pd.Timedelta(days=3) if pd.isna(a) else a + pd.Timedelta(days=1)
        )

    S = pd.date_range(start="2018-04-24", end="2019-04-29", periods=5).to_series()
    S.iloc[2:4] = None
    check_func(impl1, (S,))
    S = pd.timedelta_range(start="1 day", end="2 days", periods=5).to_series()
    S.iloc[2:4] = None
    check_func(impl2, (S,))


def test_series_map_global1(memory_leak_check):
    def test_impl(S):
        return S.map(arg=lambda a: a + GLOBAL_VAL)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def g1(a):
    return 2 * a


@bodo.jit
def g2(a):
    return 2 * a + 3


@bodo.jit
def g3(a):
    return g2(a=a)


@pytest.mark.slow
def test_series_map_func_cases1(memory_leak_check):
    """test map() called with a function defined as global/freevar outside or passed as
    argument.
    """
    # const function defined as global
    def test_impl1(S):
        return S.map(g1)

    f = lambda a: 2 * a + 1

    # const function defined as freevar
    def test_impl2(S):
        return S.map(f)

    # const function or jit function passed as argument
    def test_impl3(S, h):
        return S.map(h)

    # test function closure
    def test_impl4(S):
        def f2(a):
            return 2 * a + 1

        return S.map(lambda x: f2(x))

    # test const str freevar, function freevar, function closure
    s = "AA"

    def test_impl5(S):
        def f2(a):
            return 2 * a + 1

        return S.map(lambda x: f(x) + f2(x) + len(s))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S,))
    check_func(test_impl3, (S, g1))
    check_func(test_impl3, (S, g2))
    check_func(test_impl4, (S,))
    check_func(test_impl5, (S,))


def test_series_map_global_jit(memory_leak_check):
    """Test UDF defined as a global jit function"""

    def test_impl(S):
        return S.map(g2)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


# TODO: add memory_leak_check
def test_series_map_tup1():
    def test_impl(S):
        return S.map(lambda a: (a, 2 * a))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
    # TODO: support unbox for column of tuples
    # check_func(test_impl, (S,))


def test_series_map_tup_map1(memory_leak_check):
    def test_impl(S):
        A = S.map(lambda a: (a, 2 * a))
        return A.map(lambda a: a[1])

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_map_tup_list1(memory_leak_check):
    """test returning a list of tuples from UDF"""

    def test_impl(S):
        A = S.map(lambda a: [(a, 2 * a), (a, 3 * a)])
        return A

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_map_tup_list2(memory_leak_check):
    """test returning a list of list of tuples from UDF"""

    def test_impl(S):
        A = S.map(
            lambda a: [[(a, 2 * a), (a + 1, 3 * a)], [(3 * a, 4 * a), (a + 2, 7 * a)]]
        )
        return A

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_map_tup_list3(memory_leak_check):
    """test returning a list of tuples with variable size data from UDF"""

    def test_impl(S):
        A = S.map(lambda a: [(a, "A", 2 * a), (a, "AB", 3 * a)])
        return A

    S = pd.Series([1, 2, 3, 4, 5])
    check_func(test_impl, (S,))


def test_series_map_str(memory_leak_check):
    """test string output in map"""

    def test_impl(S):
        return S.map(lambda a: str(a))

    S = pd.Series([1, 211, 333, 43, 51])
    check_func(test_impl, (S,))


def test_series_map_list_str(memory_leak_check):
    """test list(str) output in map"""

    def test_impl(S):
        return S.map(lambda a: [str(a), "AA"] if a > 200 else ["A"])

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(test_impl, (S,))


def test_series_map_array_item(memory_leak_check):
    """test array(item) output in map"""

    def test_impl(S):
        return S.map(lambda a: [a, 3] if a > 200 else [2 * a])

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(test_impl, (S,))


def test_series_map_array_item_input(memory_leak_check):
    """test array(item) input and output in map"""

    def test_impl(S):
        return S.map(lambda a: a[1:4])

    S = pd.Series(
        [
            np.array([1.2, 3.2, 4.2, 5.5, 6.1, 7.6]),
            np.array([1.2, 4.4, 2.1, 2.1, 12.3, 1112.4]),
        ]
        * 11
    )
    check_func(test_impl, (S,))


def test_series_map_dict(memory_leak_check):
    """test dict output in map"""

    # homogeneous output
    def impl1(S):
        return S.map(lambda a: {"A": a + 1, "B": a ** 2})

    # heterogeneous output
    def impl2(S):
        return S.map(lambda a: {"A": a + 1, "B": a ** 2.0})

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(impl1, (S,))
    check_func(impl2, (S,))


def test_series_map_dict_input(memory_leak_check):
    """test dict input in map"""

    def impl(S):
        return S.map(lambda a: a[1])

    S = pd.Series([{1: 1.4, 2: 3.1}, {7: -1.2, 1: 2.2}] * 3)
    check_func(impl, (S,))


def test_series_map_date(memory_leak_check):
    """make sure datetime.date output can be handled in map() properly"""

    def test_impl(S):
        return S.map(lambda a: a.date())

    S = pd.Series(pd.date_range(start="2018-04-24", end="2019-04-29", periods=5))
    check_func(test_impl, (S,))


@pytest.mark.smoke
def test_series_map_full_pipeline(memory_leak_check):
    """make sure full Bodo pipeline is run on UDFs, including untyped pass."""

    # datetime.date.today() requires untyped pass
    def test_impl(S):
        return S.map(lambda a: datetime.date.today())

    S = pd.Series(pd.date_range(start="2018-04-24", end="2019-04-29", periods=5))
    check_func(test_impl, (S,))


def test_series_map_timestamp(memory_leak_check):
    """make sure Timestamp (converted to datetime64) output can be handled in map()
    properly
    """

    def test_impl(S):
        return S.map(lambda a: a + datetime.timedelta(days=1))

    S = pd.Series(pd.date_range(start="2018-04-24", end="2019-04-29", periods=5))
    check_func(test_impl, (S,))


def test_series_map_decimal(memory_leak_check):
    """make sure Decimal output can be handled in map() properly"""
    # just returning input value since we don't support any Decimal creation yet
    # TODO: support Decimal(str) constructor
    # TODO: fix using freevar constants in UDFs
    def test_impl(S):
        return S.map(lambda a: a)

    S = pd.Series(
        [
            Decimal("1.6"),
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
    )
    check_func(test_impl, (S,))


def test_series_map_dt_str(memory_leak_check):
    """test string output in map with dt64/Timestamp input"""

    def test_impl(S):
        return S.map(lambda a: str(a.year))

    S = pd.date_range(start="2018-04-24", periods=11).to_series()
    check_func(test_impl, (S,))


def test_series_map_nested_func(memory_leak_check):
    """test nested Bodo call in map UDF"""

    def test_impl(S):
        return S.map(lambda a: g3(a))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # NOTE: not using check_func since regular Pandas calling g3 can cause hangs due to
    # barriers generated by Bodo
    res = bodo.jit(
        test_impl, all_args_distributed_block=True, all_returns_distributed=True
    )(_get_dist_arg(S, False))
    res = bodo.allgatherv(res)
    py_res = S.apply(lambda a: 2 * a + 3)
    pd.testing.assert_series_equal(res, py_res)


def test_series_map_arg_fold(memory_leak_check):
    """test handling UDF default value (argument folding)"""

    def test_impl(S):
        return S.map(lambda a, b=1.1: a + b)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_autocorr(memory_leak_check):
    def f(S, lag):
        return S.autocorr(lag=lag)

    random.seed(5)
    n = 80
    e_list = [random.randint(1, 10) for _ in range(n)]
    S = pd.Series(e_list)
    check_func(f, (S, 1))
    check_func(f, (S, 40))


def test_monotonicity(memory_leak_check):
    def f1(S):
        return S.is_monotonic_increasing

    def f2(S):
        return S.is_monotonic_decreasing

    def f3(S):
        return S.is_monotonic

    random.seed(5)
    n = 100
    e_list = [random.randint(1, 10) for _ in range(n)]
    Srand = pd.Series(e_list)
    S_inc = Srand.cumsum()
    S_dec = Srand.sum() - S_inc
    #
    e_list_fail = e_list
    e_list[random.randint(0, n - 1)] = -1
    Srand2 = pd.Series(e_list_fail)
    S_inc_fail = Srand2.cumsum()
    check_func(f1, (S_inc,))
    check_func(f2, (S_inc,))
    check_func(f3, (S_inc,))
    check_func(f1, (S_dec,))
    check_func(f2, (S_dec,))
    check_func(f3, (S_dec,))
    check_func(f1, (S_inc_fail,))
    check_func(f3, (S_inc_fail,))


def test_series_map_error_check(memory_leak_check):
    """make sure proper error is raised when UDF is not supported"""

    def test_impl(S):
        # lambda calling a non-jit function that we don't support
        return S.map(lambda a: g1(a))

    S = pd.Series([2, 1, 3])
    with pytest.raises(
        BodoError, match="Series.map.*: user-defined function not supported"
    ):
        bodo.jit(test_impl)(S)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_rolling(S, memory_leak_check):
    def test_impl(S):
        return S.rolling(3).sum()

    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_rolling_kw(memory_leak_check):
    def test_impl(S):
        return S.rolling(window=3, center=True).sum()

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]), marks=pytest.mark.slow),
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_cumsum(S, memory_leak_check):
    # TODO: datetime64, timedelta64
    # TODO: support skipna
    def test_impl(S):
        return S.cumsum()

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]), marks=pytest.mark.slow),
        pytest.param(pd.Series([2, 3, 5, 8, 7]), marks=pytest.mark.slow),
        pytest.param(pd.Series([7, 6, 5, 4, 1]), marks=pytest.mark.slow),
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_cum_minmaxprod(S, memory_leak_check):
    def f1(S):
        return S.cumprod()

    def f2(S):
        return S.cummin()

    def f3(S):
        return S.cummax()

    check_func(f1, (S,))
    check_func(f2, (S,))
    check_func(f3, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]), marks=pytest.mark.slow),
        pytest.param(pd.Series([2, 3, 5, 8, 7]), marks=pytest.mark.slow),
        pytest.param(pd.Series([7, 6, 5, 4, 1]), marks=pytest.mark.slow),
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_np_prod(S, memory_leak_check):
    def impl(S):
        return np.prod(S)

    check_func(impl, (S,))


def test_series_rename(memory_leak_check):
    # TODO: renaming labels, etc.
    def test_impl(A):
        return A.rename("B")

    S = pd.Series([1.0, 2.0, np.nan, 1.0], name="A")
    check_func(test_impl, (S,))


def test_series_abs(memory_leak_check):
    def test_impl(S):
        return S.abs()

    S = pd.Series([np.nan, -2.0, 3.0])
    check_func(test_impl, (S,))


def test_series_min(series_val, memory_leak_check):
    # timedelta setitem not supported yet
    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    def test_impl(A):
        return A.min()

    check_func(test_impl, (series_val,))


def test_series_max(series_val, memory_leak_check):
    # timedelta setitem not supported yet
    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    def test_impl(A):
        return A.max()

    check_func(test_impl, (series_val,))


def test_series_min_max_cat(memory_leak_check):
    """test min/max of Categorical data"""

    def impl1(S):
        return S.min()

    def impl2(S):
        return S.max()

    # includes NAs since value 11 is not in categories
    S = pd.Series(
        [3, 2, 1, 3, 11, 2, -3, 11, 2],
        dtype=pd.CategoricalDtype([4, 3, 1, 2, -3, 15], ordered=True),
    )
    check_func(impl1, (S,))
    check_func(impl2, (S,))


def test_min_max_sum_series(memory_leak_check):
    """Another syntax for computing the maximum"""

    def f1(S):
        return min(S)

    def f2(S):
        return max(S)

    def f3(S):
        return sum(S)

    S = pd.Series([1, 3, 4, 2])
    check_func(f1, (S,), is_out_distributed=False)
    check_func(f2, (S,), is_out_distributed=False)
    check_func(f3, (S,), is_out_distributed=False)


def test_series_min_max_int_output_type(memory_leak_check):
    """make sure output type of min/max for integer input is not converted to float"""

    def impl1(S):
        return S.min()

    def impl2(S):
        return S.max()

    S = pd.Series([1, 3, 4, 2])
    check_func(impl1, (S,), is_out_distributed=False)
    check_func(impl2, (S,), is_out_distributed=False)


def test_series_idxmin(series_val, memory_leak_check):
    def test_impl(A):
        return A.idxmin()

    err_msg = r"Series.idxmin\(\) only supported for numeric array types."

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # not supported for Decimal yet, TODO: support and test
    # This isn't supported in Pandas
    if isinstance(series_val.values[0], Decimal):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # not supported for Strings, TODO: support and test
    # This isn't supported in Pandas
    if not isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.values[0], str
    ):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # Boolean Array, datetime.date, Categorical Array, and Nullable Integer not supported in Pandas
    if series_val.dtype == object and is_bool_object_series(series_val):
        py_output = test_impl(series_val.dropna().astype(np.bool_))
    elif series_val.dtype == object and isinstance(series_val.iat[0], datetime.date):
        py_output = test_impl(series_val.dropna().astype(np.dtype("datetime64[ns]")))
    elif isinstance(series_val.dtype, pd.CategoricalDtype):
        series_val = series_val.astype(
            pd.CategoricalDtype(series_val.dtype.categories, ordered=True)
        )
        na_dropped = series_val.dropna()
        py_output = na_dropped.index[na_dropped.values.codes.argmin()]
    elif isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        py_output = test_impl(series_val.dropna().astype(series_val.dtype.numpy_dtype))
    else:
        py_output = None
    check_func(test_impl, (series_val,), py_output=py_output)


def test_series_idxmax(series_val, memory_leak_check):
    def test_impl(A):
        return A.idxmax()

    err_msg = r"Series.idxmax\(\) only supported for numeric array types."

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # not supported for Decimal yet, TODO: support and test
    # This isn't supported in Pandas
    if isinstance(series_val.values[0], Decimal):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # not supported for Strings, TODO: support and test
    # This isn't supported in Pandas
    if not isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.values[0], str
    ):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val)
        return

    # Boolean Array, datetime.date, Categorical Array, and Nullable Integer not supported in Pandas
    if series_val.dtype == object and is_bool_object_series(series_val):
        py_output = test_impl(series_val.dropna().astype(np.bool_))
    elif series_val.dtype == object and isinstance(series_val.iat[0], datetime.date):
        py_output = test_impl(series_val.dropna().astype(np.dtype("datetime64[ns]")))
    elif isinstance(series_val.dtype, pd.CategoricalDtype):
        series_val = series_val.astype(
            pd.CategoricalDtype(series_val.dtype.categories, ordered=True)
        )
        na_dropped = series_val.dropna()
        py_output = na_dropped.index[na_dropped.values.codes.argmax()]
    elif isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        py_output = test_impl(series_val.dropna().astype(series_val.dtype.numpy_dtype))
    else:
        py_output = None
    check_func(test_impl, (series_val,), py_output=py_output)


@pytest.mark.parametrize(
    "numeric_series_median",
    [
        pytest.param(pd.Series([1, 2, 3]), marks=pytest.mark.slow),
        pytest.param(pd.Series([1, 2, 3, 4]), marks=pytest.mark.slow),
        pd.Series([1.0, 2.0, 4.5, 5.0, np.nan]),
        pytest.param(pd.Series(4 * [np.nan]), marks=pytest.mark.slow),
        pytest.param(
            pd.Series(
                [
                    Decimal("1"),
                    Decimal("2"),
                    Decimal("4.5"),
                    Decimal("5"),
                    np.nan,
                    Decimal("4.9"),
                ]
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_series_median(numeric_series_median, memory_leak_check):
    """There is a memory leak needing to be resolved"""

    def f(S):
        return S.median()

    def f_noskip(S):
        return S.median(skipna=False)

    check_func(f, (numeric_series_median,))
    check_func(f_noskip, (numeric_series_median,), check_dtype=False)


@pytest.mark.slow
def test_series_median_nullable(memory_leak_check):
    """<NA> values from pandas correspond to np.nan from bodo. So specific test"""
    S = pd.Series(pd.array([1, None, 2, 3], dtype="UInt16"))

    def f(S):
        return S.median(skipna=False)

    bodo_f = bodo.jit(f)
    ret_val1 = f(S)
    ret_val2 = bodo_f(S)
    assert pd.isnull(ret_val1) == pd.isnull(ret_val2)


def test_series_equals(memory_leak_check):
    def f(S1, S2):
        return S1.equals(S2)

    S1 = pd.Series([0] + list(range(20)))
    S2 = pd.Series([1] + list(range(20)))
    check_func(f, (S1, S1))
    check_func(f, (S1, S2))


@pytest.mark.slow
def test_series_equals_true(series_val, memory_leak_check):
    """
    Tests that all series values can be used in equals.
    Every value is expected to return True.
    """

    def test_impl(S1, S2):
        return S1.equals(S2)

    # TODO: [BE-109] support equals with ArrayItemArrayType
    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match=r"Series.equals\(\) not supported for Series where each element is an array or list",
        ):
            bodo.jit(test_impl)(series_val, series_val)
        return

    check_func(test_impl, (series_val, series_val))


@pytest.mark.slow
def test_series_equals_false(series_val, memory_leak_check):
    """
    Tests that all series values with different types
    return False.
    """
    # Series that matches another series but differs in type
    other = pd.Series([1, 8, 4, 0, 3], dtype=np.uint16)

    def test_impl(S1, S2):
        return S1.equals(S2)

    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match=r"Series.equals\(\) not supported for Series where each element is an array or list",
        ):
            bodo.jit(test_impl)(series_val, other)
        return

    check_func(test_impl, (series_val, other))


def test_series_head(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    def test_impl(S):
        return S.head(3)

    check_func(test_impl, (series_val,))


def test_series_tail(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    def test_impl(S):
        return S.tail(3)

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_tail_default_args(memory_leak_check):
    """
    Test that series.tail() works with the default args.
    """

    def test_impl(S):
        return S.tail()

    S = pd.Series(np.arange(5))
    check_func(test_impl, (S,))


# Pandas returns an empty df but Bodo is returning the whole df.
@pytest.mark.skip("[BE-95]: Incorrect result")
def test_series_tail_zero(memory_leak_check):
    S = pd.Series([3, 4, 0, 2, 5])

    def test_impl(S):
        return S.tail(0)

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S,values",
    [
        (pd.Series([3, 2, np.nan, 2, 7], [3, 4, 2, 1, 0], name="A"), [2.0, 3.0]),
        (
            pd.Series(
                ["aa", "b", "ccc", "A", np.nan, "b"], [3, 4, 2, 1, 0, -1], name="A"
            ),
            ["aa", "b"],
        ),
        (
            pd.Series([4, 9, 1, 2, 4], [3, 4, 2, 1, 0], name="A"),
            pd.Series([3, 4, 0, 2, 5]),
        ),
    ],
)
def test_series_isin(S, values, memory_leak_check):
    def test_impl(S, values):
        return S.isin(values)

    check_func(test_impl, (S, values))


# TODO: Readd the memory leak check when constant lower leak is fixed
# This leak results from Categorical Constant lowering
@pytest.mark.slow
def test_series_isin_true(series_val):
    """
    Checks Series.isin() works with a variety of Series types.
    This aims at ensuring everything can compile because all
    values should be true.
    """

    def test_impl(S, values):
        return S.isin(values)

    values = series_val.iloc[:2]

    # Pandas doesn't support nested data with isin
    # This may result in a system error
    # See: https://github.com/pandas-dev/pandas/issues/20883
    if isinstance(series_val.values[0], list):
        # This seems to work in Bodo
        py_output = pd.Series(
            [True, True] + [False] * (len(series_val) - 2), index=series_val.index
        )
    elif isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.dtype.categories, (pd.TimedeltaIndex, pd.DatetimeIndex)
    ):
        # Bug in Pandas https://github.com/pandas-dev/pandas/issues/36550
        py_output = pd.Series(
            [True, True] + [False] * (len(series_val) - 2), index=series_val.index
        )
    else:
        py_output = None
    # TODO: Check distributed
    check_func(test_impl, (series_val, values), py_output=py_output, dist_test=False)


@pytest.mark.slow
def test_series_isin_large_random(memory_leak_check):
    def get_random_array(n, len_siz):
        elist = []
        for _ in range(n):
            val = random.randint(1, len_siz)
            elist.append(val)
        return elist

    def test_impl(S, values):
        return S.isin(values)

    random.seed(5)
    S = pd.Series(get_random_array(1000, 100))
    values = pd.Series(get_random_array(100, 100))
    check_func(test_impl, (S, values))


@pytest.mark.slow
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_series_nlargest(numeric_series_val, k, memory_leak_check):
    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(S, k):
        return S.nlargest(k)

    check_func(test_impl, (numeric_series_val, k), False)


def test_series_nlargest_non_index(memory_leak_check):
    # test Series with None as Index
    def test_impl(k):
        S = pd.Series([3, 5, 6, 1, 9])
        return S.nlargest(k)

    bodo_func = bodo.jit(test_impl)
    k = 3
    pd.testing.assert_series_equal(bodo_func(k), test_impl(k))


@pytest.mark.slow
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_series_nsmallest(numeric_series_val, k, memory_leak_check):
    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(S, k):
        return S.nsmallest(k)

    check_func(test_impl, (numeric_series_val, k), False)


def test_series_nsmallest_non_index(memory_leak_check):
    # test Series with None as Index
    def test_impl(k):
        S = pd.Series([3, 5, 6, 1, 9])
        return S.nsmallest(k)

    bodo_func = bodo.jit(test_impl)
    k = 3
    pd.testing.assert_series_equal(bodo_func(k), test_impl(k))


def test_series_take(series_val, memory_leak_check):
    def test_impl(A):
        return A.take([2, 3])

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )
    # TODO: dist support for selection with index list


def test_series_argsort_fast(memory_leak_check):
    def test_impl(A):
        return A.argsort()

    S = pd.Series([3, 5, 6, 1, 9])
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
    # TODO: support distributed argsort()
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_argsort(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    def test_impl(A):
        return A.argsort()

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(series_val), test_impl(series_val))
    # TODO: support distributed argsort()
    # check_func(test_impl, (series_val,))


# TODO: add memory_leak_check
def test_series_sort_values(series_val):
    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # XXX can't push NAs to the end, TODO: fix
    if series_val.hasnans:
        return

    # BooleanArray can't be key in sort, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.sort_values()

    check_func(test_impl, (series_val,), check_typing_issues=False)


# TODO(ehsan): add memory_leak_check when categorical (series_val18) leak is fixed
def test_series_repeat(series_val):
    """Test Series.repeat() method"""

    def test_impl(S, n):
        return S.repeat(n)

    check_func(test_impl, (series_val, 3))
    check_func(test_impl, (series_val, np.arange(len(series_val))))


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_single(series_val, ignore_index, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    func_text = "def test_impl(A, B):\n"
    func_text += "  return A.append(B, {})\n".format(ignore_index)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val, series_val),
        test_impl(series_val, series_val),
        check_dtype=False,
        check_names=False,
    )  # XXX append can't set name yet


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_multi(series_val, ignore_index, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    func_text = "def test_impl(A, B, C):\n"
    func_text += "  return A.append([B, C], {})\n".format(ignore_index)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val, series_val, series_val),
        test_impl(series_val, series_val, series_val),
        check_dtype=False,
        check_names=False,
    )  # XXX append can't set name yet


def test_series_quantile(numeric_series_val, memory_leak_check):
    # quantile not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(A):
        return A.quantile(0.30)

    # TODO: needs np.testing.assert_almost_equal?
    check_func(test_impl, (numeric_series_val,))


def test_series_quantile_q(memory_leak_check):
    """ Test passing list, int, and unsupported type to q argument"""

    # List
    def test_impl(S):
        ans = S.quantile([0.15, 0.5, 0.65, 1])
        return ans

    S = pd.Series([1.2, 3.4, 4.5, 32.3, 67.8, 100])

    check_func(test_impl, (S,), is_out_distributed=False)

    # int
    def test_int(S):
        ans = S.quantile(0)
        return ans

    check_func(test_int, (S,))

    def test_str(S):
        ans = S.quantile("aa")
        return ans

    # unsupported q type
    with pytest.raises(BodoError, match=r"Series.quantile\(\) q type must be float"):
        bodo.jit(test_str)(S)


def test_series_nunique(series_val, memory_leak_check):
    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # doesn't support NAs yet, TODO: support and test
    if series_val.hasnans:
        return

    # not supported for dt64 yet, TODO: support and test
    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # BooleanArray can't be key in shuffle, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.nunique()

    check_func(test_impl, (series_val,))


def test_series_unique(series_val, memory_leak_check):
    # timedelta setitem not supported yet
    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # not supported for dt64 yet, TODO: support and test
    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # np.testing.assert_array_equal() throws division by zero for bool arrays
    # with nans for some reason
    if series_val.dtype == np.dtype("O") and series_val.hasnans:
        return

    # BooleanArray can't be key in shuffle, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.unique()

    # sorting since output order is not consistent
    check_func(test_impl, (series_val,), sort_output=True, is_out_distributed=False)


def test_series_describe(numeric_series_val, memory_leak_check):
    # not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(A):
        return A.describe()

    check_func(test_impl, (numeric_series_val,), False)


# TODO: move some cases to slow path
def test_series_reset_index_no_drop(memory_leak_check):
    """Test Series.reset_index(drop=False)"""

    def impl1(df):
        return df["A"].reset_index(drop=False)

    # add value_counts() since common pattern
    def impl2(df):
        return df["A"].value_counts().reset_index(drop=False)

    # Series name is index, so output should use level_0
    def impl3(df):
        return df["index"].reset_index(drop=False)

    df = pd.DataFrame({"A": [1, 2, 3, 4, 1, 2, 3]})
    check_func(impl1, (df,))
    # value_counts returns 2 in multiple rows, so Bodo may not match
    # the Pandas output. As a result, we sort the output and replace
    # the index. Fix [BE-253]
    check_func(impl2, (df,), sort_output=True, reset_index=True)
    df = pd.DataFrame({"index": [1, 2, 3, 4, 1, 2, 3]})
    check_func(impl3, (df,))


def test_series_reset_index(memory_leak_check):
    """Test Series.reset_index(drop=True)"""

    def impl(S):
        return S.reset_index(level=[0, 1], drop=True)

    S = pd.Series(
        [3, 2, 1, 5],
        index=pd.MultiIndex.from_arrays(([3, 2, 1, 4], [5, 6, 7, 8])),
        name="A",
    )
    check_func(impl, (S,))


def test_series_reset_index_error(memory_leak_check):
    """make sure compilation doesn't hang with reset_index/setattr combination,
    see [BE-140]
    """

    def impl(S):
        df = S.reset_index(drop=False)
        df.columns = ["A", "B"]
        return df

    S = pd.Series(
        [3, 2, 1, 5],
        index=[3, 2, 1, 4],
        name="A",
    )
    with pytest.raises(
        BodoError,
        match=r"Series.reset_index\(\) not supported for non-literal series names",
    ):
        bodo.jit(impl)(S)


@pytest.mark.parametrize(
    "S,value",
    [
        (pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"), 5.0),
        (
            pd.Series([1.0, 2.0, np.nan, 1.0, np.nan], name="A"),
            pd.Series([2.1, 3.1, np.nan, 3.3, 1.2]),
        ),
        (pd.Series(["aa", "b", "C", None, "ccc"], [3, 4, 0, 2, 1], name="A"), "dd"),
        (
            pd.Series(["aa", "b", None, "ccc", None, "AA"] * 2, name="A"),
            pd.Series(["A", "C", None, "aa", "dd", "d"] * 2),
        ),
    ],
)
def test_series_fillna(S, value, memory_leak_check):
    def test_impl(A, val):
        return A.fillna(val)

    check_func(test_impl, (S, value))


@pytest.mark.parametrize(
    "S,value",
    [
        (pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"), 5.0),
        (
            pd.Series([1.0, 2.0, np.nan, 1.0, np.nan], name="A"),
            pd.Series([2.1, 3.1, np.nan, 3.3, 1.2]),
        ),
        (pd.Series(["aa", "b", "A", None, "ccc"], [3, 4, 2, 0, 1], name="A"), "dd"),
        (
            pd.Series(["aa", "b", None, "ccc", None, "AA"] * 2, name="A"),
            pd.Series(["A", "C", None, "aa", "dd", "d"] * 2),
        ),
    ],
)
def test_series_fillna_inplace(S, value, memory_leak_check):
    def test_impl(A, val):
        return A.fillna(val, inplace=True)

    check_func(test_impl, (S, value))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"),
        pd.Series(["aa", "b", "AA", None, "ccc"], [3, 4, -1, 2, 1], name="A"),
    ],
)
def test_series_dropna(S, memory_leak_check):
    def test_impl(A):
        return A.dropna()

    check_func(test_impl, (S,))


def test_series_drop_inplace_check(memory_leak_check):
    """make sure inplace=True is not use in Series.dropna()"""

    def test_impl(S):
        S.dropna(inplace=True)

    S = pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A")
    with pytest.raises(
        BodoError, match="inplace parameter only supports default value False"
    ):
        bodo.jit(test_impl)(S)


@pytest.mark.parametrize(
    "S,to_replace,value",
    [
        (
            pd.Series([1.0, 2.0, np.nan, 1.0, 2.0, 1.3], [3, 4, 2, 1, -3, 6], name="A"),
            2.0,
            5.0,
        ),
        pytest.param(
            pd.Series(pd.Categorical([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            1,
            2,
        ),
        pytest.param(
            pd.Series(
                ["aa", "bc", None, "ccc", "bc", "A", ""],
                [3, 4, 2, 1, -3, -2, 6],
                name="A",
            ),
            "bc",
            "abdd",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(pd.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            1,
            2,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_series_replace_scalar(S, to_replace, value, memory_leak_check):
    def test_impl(A, to_replace, val):
        return A.replace(to_replace, val)

    check_func(test_impl, (S, to_replace, value))


@pytest.mark.parametrize(
    "S,to_replace_list,value",
    [
        (
            pd.Series([1.0, 2.0, np.nan, 1.0, 2.0, 1.3], [3, 4, 2, 1, -3, 6], name="A"),
            [2.0, 1.3],
            5.0,
        ),
        pytest.param(
            pd.Series(
                ["aa", "bc", None, "ccc", "bc", "A", ""],
                [3, 4, 2, 1, -3, -2, 6],
                name="A",
            ),
            ["bc", "A"],
            "abdd",
        ),
        pytest.param(
            pd.Series(pd.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            [1, 4],
            2,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(pd.Categorical([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            [1, 3],
            2,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                np.array(
                    [
                        Decimal("1.6"),
                        Decimal("-0.222"),
                        Decimal("5.1"),
                        Decimal("1111.316"),
                        Decimal("-0.2220001"),
                        Decimal("-0.2220"),
                        Decimal("1234.00046"),
                        Decimal("5.1"),
                        Decimal("-11131.0056"),
                        Decimal("0.0"),
                        Decimal("5.11"),
                        Decimal("0.00"),
                        Decimal("0.01"),
                        Decimal("0.03"),
                        Decimal("0.113"),
                        Decimal("1.113"),
                    ]
                )
            ),
            [Decimal("5.1"), Decimal("0.0")],
            Decimal("0.001"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_series_replace_list_scalar(S, to_replace_list, value, memory_leak_check):
    def test_impl(A, to_replace, val):
        return A.replace(to_replace, val)

    # Pandas 1.2.0 seems to convert the array from Int64 to int64.
    if isinstance(S.dtype, pd.core.arrays.integer._IntegerDtype):
        check_dtype = False
    else:
        check_dtype = True
    check_func(test_impl, (S, to_replace_list, value), check_dtype=check_dtype)


@pytest.mark.parametrize(
    "S,to_replace_list,value_list",
    [
        (
            pd.Series([1.0, 2.0, np.nan, 1.0, 2.0, 1.3], [3, 4, 2, 1, -3, 6], name="A"),
            [2.0, 1.3],
            [5.0, 5.0],
        ),
        pytest.param(
            pd.Series(
                ["aa", "bc", None, "ccc", "bc", "A", ""],
                [3, 4, 2, 1, -3, -2, 6],
                name="A",
            ),
            ["bc", "A"],
            ["abdd", ""],
        ),
        pytest.param(
            pd.Series(pd.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            [1, 4],
            [2, 1],
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(pd.Categorical([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3, 4])),
            [1, 3],
            [2, 7],
            marks=pytest.mark.skip("Not supported for categorical"),
        ),
        pytest.param(
            pd.Series(
                np.array(
                    [
                        Decimal("1.6"),
                        Decimal("-0.222"),
                        Decimal("5.1"),
                        Decimal("1111.316"),
                        Decimal("-0.2220001"),
                        Decimal("-0.2220"),
                        Decimal("1234.00046"),
                        Decimal("5.1"),
                        Decimal("-11131.0056"),
                        Decimal("0.0"),
                        Decimal("5.11"),
                        Decimal("0.00"),
                        Decimal("0.01"),
                        Decimal("0.03"),
                        Decimal("0.113"),
                        Decimal("1.113"),
                    ]
                )
            ),
            [Decimal("5.1"), Decimal("0.0")],
            [Decimal("0.001"), Decimal("5.1")],
            marks=pytest.mark.slow,
        ),
    ],
)
def test_series_replace_list_list(S, to_replace_list, value_list, memory_leak_check):
    def test_impl(A, to_replace, val):
        return A.replace(to_replace, val)

    check_func(test_impl, (S, to_replace_list, value_list), check_dtype=False)


def test_series_to_dict(memory_leak_check):
    def impl(S):
        return S.to_dict()

    S_index = pd.Series([2, 0, 4, 8, np.nan], [1, 5, -1, -2, 3])
    S = pd.Series(["alpha", "beta", "gamma"])
    check_func(impl, (S,), only_seq=True)
    check_func(impl, (S_index,), only_seq=True)


@pytest.mark.slow
def test_series_replace_dict_float(memory_leak_check):
    """
    Specific test for replace_dict with floats. This isn't setup using pytest
    fixtures because Numba cannot determine the dict type unless the dictionary
    is a constant.
    """

    def test_impl(A):
        return A.replace({2.0: 1.3, 5.0: 4.0})

    S = pd.Series([1.0, 2.0, np.nan, 1.0, 2.0, 1.3], [3, 4, 2, 1, -3, 6], name="A")
    check_func(test_impl, (S,), check_dtype=False)


@pytest.mark.parametrize(
    "periods",
    [2, -2],
)
def test_series_shift(numeric_series_val, periods, memory_leak_check):
    def test_impl(A, periods):
        return A.shift(periods)

    check_func(test_impl, (numeric_series_val, periods))


@pytest.mark.slow
def test_series_shift_type_check(series_val, memory_leak_check):
    """
    Make sure Series.shift() works for supported data types but throws error for
    unsupported ones.
    """

    def test_impl(A):
        return A.shift(1)

    def test_impl2(A):
        return A.shift(-1)

    # test larger shift window to trigger small relative data code path
    def test_impl3(A):
        return A.shift(3)

    # Series.shift supports ints, floats, dt64, nullable nullable int/bool/decimal/date
    # and strings
    if (
        pd.api.types.is_numeric_dtype(series_val)
        or series_val.dtype == np.dtype("datetime64[ns]")
        or isinstance(series_val.values[0], Decimal)
        or series_val.dtype == np.bool_
        or is_bool_object_series(series_val)
        or isinstance(series_val.values[0], datetime.date)
        or isinstance(series_val.values[0], str)
    ) and not isinstance(series_val.dtype, pd.CategoricalDtype):
        check_func(test_impl, (series_val,))
        check_func(test_impl2, (series_val,))
        check_func(test_impl3, (series_val,))
        return

    with pytest.raises(BodoError, match=r"Series.shift\(\): Series input type"):
        bodo.jit(test_impl)(series_val)


@pytest.mark.slow
def test_series_shift_error_periods(memory_leak_check):
    S = pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A")

    def test_impl(S, periods):
        return S.shift(periods)

    with pytest.raises(BodoError, match="'periods' input must be an integer"):
        bodo.jit(test_impl)(S, 1.0)


@pytest.mark.parametrize(
    "periods",
    [2, -2],
)
def test_series_pct_change(numeric_series_val, periods, memory_leak_check):
    # not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(A, periods):
        return A.pct_change(periods)

    check_func(test_impl, (numeric_series_val, periods))


@pytest.mark.parametrize(
    "S,bins",
    [
        (
            pd.Series([11, 21, 55, 41, 11, 77, 111, 81, 3], name="BB"),
            [31, 61, 91],
        ),
        (np.array([11, 21, 55, 41, 11, 77, 111, 81, 3]), [31, 61, 91]),
    ],
)
def test_series_digitize(S, bins, memory_leak_check):
    def test_impl(A, bins):
        return np.digitize(A, bins)

    check_func(test_impl, (S, bins))


@pytest.mark.parametrize(
    "S1,S2",
    [
        (
            pd.Series([1.1, 2.2, 1.3, -1.4, 3.1], name="BB"),
            pd.Series([6.1, 3.1, 2.2, 1.7, 9.1]),
        ),
        (
            pd.Series([1.1, 2.2, 1.3, -1.4, 3.1], name="BB"),
            np.array([6.1, 3.1, 2.2, 1.7, 9.1]),
        ),
        (
            np.array([6.1, 3.1, 2.2, 1.7, 9.1]),
            pd.Series([1.1, 2.2, 1.3, -1.4, 3.1], name="BB"),
        ),
    ],
)
def test_series_np_dot(S1, S2, memory_leak_check):
    def impl1(A, B):
        return np.dot(A, B)

    # using the @ operator
    def impl2(A, B):
        return A @ B

    check_func(impl1, (S1, S2))
    check_func(impl2, (S1, S2))


def test_np_argmax(memory_leak_check):
    def impl(A):
        return np.argmax(A, 1)

    check_func(impl, (np.random.rand(500, 50),))


def test_np_argmin(memory_leak_check):
    def impl(A):
        return np.argmin(A, 1)

    check_func(impl, (np.random.rand(500, 50),))


# TODO: fix memory leak and add memory_leak_check
def test_series_index_cast():
    # cast range index to integer index if necessary
    def test_impl(n):
        if n < 5:
            S = pd.Series([3, 4], [2, 3])
        else:
            S = pd.Series([3, 6])
        return S

    bodo_func = bodo.jit(test_impl)
    n = 10
    pd.testing.assert_series_equal(bodo_func(n), test_impl(n))


def test_series_value_counts(memory_leak_check):
    """simple test for value_counts(). More comprehensive testing is necessary"""

    def test_impl(S):
        return S.value_counts()

    S = pd.Series(["AA", "BB", "C", "AA", "C", "AA"])
    check_func(test_impl, (S,))


def test_series_sum(memory_leak_check):
    def impl(S):
        A = S.sum()
        return A

    def impl_skipna(S):
        A = S.sum(skipna=False)
        return A

    def impl_mincount(S, min_count):
        A = S.sum(min_count=min_count)
        return A

    S_int = pd.Series(np.arange(20))
    S_float = pd.Series([np.nan, 1, 2, 3])
    check_func(impl, (S_int,))
    check_func(impl_skipna, (S_float,))
    check_func(impl_mincount, (S_float, 2))
    check_func(impl_mincount, (S_float, 4))


def test_series_prod(memory_leak_check):
    def impl(S):
        A = S.prod()
        return A

    def impl_skipna(S):
        A = S.prod(skipna=False)
        return A

    def impl_mincount(S, min_count):
        A = S.product(min_count=min_count)
        return A

    S_int = pd.Series(1 + np.arange(20))
    S_float = pd.Series([np.nan, 1.0, 2.0, 3.0])
    check_func(impl, (S_int,))
    check_func(impl_skipna, (S_float,))
    check_func(impl_mincount, (S_float, 2))
    check_func(impl_mincount, (S_float, 4))


def test_singlevar_series_all(memory_leak_check):
    def impl(S):
        A = S.all()
        return A

    S = pd.Series([False] + [True] * 10)
    check_func(impl, (S,))


def test_singleval_series_any(memory_leak_check):
    def impl(S):
        A = S.any()
        return A

    S = pd.Series([True] + [False] * 10)
    check_func(impl, (S,))


@pytest.mark.slow
def test_random_series_all(memory_leak_check):
    def impl(S):
        A = S.all()
        return A

    def random_series(n):
        random.seed(5)
        eList = []
        for i in range(n):
            val = random.randint(0, 2)
            if val == 0:
                val_B = True
            if val == 1:
                val_B = False
            if val == 2:
                val_B = np.nan
            eList.append(val_B)
        return pd.Series(eList)

    S = random_series(111)
    check_func(impl, (S,))


@pytest.mark.slow
def test_random_series_any(memory_leak_check):
    def impl(S):
        A = S.any()
        return A

    def random_series(n):
        random.seed(5)
        eList = []
        for i in range(n):
            val = random.randint(0, 2)
            if val == 0:
                val_B = True
            if val == 1:
                val_B = False
            if val == 2:
                val_B = np.nan
            eList.append(val_B)
        return pd.Series(eList)

    S = random_series(111)
    check_func(impl, (S,))


def test_series_dropna_series_val(memory_leak_check, series_val):
    def impl(S):
        return S.dropna()

    check_func(impl, (series_val,))


def test_series_groupby_arr(memory_leak_check):
    """test Series.groupby() with input array as keys"""

    def impl(S, A):
        return S.groupby(A).sum()

    S = pd.Series([1, 2, 3, 4, 5, 6])
    A = np.array([1, 2, 1, 1, 2, 3])
    check_func(impl, (S, A), check_names=False, sort_output=True)


def test_series_groupby_index(memory_leak_check):
    def impl(S):
        return S.groupby(level=0).sum()

    S = pd.Series([1, 2, 3, 4, 5, 6], [1, 2, 1, 1, 2, 3])
    check_func(impl, (S,), check_names=False, sort_output=True)


def test_series_np_where_str(memory_leak_check):
    """Tests np.where() called on Series with string input (#223)."""

    def test_impl1(S):
        # wrapping array in Series to enable output comparison for NA
        return pd.Series(np.where(S == "aa", S, "d"))

    def test_impl2(S, a):
        return pd.Series(np.where(S == "aa", a, S))

    S = pd.Series(
        ["aa", "b", "aa", "cc", "s", "aa", "DD"], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S, "ddd"))


def test_series_np_where_num(memory_leak_check):
    """Tests np.where() called on Series with numeric input."""

    def test_impl1(S):
        return np.where((S == 2.0), S, 11.0)

    def test_impl2(S, a, cond):
        # cond.values to test boolean_array
        return np.where(cond.values, a, S.values)

    S = pd.Series(
        [4.0, 2.0, 1.1, 9.1, 2.0, np.nan, 2.5], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    cond = S == 2.0
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S, 12, cond))


def test_series_where_true(series_val, memory_leak_check):
    """Tests that all types can be used in Series.where(cond)
    with all True values."""
    cond = np.array([True] * len(series_val))
    val = series_val.iloc[0]

    def test_impl(S, cond, val):
        return S.where(cond, val)

    # TODO: [BE-110] support series.where for more Bodo array types
    series_err_msg = "Series.where.* Series data with type .* not yet supported"

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # not supported for CategoricalArrayType yet, TODO: support and test
    if isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.dtype.categories[0], (pd.Timestamp, pd.Timedelta)
    ):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # not supported for datetime.date yet, TODO: support and test
    if not isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.values[0], datetime.date
    ):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # Bodo differs from Pandas because Bodo sets the type before
    # it knows that the other value (np.nan) will never be chosen
    check_func(test_impl, (series_val, cond, val), check_dtype=False)


def test_series_where(memory_leak_check):
    """basic test for Series.where(cond, val)"""

    def test_impl(S, cond, val):
        return S.where(cond, val)

    S = pd.Series(
        [4.0, 2.0, 1.1, 9.1, 2.0, np.nan, 2.5], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    cond = S == 2.0
    check_func(test_impl, (S, cond, 12))


@pytest.mark.slow
@pytest.mark.parametrize(
    "where_nullable",
    [
        # Bool nullable
        pytest.param(
            WhereNullable(
                series=pd.Series(pd.array([True, True, True, True])),
                cond=pd.array([True, True, True, True]),
                other=pd.Series(pd.array([True, True, True, True])),
            ),
            id="a",
        ),
        # Int64 Series & int64 other
        pytest.param(
            WhereNullable(
                series=pd.Series(pd.array([1, 2, 3, 4]), dtype="Int64"),
                cond=pd.array([True, True, True, True]),
                other=pd.Series([1, 2, 3, 4], dtype="int64"),
            ),
            id="b",
        ),
        # Int64 Series & int32 other
        pytest.param(
            WhereNullable(
                series=pd.Series(pd.array([1, 2, 3, 4]), dtype="Int64"),
                cond=pd.array([True, True, True, True]),
                other=pd.Series([1, 2, 3, 4], dtype="int32"),
            ),
            id="c",
        ),
        # Int8 Series & Int64 other
        pytest.param(
            WhereNullable(
                series=pd.Series(pd.array([1, 2, 3, 4]), dtype="Int8"),
                cond=pd.array([True, True, True, True]),
                other=pd.Series([1, 2, 3, 4], dtype="Int64"),
            ),
            id="d",
        ),
    ],
)
def test_series_where_nullable(where_nullable):
    f = lambda S, cond, other: S.where(cond, other)
    check_func(f, where_nullable, check_dtype=False)


def test_series_where_arr(memory_leak_check):
    """Test for Series.where(cond, arr) where arr is either
    a series or array that shares a common dtype."""

    def test_impl(S, cond, val):
        return S.where(cond, val)

    S = pd.Series(np.array([1, 2, 51, 61, -2], dtype=np.int8))
    other_series = pd.Series([2.1, 232.24, 231.2421, np.nan, 3242.112])
    other_arr = np.array([323, 0, 1341, -4, 232], dtype=np.int16)
    np.random.seed(0)
    cond = np.random.ranf(len(S)) < 0.5
    check_func(test_impl, (S, cond, other_series))
    check_func(test_impl, (S, cond, other_arr))


def test_series_where_str(memory_leak_check):
    """Tests Series.where() with string input"""

    def impl(S):
        return S.where(S == "aa", "d")

    S = pd.Series(
        ["aa", "b", "aa", "cc", np.nan, "aa", "DD"], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    check_func(impl, (S,))


def test_np_where_one_arg(memory_leak_check):
    """basic test for np.where(cond)"""

    def test_impl(cond):
        return np.where(cond)

    S = pd.Series(
        [4.0, 2.0, 1.1, 9.1, 2.0, np.nan, 2.5], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    cond = S == 2.0
    check_func(test_impl, (cond,), dist_test=False)


def test_series_mask_false(series_val, memory_leak_check):
    """Tests that all types can be used in Series.mask(cond)
    with all False values."""
    cond = np.array([False] * len(series_val))
    val = series_val.iloc[0]

    def test_impl(S, cond, val):
        return S.mask(cond, val)

    # TODO: [BE-110] support series.mask for more Bodo array types
    series_err_msg = "Series.mask.* Series data with type .* not yet supported"

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    if isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.dtype.categories[0], (pd.Timestamp, pd.Timedelta)
    ):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # not supported for datetime.date yet, TODO: support and test
    if not isinstance(series_val.dtype, pd.CategoricalDtype) and isinstance(
        series_val.values[0], datetime.date
    ):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # Bodo differs from Pandas because Bodo sets the type before
    # it knows that the other value (np.nan) will never be chosen
    check_func(test_impl, (series_val, cond, val), check_dtype=False)


def test_series_mask(memory_leak_check):
    """basic test for Series.mask(cond, val)"""

    def test_impl(S, cond, val):
        return S.mask(cond, val)

    def test_impl_nan(S, cond):
        return S.mask(cond)

    S = pd.Series(
        [4.0, 2.0, 1.1, 9.1, 2.0, np.nan, 2.5], [5, 1, 2, 0, 3, 4, 9], name="AA"
    )
    cond = S == 2.0
    check_func(test_impl, (S, cond, 12))
    check_func(test_impl_nan, (S, cond))


def test_series_mask_arr(memory_leak_check):
    """Test for Series.where(cond, arr) where arr is either
    a series or array that shares a common dtype."""

    def test_impl(S, cond, val):
        return S.mask(cond, val)

    S = pd.Series(np.array([1, 2, 51, 61, -2], dtype=np.int8))
    other_series = pd.Series([2.1, 232.24, 231.2421, np.nan, 3242.112])
    other_arr = np.array([323, 0, 1341, -4, 232], dtype=np.int16)
    np.random.seed(0)
    cond = np.random.ranf(len(S)) < 0.5
    check_func(test_impl, (S, cond, other_series))
    check_func(test_impl, (S, cond, other_arr))


def test_series_mask_cat_literal(memory_leak_check):
    """Make sure string literal works for setitem of categorical data through mask()"""

    def test_impl(S, cond):
        return S.mask(cond, "AB")

    S = pd.Series(
        ["AB", "AA", "AB", np.nan, "A", "AA", "AB"], [5, 1, 2, 0, 3, 4, 9], name="AA"
    ).astype("category")
    cond = S == "AA"
    check_func(test_impl, (S, cond))


@pytest.mark.parametrize(
    "value, downcast",
    [
        (pd.Series(["1.4", "2.3333", None, "1.22", "555.1"] * 2), "float"),
        (pd.Series([1, 2, 9, 11, 3]), "integer"),
        (pd.Series(["1", "3", "12", "4", None, "-555"]), "integer"),
        # string array with invalid element
        (pd.Series(["1", "3", "12", None, "-55ss"]), "integer"),
        (pd.Series(["1", "3", "12", None, "555"]), "unsigned"),
    ],
)
def test_to_numeric(value, downcast, memory_leak_check):
    def test_impl(S):
        B = pd.to_numeric(S, errors="coerce", downcast=downcast)
        return B

    check_func(test_impl, (value,), check_dtype=False)


@pytest.fixture(
    params=[
        pd.Series([np.nan, -1.0, -1.0, 0.0, 78.0]),
        pd.Series([1.0, 2.0, 3.0, 42.3]),
        pd.Series([1, 2, 3, 42]),
        pytest.param(
            pd.Series([1, 2]),
            marks=pytest.mark.slow,
        ),
    ]
)
def series_stat(request):
    return request.param


def test_series_mad(series_stat, memory_leak_check):
    def f(S):
        return S.mad()

    def f_skip(S):
        return S.mad(skipna=False)

    check_func(f, (series_stat,))
    check_func(f_skip, (series_stat,))


def test_series_skew(series_stat, memory_leak_check):
    def f(S):
        return S.skew()

    def f_skipna(S):
        return S.skew(skipna=False)

    check_func(f, (series_stat,))
    check_func(f_skipna, (series_stat,))


def test_series_kurt(series_stat, memory_leak_check):
    def f(S):
        return S.kurt()

    def f_skipna(S):
        return S.kurt(skipna=False)

    check_func(f, (series_stat,))
    check_func(f_skipna, (series_stat,))


def test_series_kurtosis(series_stat, memory_leak_check):
    def f(S):
        return S.kurtosis()

    def f_skipna(S):
        return S.kurtosis(skipna=False)

    check_func(f, (series_stat,))
    check_func(f_skipna, (series_stat,))


def test_series_dot(memory_leak_check):
    def test_impl(S1, S2):
        return S1.dot(S2)

    S1 = pd.Series([1.0, 2.0, 3.0])
    S2 = pd.Series([3.0, 5.0, 9.0])
    check_func(test_impl, (S1, S2))


@pytest.mark.slow
@pytest.mark.filterwarnings("error:function call couldn't")
def test_astype_call_warn(memory_leak_check):
    """
    Sometimes Numba converts binop exprs to a call with a Const node, which is not
    handled properly in Bodo and throws a warning (see #1838).
    """

    def impl(S):
        return S.astype("category").cat.codes

    bodo.jit(distributed=False)(impl)(pd.Series(["A", "B"]))


############################### old tests ###############################


@pytest.mark.slow
def test_create_series1(memory_leak_check):
    def test_impl():
        A = pd.Series([1, 2, 3])
        return A.values

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_array_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_create_series_index1(memory_leak_check):
    # create and box an indexed Series
    def test_impl():
        A = pd.Series([1, 2, 3], ["A", "C", "B"])
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_create_series_index2(memory_leak_check):
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"])
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_create_series_index3(memory_leak_check):
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name="A")
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_create_series_index4(memory_leak_check):
    def test_impl(name):
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name=name)
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func("A"), test_impl("A"))


def test_series_var(memory_leak_check):
    def f(S):
        return S.var()

    def f_skipna(S):
        return np.isnan(S.var(skipna=False))

    def f_ddof(S):
        return S.var(ddof=2)

    S = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0])
    check_func(f, (S,))
    check_func(f_skipna, (S,))
    check_func(f_ddof, (S,))


def test_series_sem(memory_leak_check):
    def f(S):
        return S.sem()

    def f_skipna(S):
        return np.isnan(S.sem(skipna=False))

    def f_ddof(S):
        return S.sem(ddof=2)

    S = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0])
    check_func(f, (S,))
    check_func(f_skipna, (S,))
    check_func(f_ddof, (S,))


def test_np_pd_timedelta_truediv(memory_leak_check):
    """
    Test that Series.truediv works between a Series of td64
    and a pd.Timedelta type.
    """

    def test_impl(S, val):
        return S / val

    S = pd.Series(pd.timedelta_range(start="1 day", periods=10))
    val1 = pd.Timedelta(days=3)
    val2 = pd.Timedelta(nanoseconds=3)
    val3 = pd.Timedelta(days=-2, seconds=53, minutes=2)
    check_func(test_impl, (S, val1))
    check_func(test_impl, (S, val2))
    check_func(test_impl, (S, val3))


def test_datetime_date_pd_timedelta_ge(memory_leak_check):
    """
    Test that Series.ge works between a Series of datetimedate
    and a pd.Timestamp type.
    """

    def test_impl(S, val):
        return S >= val

    S = pd.Series(pd.date_range(start="1/1/2018", end="1/08/2018").date)
    val1 = pd.Timestamp("1/1/2018")
    val2 = pd.Timestamp("1/1/2021")
    check_func(test_impl, (S, val1))
    check_func(test_impl, (S, val2))


def test_series_std(memory_leak_check):
    def f(S):
        return S.std()

    def f_skipna(S):
        return np.isnan(S.std(skipna=False))

    def f_ddof(S):
        return S.std(ddof=2)

    S = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0])
    check_func(f, (S,))
    check_func(f_skipna, (S,))
    check_func(f_ddof, (S,))


@pytest.mark.slow
def test_add_datetime_series_timedelta(memory_leak_check):
    def test_impl(S1, S2):
        return S1.add(S2)

    datatime_arr = [datetime.datetime(year=2020, month=10, day=x) for x in range(1, 32)]
    S = pd.Series(datatime_arr)
    S2 = pd.Series([datetime.timedelta(days=x, minutes=13) for x in range(-15, 16)])
    check_func(test_impl, (S, S2))


@pytest.mark.slow
def test_add_timedelta_series_timedelta(memory_leak_check):
    def test_impl(S1, S2):
        return S1.add(S2)

    td_arr = [
        datetime.timedelta(seconds=21, minutes=45, hours=17, days=x)
        for x in range(1, 32)
    ]
    S = pd.Series(td_arr)
    S2 = pd.Series([datetime.timedelta(days=x, minutes=13) for x in range(-15, 16)])
    check_func(test_impl, (S, S2))


@pytest.mark.slow
def test_add_timedelta_series_timestamp(memory_leak_check):
    def test_impl(S1, S2):
        return S1.add(S2)

    td_arr = [
        datetime.timedelta(seconds=21, minutes=45, hours=17, days=x)
        for x in range(1, 32)
    ]
    S = pd.Series(td_arr)
    S2 = pd.Series([pd.Timestamp(year=2020, month=10, day=x) for x in range(1, 32)])
    check_func(test_impl, (S, S2))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(
            [[1, 3, None], [2], None, [4, None, 5, 6], [], [1, 1]] * 2,
            [1.2, 3.1, 4.0, -1.2, 0.3, 11.1] * 2,
        ),
        # TODO: enable when old list(str) array is removed
        # pd.Series([["AAC", None, "FG"], [], ["", "AB"], None, ["CC"], [], ["A", "CC"]]*2, ["A", "BB", "C", "", "ABC", "DD", "LKL"]*2),
        # nested array(item) array
        pd.Series(
            [
                [[1, 3], [2]],
                [[3, 1]],
                None,
                [[4, 5, 6], [1], [1, 2]],
                [],
                [[1], None, [1, 4], []],
            ]
            * 2,
            [1.2, 3.1, 4.0, -1.2, 0.3, 11.1] * 2,
        ),
    ],
)
def test_series_explode(S, memory_leak_check):
    def test_impl(S):
        return S.explode()

    check_func(test_impl, (S,))


def test_series_none_data(memory_leak_check):
    def impl():
        return pd.Series(dtype=np.float64, index=np.arange(7))

    check_func(impl, ())


@pytest.mark.skip("[BE-199] Name unsupported because input type is an array")
def test_series_apply_name(memory_leak_check):
    """
    Check that you can get name information from series.apply.
    """

    def test_impl(S):
        return S.apply(lambda x: x.name)

    S = pd.Series([1, 2, 3, 4, 1])
    check_func(test_impl, (S,))


def test_series_astype_num_constructors(memory_leak_check):
    """
    test Series.astype() with number constructor functions "float" and "int"
    """

    def impl1(A):
        return A.astype(float)

    S = pd.Series(["3.2", "1", "3.2", np.nan, "5.1"])
    check_func(impl1, (S,))

    def impl2(A):
        return A.astype(int)

    S = pd.Series(["3", "1", "-4", "2", "11"])
    check_func(impl2, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.1234, np.nan, 3.31111, 2.1334, 5.1, -6.3], dtype="float32"),
        pd.Series([1, 3, 5, -4, -3]),
    ],
)
@pytest.mark.parametrize("d", [0, 2])
def test_series_round(S, d, memory_leak_check):
    def test_impl(S, d):
        return S.round(d)

    check_func(test_impl, (S, d))


@pytest.mark.slow
def test_series_unsupported_error_checking(memory_leak_check):
    """make sure BodoError is raised for unsupported Series attributes and methods"""
    # test an example attribute
    def test_attr(S):
        return S.nbytes

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_attr)(pd.Series([1, 2]))

    #  test an example method
    def test_method(S):
        return S.to_hdf("data.dat")

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_method)(pd.Series([1, 2]))


def test_series_loc_rm_dead(memory_leak_check):
    """make sure dead code elimination before our SeriesPass doesn't remove loc setitem.
    Related to #2501.
    """

    def impl(df):
        S = pd.Series(index=df.index, data=np.nan, dtype="str")
        S.loc[df.B] = "AA"
        return S

    # test for [BE-93]
    def impl2(A, B):
        s = pd.Series(A).astype("str")
        s.loc[B] = "AAA"
        return s

    df = pd.DataFrame({"A": [1, 3, 4], "B": [True, True, True]})
    check_func(impl, (df,))
    A = np.array([1, 2, 3])
    B = np.array([True, False, True])
    check_func(impl2, (A, B))


@pytest.mark.slow
class TestSeries(unittest.TestCase):
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({"A": [1, 2, 3]})
            return (df.A == 1).sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n)})
            return (df.A == 2).sum()

        n = 11
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({"A": ["a", "b", "c"]})
            return (df.A == "a").sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({"A": A})
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_index1(self):
        def test_impl(A):
            return A

        S = pd.Series([3, 5, 6], ["a", "b", "c"], name="A")
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_attr1(self):
        def test_impl(A):
            return A.size

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_argsort1(self):
        def test_impl(A):
            return A.argsort()

        n = 11
        np.random.seed(0)
        A = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2, 3]).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_astype_str1(self):
        def test_impl(A):
            return A.astype(str)

        n = 11
        S = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_astype_str2(self):
        def test_impl(A):
            return A.astype(str)

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_shape1(self):
        def test_impl(A):
            return A.shape

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A.copy(), 0), test_impl(df.A.copy(), 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series3(self):
        def test_impl(A, i):
            S = pd.Series(A)
            S[i] = 100

        n = 11
        A = np.arange(n)
        A1 = A.copy()
        A2 = A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1, A2)

    def test_setitem_series_bool1(self):
        def test_impl(A):
            A[A > 3] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1)
        test_impl(A2)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series_bool2(self):
        def test_impl(A, B):
            A[A > 3] = B[A > 3]

        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, df.B)
        test_impl(A2, df.B)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_getitem_series_str1(self):
        def test_impl(A, i):
            return A[i]

        df = pd.DataFrame({"A": ["aa", "bb", "cc"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_op1(self):
        def test_impl(A, i):
            return A + A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 0), test_impl(df.A, 0), check_names=False
        )

    def test_series_op2(self):
        def test_impl(A, i):
            return A + i

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op3(self):
        def test_impl(A, i):
            A += i
            return A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A.copy(), 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op4(self):
        def test_impl(A):
            return A.add(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op5(self):
        def test_impl(A):
            return A.pow(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op6(self):
        def test_impl(A, B):
            return A.eq(B)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_op7(self):
        def test_impl(A):
            return -A

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_inplace_binop_array(self):
        def test_impl(A, B):
            A += B
            return A

        n = 11
        A = np.arange(n) ** 2.0  # TODO: use 2 for test int casting
        B = pd.Series(np.ones(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(A.copy(), B), test_impl(A, B))

    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    def test_series_fusion2(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

    def test_series_len(self):
        def test_impl(A, i):
            return len(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(["1", "2", "3"])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_series_list_str_unbox1(self):
        def test_impl(A):
            return A.iloc[0]

        S = pd.Series([["aa", "b"], ["ccc"], []])
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))
        # call twice to test potential refcount errors
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_np_typ_call_replace(self):
        # calltype replacement is tricky for np.typ() calls since variable
        # type can't provide calltype
        def test_impl(i):
            return np.int32(i)

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(1), test_impl(1))

    def test_series_ufunc1(self):
        def test_impl(A, i):
            return np.isinf(A).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A, 1), test_impl(df.A, 1))

    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame(
                {
                    "one": np.array([-1, np.nan, 2.5]),
                    "two": ["foo", "bar", "baz"],
                    "three": [True, False, True],
                }
            )
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(test_impl(), bodo_func(), check_dtype=False)

    @unittest.skip("needs empty_like typing fix in npydecl.py")
    def test_series_empty_like(self):
        def test_impl(A):
            return np.empty_like(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertTrue(isinstance(bodo_func(df.A), np.ndarray))

    def test_series_fillna1(self):
        def test_impl(A):
            return A.fillna(5.0)

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False
        )

    def test_series_fillna_str1(self):
        def test_impl(A):
            return A.fillna("dd")

        df = pd.DataFrame({"A": ["aa", "b", None, "ccc"]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False, check_dtype=False
        )

    def test_series_fillna_str_inplace1(self):
        def test_impl(A):
            A.fillna("dd", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2), check_dtype=False)
        # TODO: handle string array reflection
        # bodo_func(S1)
        # test_impl(S2)
        # np.testing.assert_array_equal(S1, S2)

    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2), check_dtype=False)

    def test_series_dropna_float1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str_parallel1(self):
        def test_impl(A):
            B = A.dropna()
            return (B == "gg").sum()

        S1 = pd.Series(["aa", "b", None, "ccc", "dd", "gg"])
        bodo_func = bodo.jit(distributed_block=["A"])(test_impl)
        start, end = get_start_end(len(S1))
        # TODO: gatherv
        self.assertEqual(bodo_func(S1[start:end]), test_impl(S1))

    def test_series_sum1(self):
        def test_impl(S):
            return S.sum()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 0
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_sum2(self):
        def test_impl(S):
            return (S + S).sum()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_prod1(self):
        def test_impl(S):
            return S.prod()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 1
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_count1(self):
        def test_impl(S):
            return S.count()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(["aa", "bb", np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_mean1(self):
        def test_impl(S):
            return S.mean()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_min1(self):
        def test_impl(S):
            return S.min()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_max1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_value_counts(self):
        def test_impl(S):
            return S.value_counts()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["AA", "BB", "C", "AA", "C", "AA"])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_dist_input1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(distributed_block={"S"})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(bodo_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_tuple_input1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        s_tup = (S, 1, S2)
        self.assertEqual(bodo_func(s_tup), test_impl(s_tup))

    @unittest.skip("pending handling of build_tuple in dist pass")
    def test_series_tuple_input_dist1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(locals={"s_tup:input": "distributed"})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        start, end = get_start_end(n)
        s_tup = (S, 1, S2)
        h_s_tup = (S[start:end], 1, S2[start:end])
        self.assertEqual(bodo_func(h_s_tup), test_impl(s_tup))

    def test_series_concat1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        S2 = pd.Series([6.0, 7.0])
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_concat_str1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2])

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series(["aa", "bb", np.nan, "", "GGG"])
        S2 = pd.Series(["1", "12", "", np.nan, "A"])
        # TODO: handle index in concat
        pd.testing.assert_series_equal(
            bodo_func(S1, S2),
            test_impl(S1, S2),
            check_dtype=False,
        )

    def test_series_cov1(self):
        def test_impl(S1, S2):
            return S1.cov(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_corr1(self):
        def test_impl(S1, S2):
            return S1.corr(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()

        S = pd.Series(["aa", "abc", "c", "cccd"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_str2str(self):
        str2str_methods = (
            "capitalize",
            "lower",
            "lstrip",
            "rstrip",
            "strip",
            "swapcase",
            "title",
            "upper",
        )
        for method in str2str_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            # XXX: \t support pending Numba #4188
            # S = pd.Series([' \tbbCD\t ', 'ABC', ' mCDm\t', 'abc'])
            S = pd.Series([" bbCD ", "ABC", " mCDm ", np.nan, "abc"])
            check_func(test_impl, (S,))

    def test_series_str2bool(self):
        str2bool_methods = (
            "isalnum",
            "isalpha",
            "isdigit",
            "isspace",
            "isupper",
            "islower",
            "istitle",
            "isnumeric",
            "isdecimal",
        )
        for method in str2bool_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            S = pd.Series(
                [" 1aB ", "982", "ABC", "  ", np.nan, "abc", "Hi There", "100.20"]
            )
            check_func(test_impl, (S,))

    def test_series_append1(self):
        def test_impl(S, other):
            return S.append(other).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        # Test single series
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_append2(self):
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        S3 = pd.Series([1.0])
        # Test series tuple
        np.testing.assert_array_equal(bodo_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_isna1(self):
        def test_impl(S):
            return S.isna()

        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        check_func(test_impl, (S,))

    def test_series_isnull1(self):
        def test_impl(S):
            return S.isnull()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_notna1(self):
        def test_impl(S):
            return S.notna()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_str_isna1(self):
        def test_impl(S):
            return S.isna()

        S = pd.Series(["aa", None, "AB", "ABC", "c", "cccd"])
        check_func(test_impl, (S,))

    def test_series_nlargest1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_default1(self):
        def test_impl(S):
            return S.nlargest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_nan1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_nsmallest1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_default1(self):
        def test_impl(S):
            return S.nsmallest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_nan1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_default1(self):
        def test_impl(S):
            return S.head()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_index1(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_head_index2(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], ["a", "ab", "abc", "c", "f", "hh", ""])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_median1(self):
        def test_impl(S):
            return S.median()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))
        # odd size
        m = 101
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_median_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.median()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_series_argsort_parallel(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.argsort().values

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(A):
            return A.idxmin()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_idxmax1(self):
        def test_impl(A):
            return A.idxmax()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values1(self):
        def test_impl(A):
            return A.sort_values()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values_index1(self):
        def test_impl(A, B):
            S = pd.Series(A, B)
            return S.sort_values()

        n = 11
        np.random.seed(0)
        # TODO: support passing Series with Index
        # S = pd.Series(np.random.ranf(n), np.random.randint(0, 100, n))
        A = np.random.ranf(n)
        B = np.random.ranf(n)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_sort_values_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.sort_values()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_series_shift_default1(self):
        def test_impl(S):
            return S.shift()

        # testing for correct default values
        def test_unsup_1(S):
            return S.shift(freq=None, axis=0, fill_value=None)

        # testing when integer default value is unsupported
        def test_unsup_2(S):
            return S.shift(axis=1)

        # testing when nonetype default value is unsupported
        def test_unsup_3(S):
            return S.shift(freq=1)

        bodo_func = bodo.jit(test_impl)
        bodo_func_1 = bodo.jit(test_unsup_1)
        bodo_func_2 = bodo.jit(test_unsup_2)
        bodo_func_3 = bodo.jit(test_unsup_3)
        S = pd.Series([np.nan, 2.0, 3.0, 5.0, np.nan, 6.0, 7.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
        pd.testing.assert_series_equal(test_impl(S), bodo_func_1(S))
        with pytest.raises(BodoError, match="parameter only supports default value"):
            bodo_func_2(S)
        with pytest.raises(BodoError, match="parameter only supports default value"):
            bodo_func_3(S)


if __name__ == "__main__":
    unittest.main()
