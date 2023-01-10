# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.series_common import (  # noqa
    GLOBAL_VAL,
    WhereNullable,
    numeric_series_val,
    series_stat,
    series_val,
)
from bodo.tests.utils import (
    AnalysisTestPipeline,
    _get_dist_arg,
    check_func,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError


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


@pytest.mark.slow
def test_series_map_global_jit(memory_leak_check):
    """Test UDF defined as a global jit function"""

    def test_impl(S):
        return S.map(g2)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


# TODO: add memory_leak_check
@pytest.mark.slow
def test_series_map_tup1():
    def test_impl(S):
        return S.map(lambda a: (a, 2 * a))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
    # TODO: support unbox for column of tuples
    # check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_tup_map1(memory_leak_check):
    def test_impl(S):
        A = S.map(lambda a: (a, 2 * a))
        return A.map(lambda a: a[1])

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_tup_list1(memory_leak_check):
    """test returning a list of tuples from UDF"""

    def test_impl(S):
        A = S.map(lambda a: [(a, 2 * a), (a, 3 * a)])
        return A

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_tup_list2(memory_leak_check):
    """test returning a list of list of tuples from UDF"""

    def test_impl(S):
        A = S.map(
            lambda a: [[(a, 2 * a), (a + 1, 3 * a)], [(3 * a, 4 * a), (a + 2, 7 * a)]]
        )
        return A

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_tup_list3(memory_leak_check):
    """test returning a list of tuples with variable size data from UDF"""

    def test_impl(S):
        A = S.map(lambda a: [(a, "A", 2 * a), (a, "AB", 3 * a)])
        return A

    S = pd.Series([1, 2, 3, 4, 5])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_str(memory_leak_check):
    """test string output in map"""

    def test_impl(S):
        return S.map(lambda a: str(a))

    S = pd.Series([1, 211, 333, 43, 51])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_list_str(memory_leak_check):
    """test list(str) output in map"""

    def test_impl(S):
        return S.map(lambda a: [str(a), "AA"] if a > 200 else ["A"])

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_series_map_array_item(memory_leak_check):
    """test array(item) output in map"""

    def test_impl(S):
        return S.map(lambda a: [a, 3] if a > 200 else [2 * a])

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(test_impl, (S,))


@pytest.mark.slow
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


@pytest.mark.slow
def test_series_map_dict(memory_leak_check):
    """test dict output in map"""

    # homogeneous output
    def impl1(S):
        return S.map(lambda a: {"A": a + 1, "B": a**2})

    # heterogeneous output
    def impl2(S):
        return S.map(lambda a: {"A": a + 1, "B": a**2.0})

    S = pd.Series([1, 211, 333, 43, 51, 12, 15])
    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_series_map_dict_input(memory_leak_check):
    """test dict input in map"""

    def impl(S):
        return S.map(lambda a: a[1])

    S = pd.Series([{1: 1.4, 2: 3.1}, {7: -1.2, 1: 2.2}] * 3)
    check_func(impl, (S,))


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_series_map_arg_fold(memory_leak_check):
    """test handling UDF default value (argument folding)"""

    def test_impl(S):
        return S.map(lambda a, b=1.1: a + b)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.slow
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

    # skip binary, TODO: handle binary min BE-1253
    if isinstance(series_val.values[0], bytes):
        return

    def test_impl(A):
        return A.min()

    check_func(test_impl, (series_val,))


@pytest.mark.slow
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

    # skip binary, TODO: handle binary max BE-1253
    if isinstance(series_val.values[0], bytes):
        return

    def test_impl(A):
        return A.max()

    check_func(test_impl, (series_val,))


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_series_idxmin(series_val, memory_leak_check):

    # Binary not supported in pandas
    if isinstance(series_val.values[0], bytes):
        return

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

    # Binary not supported in pandas
    if isinstance(series_val.values[0], bytes):
        return

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
    "S",
    [
        pytest.param(pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]), marks=pytest.mark.slow),
        pytest.param(pd.Series([2, 3, 5.2, 8, 7]), marks=pytest.mark.slow),
        pytest.param(pd.Series([7, 6, 5, 4, 1]), marks=pytest.mark.slow),
        pytest.param(pd.Series([1, None, 5, None, 1]), marks=pytest.mark.slow),
        pytest.param(pd.Series(["a", "b", "c", "d", "e"]), marks=pytest.mark.slow),
    ],
)
def test_series_infer_objects(S, memory_leak_check):
    def test_impl(S):
        return S.infer_objects()

    check_func(test_impl, (S,))


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


@pytest.mark.slow
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


@pytest.mark.slow
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
        (
            pd.Series([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], name="A"),
            [(0, 1), (2, 1), (4, 5)],
        ),
    ],
)
def test_series_isin(S, values, memory_leak_check):
    def test_impl(S, values):
        return S.isin(values)

    check_func(test_impl, (S, values), check_dtype=False)


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
    # setting check_dtype to False because as bodo returns a series with non-nullable boolean type
    # as opposed to nullable pandas type. See [BE-1162]
    check_func(
        test_impl,
        (series_val, values),
        py_output=py_output,
        dist_test=False,
        check_dtype=False,
    )


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

    check_func(test_impl, (numeric_series_val, k), False, check_dtype=False)


@pytest.mark.slow
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

    check_func(test_impl, (numeric_series_val, k), False, check_dtype=False)


def test_series_nsmallest_non_index(memory_leak_check):
    # test Series with None as Index
    def test_impl(k):
        S = pd.Series([3, 5, 6, 1, 9])
        return S.nsmallest(k)

    bodo_func = bodo.jit(test_impl)
    k = 3
    pd.testing.assert_series_equal(bodo_func(k), test_impl(k))


@pytest.mark.slow
def test_series_take(series_val, memory_leak_check):
    def test_impl(A):
        return A.take([2, 3])

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val),
        test_impl(series_val),
        check_dtype=False,
        check_index_type=False,
        check_categorical=False,
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
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_index_type=False
    )
    # TODO: support distributed argsort()
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_sort_index(series_val, memory_leak_check):
    """test Series.sort_index() on various Series types"""

    def test_impl(A):
        return A.sort_index()

    check_func(test_impl, (series_val,))


def test_series_sort_index_fast(memory_leak_check):
    """test Series.sort_index() on a single Series type for faster CI"""

    def test_impl(A):
        return A.sort_index()

    S = pd.Series([1, 3, 5, 2, -3], index=[5, 3, 6, -4, 9])
    check_func(test_impl, (S,), dist_test=False)


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
@pytest.mark.slow
def test_series_repeat(series_val):
    """Test Series.repeat() method"""

    # TODO: support for binary type, see BE-1320
    if isinstance(series_val.values[0], bytes):
        return

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
        check_index_type=False,
        check_categorical=False,
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
        check_index_type=False,
        check_categorical=False,
    )  # XXX append can't set name yet


@pytest.mark.slow
def test_series_quantile(numeric_series_val, memory_leak_check):

    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        # as of Pandas 1.3, quantile throws an error when called on nullable integer Series
        # In bodo, this doesn't cause an error at the moment.
        py_out = (
            numeric_series_val.dropna()
            .astype(numeric_series_val.dtype.numpy_dtype)
            .quantile(0.30)
        )
    else:
        py_out = None

    def test_impl(A):
        return A.quantile(0.30)

    # TODO: needs np.testing.assert_almost_equal?
    check_func(test_impl, (numeric_series_val,), py_output=py_out)


@pytest.mark.slow
def test_series_quantile_q(memory_leak_check):
    """Tests passing list, int, and unsupported type to q argument"""

    # List
    def test_impl(S):
        ans = S.quantile([0.15, 0.5, 0.65, 1])
        return ans

    S = pd.Series([1.2, 3.4, 4.5, 32.3, 67.8, 100])

    check_func(test_impl, (S,), is_out_distributed=False, atol=1e-4, check_dtype=False)

    # dt64
    S = pd.Series(pd.date_range("2030-01-1", periods=11))
    check_func(test_impl, (S,), is_out_distributed=False, atol=1e-4, check_dtype=False)

    # int
    def test_int(S):
        ans = S.quantile(0)
        return ans

    check_func(test_int, (S,), check_dtype=False)

    def test_str(S):
        ans = S.quantile("aa")
        return ans

    # unsupported q type
    with pytest.raises(BodoError, match=r"Series.quantile\(\) q type must be float"):
        bodo.jit(test_str)(S)


@pytest.mark.slow
def test_series_nunique(series_val, memory_leak_check):
    # Not supported for lists because they aren't hashable
    if isinstance(series_val.values[0], list):
        return

    def test_impl(A):
        return A.nunique()

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_nunique_dropna_false(series_val, memory_leak_check):
    # Not supported for lists because they aren't hashable
    if isinstance(series_val.values[0], list):
        return

    def test_impl(A):
        return A.nunique(dropna=False)

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


@pytest.mark.slow
def test_series_describe(numeric_series_val, memory_leak_check):
    def test_impl(A):
        return A.describe(datetime_is_numeric=True)

    check_func(test_impl, (numeric_series_val,), False)


@pytest.mark.slow
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


@pytest.fixture(
    params=[
        pd.Series(
            [None, None, "a", "b", None, None, "c", "d", None] * 3,
            [3, 4, 2, 1, 6, 5, 7, 8, 9] * 3,
        ),
        pd.Series([2, 3, -4, 5, 1] * 3, dtype=np.int16),
        pd.Series([2, 3, 4, 5, 1] * 3, dtype=np.uint32),
        pd.Series([True, False, True, False, False] * 3),
        pd.Series([None, True, None, False, True, False, None] * 3, dtype="boolean"),
        pd.Series([None, None, 2, -1, None, -3, 4, 5] * 3, dtype=pd.Int64Dtype()),
        pd.Series(
            [None, 1, 2, None, 3, None] * 3,
            [3, 4, 2, 1, 5, 6] * 3,
            dtype=pd.Int8Dtype(),
        ),
        pd.Series([np.nan, 1.0, 2, np.nan, np.nan, 3.0, 4] * 3, dtype=np.float32),
        pd.Series([np.nan, 1.0, -2, np.nan, np.nan, 3.0, 4] * 3, dtype=np.float64),
        pd.Series(
            [np.nan, np.nan, 1e16, 3e17, np.nan, 1e18, 500, np.nan] * 3,
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [np.nan, 3e17, 5e17, np.nan, np.nan, 1e18, 500, np.nan] * 3,
            dtype="timedelta64[ns]",
        ),
        pd.Series([None] * 10 + ["a"], dtype="string"),
        pd.Series([1] + [None] * 10, dtype="Int16"),
        pd.Series([np.nan] * 5 + [1.1] + [np.nan] * 5, dtype="float32"),
        pytest.param(pd.Series([np.nan] * 35, dtype="boolean"), marks=pytest.mark.slow),
    ],
)
def fillna_series(request):
    return request.param


@pytest.mark.parametrize(
    "method",
    [
        "bfill",
        pytest.param("backfill", marks=pytest.mark.slow),
        "ffill",
        pytest.param("pad", marks=pytest.mark.slow),
    ],
)
def test_series_fillna_method(fillna_series, method, memory_leak_check):
    def test_impl(S, method):
        return S.fillna(method=method)

    # Set check_dtype=False because Bodo's unboxing type does not match
    # dtype="string"
    check_func(test_impl, (fillna_series, method), check_dtype=False)


@pytest.mark.parametrize(
    "name,test_impl",
    [
        ("bfill", lambda S: S.bfill()),
        ("backfill", lambda S: S.backfill()),
        ("ffill", lambda S: S.ffill()),
        ("pad", lambda S: S.pad()),
    ],
)
def test_series_fillna_specific_method(
    fillna_series, name, test_impl, memory_leak_check
):
    # Set check_dtype=False because Bodo's unboxing type does not match
    # dtype="string"
    check_func(test_impl, (fillna_series,), check_dtype=False)


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
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

    check_func(test_impl, (S, value), use_dict_encoded_strings=False)


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


def test_series_to_frame(memory_leak_check):
    """test Series.to_frame(). Series name should be known at compile time"""
    # Series name is constant
    def impl1():
        S = pd.Series([1, 2, 3], name="A")
        return S.to_frame()

    # Series name should be None
    def impl2(S):
        return S.to_frame()

    # name is provided
    def impl3(S, name):
        return S.to_frame(name)

    # name is not constant
    def impl4(S, A):
        return S.to_frame(A[0])

    check_func(impl1, (), only_seq=True)
    # name is None case
    S = pd.Series([1, 2, 5, 1, 11, 12])
    check_func(impl2, (S,))
    # name is not compile time constant
    S = pd.Series([1, 2, 5, 1, 11, 12], name="A")
    with pytest.raises(
        BodoError,
        match=r"Series.to_frame\(\): output column name should be known at compile time",
    ):
        bodo.jit(impl2)(S)

    check_func(impl3, (S, "A"))
    with pytest.raises(
        BodoError,
        match=r"Series.to_frame\(\): output column name should be known at compile time",
    ):
        bodo.jit(impl4)(S, ["A", "B"])


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

    # Series.shift supports ints, floats, dt64, nullable int/bool/decimal/date
    # and strings
    bodo_type = bodo.typeof(series_val)
    if bodo.hiframes.rolling.is_supported_shift_array_type(bodo_type.data):
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

    np.random.seed(0)
    check_func(impl, (np.random.rand(500, 50),))


def test_np_argmin(memory_leak_check):
    def impl(A):
        return np.argmin(A, 1)

    np.random.seed(0)
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

    def impl1(S, sort, normalize):
        return S.value_counts(sort=sort, normalize=normalize)

    def impl2(S, sort, normalize):
        return S.value_counts(
            bins=[-1.1, 3.0, 4.5, 7.1], sort=sort, normalize=normalize
        )

    def impl3(S):
        return S.value_counts(bins=[-1, 3, 7])

    def impl4(S):
        return S.value_counts(
            bins=[
                pd.Timestamp("2017-01-01"),
                pd.Timestamp("2017-01-05"),
                pd.Timestamp("2017-01-15"),
                pd.Timestamp("2017-01-21"),
            ]
        )

    def impl5(S):
        return S.value_counts(bins=3)

    S_str = pd.Series(["AA", "BB", "C", "AA", "C", "AA"])
    check_func(impl1, (S_str, True, False))
    check_func(impl1, (S_str, False, True), sort_output=True)
    S_float = pd.Series([1.1, 2.2, 1.3, 4.4, 3.0, 1.7, np.nan, 6.6, 4.3, np.nan])
    check_func(
        impl2, (S_float, True, False), check_dtype=False, is_out_distributed=False
    )
    check_func(
        impl2,
        (S_float, False, True),
        check_dtype=False,
        is_out_distributed=False,
        sort_output=True,
    )
    S_int = pd.Series(pd.array([-1, 4, None, 7, 11, 2, 5, 6, None, 1, -11]))
    check_func(impl3, (S_int,), check_dtype=False, is_out_distributed=False)
    S_dt = pd.Series(
        [
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-01-03"),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-01-03"),
            pd.Timestamp("2017-01-11"),
            pd.Timestamp("2017-01-7"),
        ]
    )
    check_func(impl4, (S_dt,), check_dtype=False, is_out_distributed=False)
    check_func(impl5, (S_int,), check_dtype=False, is_out_distributed=False)
    check_func(impl5, (S_dt,), check_dtype=False, is_out_distributed=False)


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


@pytest.mark.slow
def test_series_np_where_binary(memory_leak_check):
    """Tests np.where() called on Series with binary input (#223)."""

    def test_impl1(S, true_val, false_val):
        # wrapping array in Series to enable output comparison for NA
        return pd.Series(np.where(S == b"aa", true_val, false_val))

    def test_impl2(S):
        # wrapping array in Series to enable output comparison for NA
        return np.where(S == b"aa")

    S1 = pd.Series(
        [b"aa", b"asdfa", b"aa", np.NaN, b"s", b"aa", b"asdgs"] * 2,
    )
    S2 = pd.Series(
        [np.NaN, b"asdga", b"alsdnf", np.NaN, b"aa", b"aa", b"mnbhju"] * 2,
    )

    check_func(test_impl1, (S1, S2, b"nabjhij"))
    check_func(test_impl1, (S1, b"nabjhij", S2))
    check_func(test_impl1, (S1, S1, S2))
    check_func(test_impl2, (S1,))


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


def test_series_where_true_scalar(series_val, memory_leak_check):
    """Tests that all types can be used in Series.where(cond, scalar)
    with all True values."""

    cond = np.array([True] * len(series_val))
    val = series_val.iloc[0]

    def test_impl(S, cond, val):
        return S.where(cond, val)

    # TODO: [BE-110] support series.where for more Bodo array types
    series_err_msg = "Series.where.* Series data with type .* not yet supported"

    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    # Bodo differs from Pandas because Bodo sets the type before
    # it knows that the other value (np.nan) will never be chosen
    check_func(test_impl, (series_val, cond, val), check_dtype=False)


def test_series_where_np_array(series_val, memory_leak_check):
    """Tests that all types can be used in Series.where(cond, ndarray)"""

    np.random.seed(42)
    cond = np.random.randint(2, size=len(series_val)).astype(bool)

    fill_val = series_val.loc[series_val.first_valid_index()]
    # wierd case where, we can have two values at the same index,
    # seems to only occur for categorical.
    if isinstance(fill_val, pd.Series):
        fill_val = fill_val.iloc[0]

    # Having a iterable fill_val causes issues
    if isinstance(fill_val, list):
        fill_val = None
    val = series_val.to_numpy(na_value=fill_val)

    def test_impl(S, cond, val):
        return S.where(cond, val)

    # TODO: [BE-110] support series.where for more Bodo array types
    series_err_msg = "Series.where.* Series data with type .* not yet supported"

    if series_val.dtype.name == "category":
        val = val.astype(series_val.dtype.categories.dtype)

    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    check_func(test_impl, (series_val, cond, val), check_dtype=False)


def test_series_where_series(series_val, memory_leak_check):
    """Tests that all types can be used in Series.where(cond, Series)"""

    np.random.seed(42)
    cond = np.random.randint(2, size=len(series_val)).astype(bool)

    def test_impl(S, cond, val):
        return S.where(cond, val)

    # TODO: [BE-110] support series.where for more Bodo array types
    series_err_msg = "Series.where.* Series data with type .* not yet supported"
    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, series_val)
        return

    # TODO: support series.where with two categorical inputs
    if series_val.dtype.name == "category":
        cat_err_msg = f"Series.where.* 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Series."
        with pytest.raises(BodoError, match=cat_err_msg):
            bodo.jit(test_impl)(series_val, cond, series_val)
        return

    check_func(test_impl, (series_val, cond, series_val), check_dtype=False)


def test_series_mask_np_array(series_val, memory_leak_check):
    """Tests that all types can be used in Series.mask(cond, ndarray)"""
    np.random.seed(42)

    cond = np.random.randint(2, size=len(series_val)).astype(bool)

    fill_val = series_val.loc[series_val.first_valid_index()]
    # wierd case where, we can have two values at the same index,
    # seems to only occur for categorical.
    if isinstance(fill_val, pd.Series):
        fill_val = fill_val.iloc[0]

    # Having a iterable fill_val causes issues
    if isinstance(fill_val, list):
        fill_val = None
    val = series_val.to_numpy(na_value=fill_val)

    def test_impl(S, cond, val):
        return S.mask(cond, val)

    # TODO: [BE-110] support series.mask for more Bodo array types
    series_err_msg = "Series.mask.* Series data with type .* not yet supported"

    if series_val.dtype.name == "category":
        val = val.astype(series_val.dtype.categories.dtype)

    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, val)
        return

    check_func(test_impl, (series_val, cond, val), check_dtype=False)


def test_series_mask_series(series_val, memory_leak_check):
    """Tests that all supported types can be used in Series.mask(cond, Series)."""
    np.random.seed(42)

    cond = np.random.randint(2, size=len(series_val)).astype(bool)

    def test_impl(S, cond, val):
        return S.mask(cond, val)

    # TODO: [BE-110] support series.where for more Bodo array types
    series_err_msg = "Series.mask.* Series data with type .* not yet supported"
    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=series_err_msg):
            bodo.jit(test_impl)(series_val, cond, series_val)
        return

    # TODO: support series.where with two categorical inputs
    if series_val.dtype.name == "category":
        cat_err_msg = f"Series.mask.* 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Series."
        with pytest.raises(BodoError, match=cat_err_msg):
            bodo.jit(test_impl)(series_val, cond, series_val)
        return

    check_func(test_impl, (series_val, cond, series_val), check_dtype=False)


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

    def impl(S, v):
        return S.where(S == "aa", v)

    S = pd.Series(
        ["aa", "b", "aa", None, "s", "aa", "DD"] * 3,
    )
    S2 = pd.Series(
        ["asgdk", "bsadf", "sdaaa", "", "s", None, "DD"] * 3,
    )
    A = np.array(["adsgk", "", "ags", "", "askjdga", None, "asdf"] * 3, dtype=object)

    check_func(impl, (S, "bksd"))
    check_func(impl, (S, S2))
    check_func(impl, (S, A))


def test_series_where_binary(memory_leak_check):
    """Tests Series.where() with binary input"""

    def impl(S, v):
        return S.where(S == b"hello", v)

    S = pd.Series(
        [b"hello", b"b", b"hello world", b"csjk", np.nan, b"aa", b"hello"] * 3,
    )
    S2 = pd.Series(
        [b"hello", b"b", b"hello world", b"csjk", np.nan, b"aa", b"hello"] * 3,
    )
    A = np.array(
        [b"adsgk", b"", b"ags", b"", b"askjdga", None, b"asdf"] * 3, dtype=object
    )

    check_func(impl, (S, b"hlhsha"))
    check_func(impl, (S, S2))
    check_func(impl, (S, A))


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
    check_func(test_impl_nan, (S, cond), check_dtype=False)


def test_series_mask_arr(memory_leak_check):
    """Test for Series.mask(cond, arr) where arr is either
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


def test_series_mask_binary(memory_leak_check):
    """Test for Series.mask(cond, arr) where arr is either
    a string series or array"""

    def test_impl(S, val):
        return S.mask(S != b"hsjldf", val)

    S = pd.Series(
        np.array([np.NaN, bytes(2), b"hsjldf", b"asdgfa", b"nsldgjh"], dtype=object)
    )
    other_series = pd.Series([b"sadkjf", b"asdf", b"nkjhg", np.nan, b"sdlfj"])
    other_arr = np.array([b"sadkjf", b"asdf", b"nkjhg", np.NaN, b"sdlfj"], dtype=object)
    np.random.seed(0)

    check_func(test_impl, (S, b"sasadgk"))
    check_func(test_impl, (S, other_series))
    check_func(test_impl, (S, other_arr))


def test_series_mask_str(memory_leak_check):
    """Test for Series.mask(cond, arr) where arr is either
    a binary series or array"""

    def test_impl(S, val):
        return S.mask(S != "", val)

    S = pd.Series(np.array([None, "", "hello", "hello", "hello world"], dtype=object))
    other_series = pd.Series(["sadkjf", "asdf", "nkjhg", None, "sdlfj"])
    other_arr = np.array(["sadkjf", "asdf", "nkjhg", None, "sdlfj"], dtype=object)
    np.random.seed(0)
    check_func(test_impl, (S, "sasadgk"))
    check_func(test_impl, (S, other_series))
    check_func(test_impl, (S, other_arr))


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


# TODO: add memory_leak_check when memory leaks are fixed
def test_cut():
    """Tests for pd.cut()"""

    def impl(S, bins, include_lowest):
        return pd.cut(S, bins, include_lowest=include_lowest)

    S = pd.Series(
        [-2, 1, 3, 4, 5, 11, 15, 20, 22],
        ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
        name="ABC",
    )
    A = pd.date_range("2017-01-03", "2017-02-11")
    check_func(impl, (S, [1, 5, 20], True), check_dtype=False)
    # categorical types may be different in interval boundaries (int vs float)
    check_func(impl, (S, [1, 5, 20], False), check_dtype=False, check_categorical=False)
    check_func(
        impl, (S, [-4, 6, 29], False), check_dtype=False, check_categorical=False
    )
    check_func(impl, (S, 4, False), check_dtype=False)
    # TODO(ehsan): enable when DatetimeArray is supported [BE-1242]
    # check_func(impl, (A, 4, False), check_dtype=False)


def test_qcut(memory_leak_check):
    """Tests for pd.qcut()"""

    def impl(S, q):
        return pd.qcut(S, q)

    S = pd.Series(
        [-2, 1, 3, 4, 5, 11, 15, 20, 22],
        ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
        name="ABC",
    )
    check_func(impl, (S, 3), check_dtype=False, check_categorical=False)
    check_func(impl, (S, [0.1, 0.4, 0.9]), check_dtype=False, check_categorical=False)
    with pytest.raises(BodoError, match="should be an integer or a list"):
        bodo.jit(impl)(S, "ABC")


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


def test_get_series_index_array_analysis():
    """make sure shape equivalence for get_series_index() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(S):
        B = S.index
        return B

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.Series(np.ones(4), index=[3, 2, 1, 3]))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("S#0") == eq_set._get_ind("B#0")


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
    pd.testing.assert_series_equal(bodo_func(), test_impl(), check_index_type=False)


@pytest.mark.slow
def test_create_series_index2(memory_leak_check):
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"])
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl(), check_index_type=False)


@pytest.mark.slow
def test_create_series_index3(memory_leak_check):
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name="A")
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl(), check_index_type=False)


@pytest.mark.slow
def test_create_series_index4(memory_leak_check):
    def test_impl(name):
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name=name)
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func("A"), test_impl("A"), check_index_type=False
    )


def test_series_var(memory_leak_check):
    def f(S):
        return S.var()

    def f_skipna(S):
        return np.isnan(S.var(skipna=False))

    def f_ddof(S):
        return S.var(ddof=2)

    S = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0])
    check_func(f, (S,))
    check_func(f_skipna, (S,), py_output=True)
    check_func(f_ddof, (S,))
    # Empty Series
    S_empty = pd.Series()
    check_func(f, (S_empty,))


def test_series_sem(memory_leak_check):
    """Test Series.sem"""

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
    # Empty Series
    S_empty = pd.Series()
    check_func(f, (S_empty,))


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
    check_func(f_skipna, (S,), py_output=True)
    check_func(f_ddof, (S,))
    # Empty Series
    S_empty = pd.Series()
    check_func(f, (S_empty,))


def test_series_std_dt64(memory_leak_check):
    def f():
        S = pd.Series(pd.date_range("2017-01-03", periods=11))
        return S.std()

    # Bodo differs in micro/nano seconds accuracy from Pandas
    # bodo_out: 3 days 07:35:56.381537975
    # pandas_out: 3 days 07:35:56.381886706
    # TODO: [BE-765]
    # check_func(f, (S,))
    bodo_out = bodo.jit(f)()
    py_out = f()
    assert abs(bodo_out - py_out) < datetime.timedelta(milliseconds=1)


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
        pd.Series([0, -123, 12, None, 4], dtype="Int64"),
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
        return S.axes

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
def test_datetime_series_mean(memory_leak_check):
    """
    Test Series.mean() with a datetime Series.
    """
    S = pd.Series(pd.date_range("2030-01-1", periods=11))

    def impl_mean(S):
        return S.mean()

    check_func(impl_mean, (S,))


@pytest.mark.slow
def test_series_mem_usage(memory_leak_check):
    """Test Series.memory_usage() with and w/o index"""

    def impl1(S):
        return S.memory_usage()

    def impl2(S):
        return S.memory_usage(index=False)

    S = pd.Series([1, 2, 3, 4, 5, 6], index=pd.Int64Index([10, 12, 24, 30, -10, 40]))
    py_out = 96
    check_func(impl1, (S,), py_output=py_out)
    py_out = 48
    check_func(impl2, (S,), py_output=py_out)

    # Empty Series
    S = pd.Series()
    # StringIndex only. Bodo has different underlying arrays than Pandas
    # Test sequential case
    py_out = 8
    check_func(impl1, (S,), only_seq=True, py_output=py_out, is_out_distributed=False)
    # Test parallel case. Index is replicated across ranks
    py_out = 8 * bodo.get_size()
    check_func(impl1, (S,), only_1D=True, py_output=py_out, is_out_distributed=False)

    # Empty and no index.
    check_func(impl2, (S,), is_out_distributed=False)


@pytest.mark.slow
def test_series_nbytes(memory_leak_check):
    def impl(S):
        return S.nbytes

    S = pd.Series([1, 2, 3, 4, 5, 6])
    check_func(impl, (S,))


# helper function, that returns if a particular series can be used with np.where
def is_where_mask_supported_series(S):
    if isinstance(S.values[0], list):
        return False
    if isinstance(S.values[0], Decimal):
        return False
    if isinstance(S.dtype, pd.CategoricalDtype) and isinstance(
        S.dtype.categories[0], (pd.Timestamp, pd.Timedelta)
    ):
        return False
    if not isinstance(S.dtype, pd.CategoricalDtype) and isinstance(
        S.values[0], datetime.date
    ):
        return False
    return True


def test_series_np_select(series_val, memory_leak_check):
    """tests np select for nullable series"""
    np.random.seed(42)

    cond1 = np.random.randint(2, size=len(series_val)).astype(bool)
    cond2 = np.random.randint(2, size=len(series_val)).astype(bool)

    A1 = series_val
    # Some trivial sorting, to insure we have different values at each index
    # So we can catch correctness issues
    A2 = series_val.sort_values(ignore_index=True)

    def impl1(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl2(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    def impl3(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl4(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    err_msg = r".*np.select\(\): data with choicelist of type .* not yet supported.*"
    if not is_where_mask_supported_series(series_val):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(impl1)(A1, A2, cond1, cond2)
        return

    err_msg_cat = (
        r".*np.select\(\): data with choicelist of type Categorical not yet supported.*"
    )
    if series_val.dtype.name == "category":
        with pytest.raises(BodoError, match=err_msg_cat):
            bodo.jit(impl1)(A1, A2, cond1, cond2)
        return

    # for numeric/bool, we default to 0/false. This is to keep the expected behavior of np select
    # in situations that a user might resonably want/expect to have the default set to 0.
    # for all other types, we default to NA, for type stability.
    def na_impl(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default=pd.NA)

    from numba.core import types

    infered_typ = bodo.hiframes.boxing._infer_series_arr_type(series_val).dtype
    if not (infered_typ == bodo.bool_ or isinstance(infered_typ, types.Number)):
        py_out = na_impl(A1, A2, cond1, cond2)
        if infered_typ == bodo.datetime64ns:
            # need to do a bit of conversion in this case, as numpy by default casts the dt64 to int
            # when the np output is an object array
            py_out = np.array(pd.Series(py_out).astype("datetime64[ns]"))
        if infered_typ == bodo.timedelta64ns:
            # need to do a bit of conversion in this case, numpy does a cast to int
            # when the np output is an object array
            py_out = np.array(pd.Series(py_out).astype("timedelta64[ns]"))
    else:
        py_out = None

    for impl in [impl1, impl2, impl3, impl4]:
        check_func(
            impl,
            (A1, A2, cond1, cond2),
            check_dtype=False,
            py_output=py_out,
        )


def test_series_np_select_non_unitype(series_val, memory_leak_check):
    """tests np select when passed a non unitype choicelist"""
    np.random.seed(42)

    cond1 = np.random.randint(2, size=len(series_val)).astype(bool)
    cond2 = np.random.randint(2, size=len(series_val)).astype(bool)
    A1 = series_val
    # Some trivial sorting, to insure we have different values at each index
    # So we can catch correctness issues
    sorted_series = series_val.sort_values(ignore_index=True)
    fill_val = sorted_series.loc[sorted_series.first_valid_index()]
    # na_value doesn't play nice with iterables, so just use the nullable array type
    if isinstance(fill_val, list):
        fill_val = None
    A2 = sorted_series.to_numpy(na_value=fill_val)

    def impl(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    # Already checked error messages in test_series_np_select
    if (
        not is_where_mask_supported_series(series_val)
        or series_val.dtype.name == "category"
    ):
        pytest.skip()

    # for numeric/bool, we default to 0/false. This is to keep the expected behavior of np select
    # in situations that a user might resonably want/expect to have the default set to 0.
    # for all other types, we default to NA, for type stability.
    def na_impl(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist, default=pd.NA)

    from numba.core import types

    infered_typ = bodo.hiframes.boxing._infer_series_arr_type(series_val).dtype
    if not (infered_typ == bodo.bool_ or isinstance(infered_typ, types.Number)):
        py_out = na_impl(A1, A2, cond1, cond2)

        if infered_typ == bodo.datetime64ns:
            # need to do a bit of conversion in this case, as numpy by default casts the dt64 to int
            py_out = np.array(pd.Series(py_out).astype("datetime64[ns]"))
        if infered_typ == bodo.timedelta64ns:
            # need to do a bit of conversion in this case, again, numpy does a wierd conversion
            py_out = np.array(pd.Series(py_out).astype("timedelta64[ns]"))
        if isinstance(infered_typ, bodo.PDCategoricalDtype):
            if isinstance(
                series_val.dtype.categories, (pd.TimedeltaIndex, pd.DatetimeIndex)
            ):
                py_out = pd.array(
                    pd.Series(py_out)
                    .astype(series_val.dtype.categories.dtype)
                    .astype(series_val.dtype)
                )
            else:
                py_out = pd.array(pd.Series(py_out).astype(series_val.dtype))
    else:
        py_out = None

    check_func(
        impl,
        (A1, A2, cond1, cond2),
        check_dtype=False,
        py_output=py_out,
    )


def test_series_np_select_non_unitype_none_default(series_val, memory_leak_check):
    """tests np select when passed a non unitype choicelist"""
    np.random.seed(42)

    cond1 = np.random.randint(2, size=len(series_val)).astype(bool)
    cond2 = np.random.randint(2, size=len(series_val)).astype(bool)
    A1 = series_val
    # Some trivial sorting, to insure we have different values at each index
    # So we can catch correctness issues
    sorted_series = series_val.sort_values(ignore_index=True)
    fill_val = sorted_series.loc[sorted_series.first_valid_index()]
    # na_value doesn't play nice with iterables, so just use the nullable array type
    if isinstance(fill_val, list):
        fill_val = None
    A2 = sorted_series.to_numpy(na_value=fill_val)

    def impl(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist, None)

    # Already checked error messages in test_series_np_select
    if (
        not is_where_mask_supported_series(series_val)
        or series_val.dtype.name == "category"
    ):
        pytest.skip()

    # for numeric/bool, we default to 0/false. This is to keep the expected behavior of np select
    # in situations that a user might resonably want/expect to have the default set to 0.
    # for all other types, we default to NA, for type stability.
    def na_impl(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist, default=pd.NA)

    if series_val.dtype.name.startswith("float"):
        py_out = impl(A1, A2, cond1, cond2)
        py_out[pd.isna(py_out)] = np.NAN
        py_out = py_out.astype(float)
    else:
        py_out = None

    check_func(
        impl,
        (A1, A2, cond1, cond2),
        check_dtype=False,
        py_output=py_out,
    )


def test_series_np_select_non_unitype_set_default(series_val, memory_leak_check):
    """tests np select when passed a non unitype choicelist"""
    np.random.seed(42)

    cond1 = np.random.randint(2, size=len(series_val)).astype(bool)
    cond2 = np.random.randint(2, size=len(series_val)).astype(bool)
    A1 = series_val
    # Some trivial sorting, to insure we have different values at each index
    # So we can catch correctness issues
    sorted_series = series_val.sort_values(ignore_index=True)
    orig_fill_val = sorted_series.loc[sorted_series.first_valid_index()]
    # na_value doesn't play nice with iterables, so just use the nullable array type
    if isinstance(orig_fill_val, list):
        fill_val = None
    else:
        fill_val = orig_fill_val
    A2 = sorted_series.to_numpy(na_value=fill_val)

    def impl(A1, A2, cond1, cond2, default):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist, default=default)

    # Already checked error messages in test_series_np_select
    if (
        not is_where_mask_supported_series(series_val)
        or series_val.dtype.name == "category"
    ):
        pytest.skip()

    if isinstance(orig_fill_val, pd.Timestamp):
        # setitem for array(datetime64[ns], 1d, C) with timestamp not supported
        default_val = orig_fill_val.to_datetime64()
    elif isinstance(orig_fill_val, pd.Timedelta):
        # setitem for array(timedelta64[ns], 1d, C) with pd timedelta not supported
        default_val = orig_fill_val.to_timedelta64()
    else:
        default_val = orig_fill_val

    check_func(
        impl,
        (A1, A2, cond1, cond2, default_val),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"),
        pd.Series([1.0, 2.0, np.nan, 1.0, np.nan], name="A"),
    ],
)
def test_series_keys(S, memory_leak_check):
    def test_impl(A):
        return A.keys()

    check_func(test_impl, (S,))
