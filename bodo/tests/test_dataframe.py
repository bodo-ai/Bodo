# Copyright (C) 2019 Bodo Inc. All rights reserved.
import unittest
import os
import sys
import random
import string
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    check_func,
    is_bool_object_series,
)


# TODO: other possible df types like Categorical, dt64, td64, ...
@pytest.fixture(
    params=[
        # int and float columns
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 8, 4, 11, -3],
                    "B": [1.1, np.nan, 4.2, 3.1, -1.3],
                    "C": [True, False, False, True, True],
                }
            ),
            marks=pytest.mark.slow,
        ),
        pd.DataFrame(
            {
                "A": pd.Series([1, 8, 4, 10, 3], dtype="Int32"),
                "B": [1.1, np.nan, 4.2, 3.1, -1.3],
                "C": [True, False, False, np.nan, True],
            },
            ["A", "BA", "", "DD", "C"],
        ),
        # uint8, float32 dtypes
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.array([1, 8, 4, 0, 3], dtype=np.uint8),
                    "B": np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # string and int columns, float index
        pytest.param(
            pd.DataFrame(
                {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
                [-2.1, 0.1, 1.1, 7.1, 9.0],
            ),
            marks=pytest.mark.slow,
        ),
        # range index
        pytest.param(
            pd.DataFrame(
                {"A": [1, 8, 4, 1, -2], "B": ["A", "B", "CG", "ACDE", "C"]},
                range(0, 5, 1),
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: parallel range index with start != 0 and stop != 1
        # int index
        pd.DataFrame(
            {"A": [1, 8, 4, 1, -3], "B": ["A", "B", "CG", "ACDE", "C"]},
            [-2, 1, 3, 5, 9],
        ),
        # string index
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3, -1, 4]}, ["A", "BA", "", "DD", "C"]),
            marks=pytest.mark.slow,
        ),
        # datetime column
        pd.DataFrame(
            {"A": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)}
        ),
        # datetime index
        pytest.param(
            pd.DataFrame(
                {"A": [3, 5, 1, -1, 4]},
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: timedelta
    ]
)
def df_value(request):
    return request.param


@pytest.fixture(
    params=[
        # int
        pytest.param(pd.DataFrame({"A": [1, 8, 4, 11, -3]}), marks=pytest.mark.slow),
        # int and float columns
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, 11, -3], "B": [1.1, np.nan, 4.2, 3.1, -1.1]}),
            marks=pytest.mark.slow,
        ),
        # uint8, float32 dtypes
        pd.DataFrame(
            {
                "A": np.array([1, 8, 4, 0, 2], dtype=np.uint8),
                "B": np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
            }
        ),
        # pd.DataFrame({'A': np.array([1, 8, 4, 0], dtype=np.uint8),
        # }),
        # int column, float index
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, -1, 3]}, [-2.1, 0.1, 1.1, 7.1, 9.0]),
            marks=pytest.mark.slow,
        ),
        # range index
        pytest.param(
            pd.DataFrame({"A": [1, 8, 4, 1, -2]}, range(0, 5, 1)),
            marks=pytest.mark.slow,
        ),
        # datetime column
        pd.DataFrame(
            {"A": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)}
        ),
        # datetime index
        pytest.param(
            pd.DataFrame(
                {"A": [3, 5, 1, -1, 2]},
                pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: timedelta
    ]
)
def numeric_df_value(request):
    return request.param


@pytest.fixture(
    params=[
        # column name overlaps with pandas function
        pd.DataFrame({"product": ["a", "b", "c"]}),
        pd.DataFrame({"product": ["a", "b", "c"], "keys": [1, 2, 3]}),
    ]
)
def column_name_df_value(request):
    return request.param


def test_unbox_df1(df_value):
    # just unbox
    def impl(df_arg):
        return True

    check_func(impl, (df_value,))

    # unbox and box
    def impl2(df_arg):
        return df_arg

    check_func(impl2, (df_value,))

    # unbox and return Series data with index
    # (previous test can box Index unintentionally)
    def impl3(df_arg):
        return df_arg.A

    check_func(impl3, (df_value,))


def test_unbox_df2(column_name_df_value):
    # unbox column with name overlaps with pandas function
    def impl1(df_arg):
        return df_arg["product"]

    check_func(impl1, (column_name_df_value,))


def test_box_df():
    # box dataframe contains column with name overlaps with pandas function
    def impl():
        df = pd.DataFrame({"product": ["a", "b", "c"], "keys": [1, 2, 3]})
        return df

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


def test_df_index(df_value):
    def impl(df):
        return df.index

    check_func(impl, (df_value,))


def test_df_index_non():
    # test None index created inside the function
    def impl():
        df = pd.DataFrame({"A": [2, 3, 1]})
        return df.index

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_df_columns(df_value):
    def impl(df):
        return df.columns

    check_func(impl, (df_value,), False)


def test_df_values(numeric_df_value):
    def impl(df):
        return df.values

    check_func(impl, (numeric_df_value,))


def test_df_get_values(numeric_df_value):
    def impl(df):
        return df.get_values()

    check_func(impl, (numeric_df_value,))


def test_df_to_numpy(numeric_df_value):
    def impl(df):
        return df.to_numpy()

    check_func(impl, (numeric_df_value,))


def test_df_ndim(df_value):
    def impl(df):
        return df.ndim

    check_func(impl, (df_value,))


def test_df_size(df_value):
    def impl(df):
        return df.size

    check_func(impl, (df_value,))


def test_df_shape(df_value):
    def impl(df):
        return df.shape

    check_func(impl, (df_value,))


# TODO: empty df: pd.DataFrame()
@pytest.mark.parametrize("df", [pd.DataFrame({"A": [1, 3]}), pd.DataFrame({"A": []})])
def test_df_empty(df):
    def impl(df):
        return df.empty

    bodo_func = bodo.jit(impl)
    assert bodo_func(df) == impl(df)


def test_df_astype_num(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.astype(np.float32)

    check_func(impl, (numeric_df_value,))


def test_df_astype_str(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    # XXX str(float) not consistent with Python yet
    if any(
        d == np.dtype("float64") or d == np.dtype("float32")
        for d in numeric_df_value.dtypes
    ):
        return

    def impl(df):
        return df.astype(str)

    check_func(impl, (numeric_df_value,))


def test_df_copy_deep(df_value):
    def impl(df):
        return df.copy()

    check_func(impl, (df_value,))


def test_df_copy_shallow(df_value):
    def impl(df):
        return df.copy(deep=False)

    check_func(impl, (df_value,))


def test_df_rename():
    def impl(df):
        return df.rename(columns={"B": "bb", "C": "cc"})

    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": [True, False, False, True, True],
        }
    )
    check_func(impl, (df,))


def test_df_isna(df_value):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.isna()

    check_func(impl, (df_value,))


def test_df_notna(df_value):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.notna()

    check_func(impl, (df_value,))


def test_df_head(df_value):
    def impl(df):
        return df.head(3)

    check_func(impl, (df_value,), False)


def test_df_tail(df_value):
    def impl(df):
        return df.tail(3)

    check_func(impl, (df_value,), False)


@pytest.mark.parametrize(
    "other", [pd.DataFrame({"A": np.arange(5), "C": np.arange(5) ** 2}), [2, 3, 4, 5]]
)
def test_df_isin(other):
    # TODO: more tests, other data types
    # TODO: Series and dictionary values cases
    def impl(df, other):
        return df.isin(other)

    df = pd.DataFrame({"A": np.arange(5), "B": np.arange(5) ** 2})
    check_func(impl, (df, other))


def test_df_abs(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.abs()

    check_func(impl, (numeric_df_value,))


def test_df_corr(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # XXX pandas excludes bool columns with NAs, which we can't do dynamically
    for c in df_value.columns:
        if is_bool_object_series(df_value[c]) and df_value[c].hasnans:
            return

    def impl(df):
        return df.corr()

    check_func(impl, (df_value,), False)


def test_df_corr_parallel():
    def impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        return df.corr()

    bodo_func = bodo.jit(impl)
    n = 11
    pd.testing.assert_frame_equal(bodo_func(n), impl(n))
    assert count_array_OneDs() >= 3
    assert count_parfor_OneDs() >= 1


def test_df_cov(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # XXX pandas excludes bool columns with NAs, which we can't do dynamically
    for c in df_value.columns:
        if is_bool_object_series(df_value[c]) and df_value[c].hasnans:
            return

    def impl(df):
        return df.cov()

    check_func(impl, (df_value,), False)


def test_df_count(df_value):
    def impl(df):
        return df.count()

    check_func(impl, (df_value,), False)


def test_df_prod(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.prod()

    check_func(impl, (df_value,), False)


def test_df_sum(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.sum()

    check_func(impl, (numeric_df_value,), False)


def test_df_min(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.min()

    check_func(impl, (numeric_df_value,), False)


def test_df_max(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.max()

    check_func(impl, (numeric_df_value,), False)


def test_df_mean(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.mean()

    check_func(impl, (numeric_df_value,), False)


def test_df_var(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.var()

    check_func(impl, (numeric_df_value,), False)


def test_df_std(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.std()

    check_func(impl, (numeric_df_value,), False)


def test_df_median(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    # skip NAs
    # TODO: handle NAs
    if numeric_df_value._get_numeric_data().isna().sum().sum():
        return

    def impl(df):
        return df.median()

    check_func(impl, (numeric_df_value,), False)


def test_df_quantile(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # pandas returns object Series for some reason when input has IntegerArray
    if isinstance(df_value.A.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def impl(df):
        return df.quantile(0.3)

    check_func(impl, (df_value,), False, check_names=False)


def test_df_pct_change(numeric_df_value):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def test_impl(df):
        return df.pct_change(2)

    check_func(test_impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_describe(numeric_df_value):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def test_impl(df):
        return df.describe()

    check_func(test_impl, (numeric_df_value,), False)


@pytest.mark.skip(reason="distributed cumprod not available yet")
def test_df_cumprod(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    # skip NAs
    # TODO: handle NAs
    if numeric_df_value._get_numeric_data().isna().sum().sum():
        return

    def impl(df):
        return df.cumprod()

    check_func(impl, (numeric_df_value,))


def test_df_cumsum(numeric_df_value):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    # skip NAs
    # TODO: handle NAs
    if numeric_df_value._get_numeric_data().isna().sum().sum():
        return

    def impl(df):
        return df.cumsum()

    check_func(impl, (numeric_df_value,))


def test_df_nunique(df_value):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in df_value.dtypes):
        return

    # skip NAs
    # TODO: handle NAs
    if df_value.isna().sum().sum():
        return

    def impl(df):
        return df.nunique()

    # TODO: make sure output is REP
    check_func(impl, (df_value,), False)


def _is_supported_argminmax_typ(d):
    # distributed argmax types, see distributed_api.py
    supported_typs = [np.int32, np.float32, np.float64]
    if not sys.platform.startswith("win"):
        # long is 4 byte on Windows
        supported_typs.append(np.int64)
        supported_typs.append(np.dtype("datetime64[ns]"))
    return d in supported_typs


def test_df_idxmax(numeric_df_value):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmax()

    check_func(impl, (numeric_df_value,), False)


def test_df_idxmin(numeric_df_value):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmin()

    check_func(impl, (numeric_df_value,), False)


def test_df_take(df_value):
    def impl(df):
        return df.take([1, 3])

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(df_value), impl(df_value), check_dtype=False
    )


def test_df_sort_values(df_value):
    # skip NAs
    # TODO: handle NA order
    if df_value.isna().sum().sum():
        return

    def impl(df):
        return df.sort_values(by="A")

    check_func(impl, (df_value,))


def test_df_sort_index(df_value):
    # skip NAs
    # TODO: handle NA order
    if df_value.isna().sum().sum():
        return

    def impl(df):
        return df.sort_index()

    check_func(impl, (df_value,))


def test_df_shift(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.shift(2)

    check_func(impl, (numeric_df_value,))


def test_df_set_index(df_value):
    # singe column dfs become zero column which are not supported, TODO: fix
    if len(df_value.columns) < 2:
        return

    # TODO: fix nullable int
    if isinstance(df_value.A.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def impl(df):
        return df.set_index("A")

    check_func(impl, (df_value,))


def test_df_duplicated():
    def impl(df):
        return df.duplicated()

    df = pd.DataFrame({"A": ["A", "B", "A", "B", "C"], "B": ["F", "E", "F", "S", "C"]})
    check_func(impl, (df,), sort_output=True)
    df = pd.DataFrame(
        {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]}, index=[3, 1, 2, 4, 6]
    )
    check_func(impl, (df,), sort_output=True)


@pytest.mark.parametrize(
    "expr",
    [
        "`B B` > @a + 1 & 5 > index > 1",
        "(A == @a) | (C == 'AA')",
        "C in ['AA', 'C']",
        "C not in ['AA', 'C']",
        "C.str.contains('C')",
        "abs(A) > @a",
        "A in [1, 4]",
        "A not in [1, 4]",
    ],
)
def test_df_query(expr):
    def impl(df, expr, a):
        return df.query(expr)

    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": ["AA", "BBB", "C", "AA", "C"],
        },
        index=[3, 1, 2, 4, 5],
    )
    check_func(impl, (df, expr, 1))


@pytest.mark.parametrize(
    "test_df",
    [
        # all string
        pd.DataFrame({"A": ["A", "B", "A", "B", "C"], "B": ["F", "E", "F", "S", "C"]}),
        # mix string and numbers and index
        pd.DataFrame(
            {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]},
            index=[3, 1, 2, 4, 6],
        ),
        # string index
        pd.DataFrame(
            {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]},
            index=["A", "C", "D", "E", "AA"],
        ),
        # all numbers
        pd.DataFrame(
            {"A": [1, 3, 1, 2, 3], "B": [1.2, 3.1, 1.2, 2.5, 3.3]},
            index=[3, 1, 2, 4, 6],
        ),
    ],
)
def test_df_drop_duplicates(test_df):
    def impl(df):
        return df.drop_duplicates()

    check_func(impl, (test_df,), sort_output=True)


def _gen_df_str(n):
    str_vals = []
    for _ in range(n):
        # store NA with 30% chance
        if random.random() < 0.3:
            str_vals.append(np.nan)
            continue

        k = random.randint(1, 10)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
        str_vals.append(val)

    A = np.random.randint(0, 1000, n)
    df = pd.DataFrame({"A": A, "B": str_vals}).drop_duplicates("A")
    return df


def test_sort_values_str():
    def test_impl(df):
        return df.sort_values(by="A")

    # seeds should be the same on different processors for consistent input
    random.seed(2)
    np.random.seed(3)
    n = 17  # 1211
    df = _gen_df_str(n)
    check_func(test_impl, (df,))


##################### binary ops ###############################


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_binary_ops)
def test_dataframe_binary_op(op):
    # TODO: test parallelism
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(df, other):\n"
    func_text += "  return df {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    df = pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=[3, 5, 0, 7, 2])
    # df/df
    check_func(test_impl, (df, df))
    # df/scalar
    check_func(test_impl, (df, 2))
    check_func(test_impl, (2, df))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_inplace_binary_ops)
def test_dataframe_inplace_binary_op(op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(df, other):\n"
    func_text += "  df {} other\n".format(op_str)
    func_text += "  return df\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    df = pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=[3, 5, 0, 7, 2])
    check_func(test_impl, (df, df), copy_input=True)
    check_func(test_impl, (df, 2), copy_input=True)


@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_unary_ops)
def test_dataframe_unary_op(op):
    # TODO: fix operator.pos
    import operator

    if op == operator.pos:
        return

    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(df):\n"
    func_text += "  return {} df\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    df = pd.DataFrame({"A": [4, 6, 7, 1, -3]}, index=[3, 5, 0, 7, 2])
    check_func(test_impl, (df,))


@pytest.fixture(
    params=[
        # array-like
        [2, 3, 5],
        [2.1, 3.2, 5.4],
        ["A", "C", "AB"],
        np.array([2, 3, 5, -1, -4]),
        pd.Series([2.1, 5.3, 6.1, -1.0, -3.7], [3, 5, 6, -2, 4], name="C"),
        pd.Int64Index([10, 12, 14, 17, 19], name="A"),
        pd.RangeIndex(5),
        # dataframe
        pd.DataFrame(
            {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
            [1.1, -2.1, 7.1, 0.1, 3.1],
        ),
        # scalars
        3,
        1.3,
        np.nan,
        "ABC",
    ]
)
def na_test_obj(request):
    return request.param


def test_pd_isna(na_test_obj):
    obj = na_test_obj

    def impl(obj):
        return pd.isna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


def test_pd_notna(na_test_obj):
    obj = na_test_obj

    def impl(obj):
        return pd.notna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


def test_set_column_cond1():
    # df created inside function case
    def test_impl(n, cond):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df.A

    bodo_func = bodo.jit(test_impl)
    n = 11
    pd.testing.assert_series_equal(bodo_func(n, True), test_impl(n, True))
    pd.testing.assert_series_equal(bodo_func(n, False), test_impl(n, False))


def test_set_column_cond2():
    # df is assigned to other variable case (mutability)
    def test_impl(n, cond):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df2 = df
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df2  # df2.A, TODO: pending set_dataframe_data() analysis fix
        # to avoid incorrect optimization

    bodo_func = bodo.jit(test_impl)
    n = 11
    pd.testing.assert_frame_equal(bodo_func(n, True), test_impl(n, True))
    pd.testing.assert_frame_equal(bodo_func(n, False), test_impl(n, False))


def test_set_column_cond3():
    # df is assigned to other variable case (mutability) and has parent
    def test_impl(df, cond):
        df2 = df
        # df2['A'] = np.arange(n) + 1.0, TODO: make set column inplace
        # when there is another reference
        if cond:
            df["A"] = np.arange(n) + 2.0
        return df2  # df2.A, TODO: pending set_dataframe_data() analysis fix
        # to avoid incorrect optimization

    bodo_func = bodo.jit(test_impl)
    n = 11
    df1 = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
    df2 = df1.copy()
    pd.testing.assert_frame_equal(bodo_func(df1, True), test_impl(df2, True))
    pd.testing.assert_frame_equal(df1, df2)
    df1 = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
    df2 = df1.copy()
    pd.testing.assert_frame_equal(bodo_func(df1, False), test_impl(df2, False))
    pd.testing.assert_frame_equal(df1, df2)


def test_df_filter():
    def test_impl(df, cond):
        df2 = df[cond]
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    cond = df.A > 1
    check_func(test_impl, (df, cond))


def test_create_series_input1():
    def test_impl(S):
        df = pd.DataFrame({"A": S})
        return df

    bodo_func = bodo.jit(test_impl)
    S = pd.Series([2, 4], [3, -1])
    pd.testing.assert_frame_equal(bodo_func(S), test_impl(S))


def test_df_apply_bool():
    # check bool output of UDF for BooleanArray use
    def test_impl(df):
        return df.apply(lambda r: r.A == 2, axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df,))


def test_df_apply_str():
    """make sure string output can be handled in apply() properly
    """

    def test_impl(df):
        return df.apply(lambda r: r.A if r.A == "AA" else "BB", axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_drop_inplace_branch():
    def test_impl(cond):
        if cond:
            df = pd.DataFrame({"A": [2, 3, 4], "B": [1, 2, 6]})
        else:
            df = pd.DataFrame({"A": [5, 6, 7], "B": [1, 0, -6]})
        df.drop("B", axis=1, inplace=True)
        return df

    check_func(test_impl, (True,), False)


from numba.compiler_machinery import FunctionPass, register_pass


@register_pass(analysis_only=False, mutates_CFG=True)
class ArrayAnalysisPass(FunctionPass):
    _name = "array_analysis_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        array_analysis = numba.array_analysis.ArrayAnalysis(
            state.typingctx,
            state.func_ir,
            state.type_annotation.typemap,
            state.type_annotation.calltypes,
        )
        array_analysis.run(state.func_ir.blocks)
        state.func_ir._definitions = numba.ir_utils.build_definitions(
            state.func_ir.blocks
        )
        state.metadata["preserved_array_analysis"] = array_analysis
        return False


class AnalysisTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeleine used in test_dataframe_array_analysis()
    additional ArrayAnalysis pass that preseves analysis object
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(True)
        pipeline._finalized = False
        pipeline.add_pass_after(ArrayAnalysisPass, bodo.compiler.BodoSeriesPass)
        pipeline.finalize()
        return [pipeline]


def test_init_dataframe_array_analysis():
    """make sure shape equivalence for init_dataframe() is applied correctly
    """
    import numba.tests.test_array_analysis

    def impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        return df

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline)(impl)
    test_func(10)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("n")


def test_get_dataframe_data_array_analysis():
    """make sure shape equivalence for get_dataframe_data() is applied correctly
    """
    import numba.tests.test_array_analysis

    def impl(df):
        B = df.A.values
        return B

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline)(impl)
    test_func(pd.DataFrame({"A": np.ones(10), "B": np.arange(10)}))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("B#0")


############################# old tests ###############################


@bodo.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


class TestDataFrame(unittest.TestCase):
    def test_create1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_kws1(self):
        def test_impl(n):
            df = pd.DataFrame(data={"A": np.ones(n), "B": np.random.ranf(n)})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_dtype1(self):
        def test_impl(n):
            df = pd.DataFrame(
                data={"A": np.ones(n), "B": np.random.ranf(n)}, dtype=np.int8
            )
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_column1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)}, columns=["B"])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_column2(self):
        # column arg uses list('AB')
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)}, columns=list("AB"))
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_range_index1(self):
        def test_impl(n):
            df = pd.DataFrame(
                {"A": np.zeros(n), "B": np.ones(n)},
                index=range(0, n),
                columns=["A", "B"],
            )
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_ndarray1(self):
        def test_impl(n):
            # TODO: fix in Numba
            # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            data = np.arange(9).reshape(3, 3)
            df = pd.DataFrame(data, columns=["a", "b", "c"])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_ndarray_copy1(self):
        def test_impl(data):
            # TODO: fix in Numba
            # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            df = pd.DataFrame(data, columns=["a", "b", "c"], copy=True)
            data[0] = 6
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        data = np.arange(9).reshape(3, 3)
        pd.testing.assert_frame_equal(bodo_func(data.copy()), test_impl(data.copy()))

    def test_create_empty_column1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)}, columns=["B", "C"])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        # bodo uses np.nan for empty columns currently but Pandas uses objects
        df1 = bodo_func(n)
        df2 = test_impl(n)
        df2["C"] = df2.C.astype(np.float64)
        pd.testing.assert_frame_equal(df1, df2)

    def test_create_cond1(self):
        def test_impl(A, B, c):
            if c:
                df = pd.DataFrame({"A": A})
            else:
                df = pd.DataFrame({"A": B})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        A = np.ones(n)
        B = np.arange(n) + 1.0
        c = 0
        pd.testing.assert_series_equal(bodo_func(A, B, c), test_impl(A, B, c))
        c = 2
        pd.testing.assert_series_equal(bodo_func(A, B, c), test_impl(A, B, c))

    def test_unbox1(self):
        def test_impl(df):
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.random.ranf(n)})
        pd.testing.assert_series_equal(bodo_func(df), test_impl(df))

    def test_unbox2(self):
        def test_impl(df, cond):
            n = len(df)
            if cond:
                df["A"] = np.arange(n) + 2.0
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
        pd.testing.assert_series_equal(
            bodo_func(df.copy(), True), test_impl(df.copy(), True)
        )
        pd.testing.assert_series_equal(
            bodo_func(df.copy(), False), test_impl(df.copy(), False)
        )

    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "bb", "ccc"]})
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_box3(self):
        def test_impl(df):
            df2 = df[df.A != "dd"]
            return df2

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({"A": ["aa", "bb", "dd", "cc"]}, [3, 1, 2, -1])
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_box_dist_return(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df

        bodo_func = bodo.jit(distributed={"df"})(test_impl)
        n = 11
        hres, res = bodo_func(n), test_impl(n)
        self.assertTrue(count_array_OneDs() >= 3)
        self.assertTrue(count_parfor_OneDs() >= 1)
        dist_sum = bodo.jit(
            lambda a: bodo.libs.distributed_api.dist_reduce(
                a, np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
            )
        )
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(dist_sum(hres.A.sum()), res.A.sum())
        np.testing.assert_allclose(dist_sum(hres.B.sum()), res.B.sum())

    def test_len1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.random.ranf(n)})
            return len(df)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.random.ranf(n)})
            return df.shape

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_column_getitem1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
            Ac = df["A"].values
            return Ac.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_column_list_getitem1(self):
        def test_impl(df):
            return df[["A", "C"]]

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n), "C": np.random.ranf(n)})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_filter1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + n, "B": np.arange(n) ** 2})
            df1 = df[df.A > 0.5]
            return df1.B.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + n, "B": np.arange(n) ** 2})
            df1 = df.loc[df.A > 0.5]
            return np.sum(df1.B)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter3(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + n, "B": np.arange(n) ** 2})
            df1 = df.iloc[(df.A > 0.5).values]
            return np.sum(df1.B)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_iloc1(self):
        def test_impl(df, n):
            return df.iloc[1:n].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc2(self):
        def test_impl(df, n):
            return df.iloc[np.array([1, 4, 9])].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc3(self):
        def test_impl(df):
            return df.iloc[:, 1].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    @unittest.skip("TODO: support A[[1,2,3]] in Numba")
    def test_iloc4(self):
        def test_impl(df, n):
            return df.iloc[[1, 4, 9]].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc5(self):
        # test iloc with global value
        def test_impl(df):
            return df.iloc[:, COL_IND].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_loc1(self):
        def test_impl(df):
            return df.loc[:, "B"].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
            return df.iat[3, 1]

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n - 1, 1]

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
        self.assertEqual(bodo_func(df, n), test_impl(df, n))

    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n ** 2
            return df.A  # return the column to check column aliasing

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_series_equal(bodo_func(df, n), test_impl(df2, n))

    def test_iat_set2(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n ** 2
            return df  # check df aliasing/boxing

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_frame_equal(bodo_func(df, n), test_impl(df2, n))

    def test_set_column1(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n) + 3.0})
            df["A"] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column_reflect4(self):
        # set existing column
        def test_impl(df, n):
            df["A"] = np.arange(n)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n) + 3.0})
        df2 = df1.copy()
        bodo_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    def test_set_column_new_type1(self):
        # set existing column with a new type
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n) + 3.0})
            df["A"] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n) + 1.0})
            df["C"] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column_reflect3(self):
        # create new column
        def test_impl(df, n):
            df["C"] = np.arange(n)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n) + 3.0})
        df2 = df1.copy()
        bodo_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    def test_set_column_bool1(self):
        def test_impl(df):
            df["C"] = df["A"][df["B"]]

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({"A": [1, 2, 3], "B": [True, False, True]})
        df2 = df.copy()
        test_impl(df2)
        bodo_func(df)
        pd.testing.assert_series_equal(df.C, df2.C)

    def test_set_column_reflect1(self):
        def test_impl(df, arr):
            df["C"] = arr
            return df.C.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
        bodo_func(df, arr)
        self.assertIn("C", df)
        np.testing.assert_almost_equal(df.C.values, arr)

    def test_set_column_reflect2(self):
        def test_impl(df, arr):
            df["C"] = arr
            return df.C.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
        df2 = df.copy()
        np.testing.assert_almost_equal(bodo_func(df, arr), test_impl(df2, arr))

    def test_df_values1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(bodo_func(n), test_impl(n))

    def test_df_values2(self):
        def test_impl(df):
            return df.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_df_values_parallel1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df.values.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(n), test_impl(n))

    def test_df_apply_branch(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n)})
            B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
            return df.B.sum()

        n = 121
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(n), test_impl(n))

    def test_df_describe1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(0, n, 1, np.float32), "B": np.arange(n)})
            # df.A[0:1] = np.nan
            return df.describe()

        bodo_func = bodo.jit(test_impl)
        n = 1001
        bodo_func(n)
        # XXX: test actual output
        # self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_sort_values(self):
        def test_impl(df):
            df.sort_values("A", inplace=True)
            return df.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame(
            {"A": np.random.ranf(n), "B": np.arange(n), "C": np.random.ranf(n)}
        )
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_copy(self):
        def test_impl(df):
            df2 = df.sort_values("A")
            return df2.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame(
            {"A": np.random.ranf(n), "B": np.arange(n), "C": np.random.ranf(n)}
        )
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_single_col(self):
        def test_impl(df):
            df.sort_values("A", inplace=True)
            return df.A.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({"A": np.random.ranf(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_single_col_str(self):
        def test_impl(df):
            df.sort_values("A", inplace=True)
            return df.A.values

        n = 1211
        random.seed(2)
        str_vals = []

        for _ in range(n):
            k = random.randint(1, 30)
            val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        bodo_func = bodo.jit(test_impl)
        self.assertTrue((bodo_func(df.copy()) == test_impl(df)).all())

    def test_sort_values_str(self):
        def test_impl(df):
            df.sort_values("A", inplace=True)
            return df.B.values

        n = 1211
        random.seed(2)
        str_vals = []
        str_vals2 = []

        for _ in range(n):
            k = random.randint(1, 30)
            val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
            val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals2.append(val)

        df = pd.DataFrame({"A": str_vals, "B": str_vals2})
        # use mergesort for stability, in str generation equal keys are more probable
        sorted_df = df.sort_values("A", inplace=False, kind="mergesort")
        bodo_func = bodo.jit(test_impl)
        self.assertTrue((bodo_func(df) == sorted_df.B.values).all())

    def test_sort_parallel_single_col(self):
        # TODO: better parallel sort test
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            df.sort_values("points", inplace=True)
            res = df.points.values
            return res

        bodo_func = bodo.jit(locals={"res:return": "distributed"})(test_impl)

        save_min_samples = bodo.ir.sort.MIN_SAMPLES
        try:
            bodo.ir.sort.MIN_SAMPLES = 10
            res = bodo_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            bodo.ir.sort.MIN_SAMPLES = save_min_samples  # restore global val

    def test_sort_parallel(self):
        # TODO: better parallel sort test
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            df["A"] = df.points.astype(np.float64)
            df.sort_values("points", inplace=True)
            res = df.A.values
            return res

        bodo_func = bodo.jit(locals={"res:return": "distributed"})(test_impl)

        save_min_samples = bodo.ir.sort.MIN_SAMPLES
        try:
            bodo.ir.sort.MIN_SAMPLES = 10
            res = bodo_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            bodo.ir.sort.MIN_SAMPLES = save_min_samples  # restore global val

    def test_itertuples(self):
        def test_impl(df):
            res = 0.0
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n, np.int64)})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_itertuples_str(self):
        def test_impl(df):
            res = ""
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 3
        df = pd.DataFrame({"A": ["aa", "bb", "cc"], "B": np.ones(n, np.int64)})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_itertuples_order(self):
        def test_impl(n):
            res = 0.0
            df = pd.DataFrame({"B": np.arange(n), "A": np.ones(n, np.int64)})
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_itertuples_analysis(self):
        """tests array analysis handling of generated tuples, shapes going
        through blocks and getting used in an array dimension
        """

        def test_impl(n):
            res = 0
            df = pd.DataFrame({"B": np.arange(n), "A": np.ones(n, np.int64)})
            for r in df.itertuples():
                if r[1] == 2:
                    A = np.ones(r[1])
                    res += len(A)
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_df_head1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df.head(3)

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_pct_change1(self):
        def test_impl(n):
            df = pd.DataFrame(
                {"A": np.arange(n) + 1.0, "B": np.arange(n) + 1}, np.arange(n) + 1.3
            )
            return df.pct_change(3)

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_mean1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.mean()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_std1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.std()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_var1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.var()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_max1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.max()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_min1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.min()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_sum1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_prod1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.prod()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_count1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
            return df.count()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_df_fillna1(self):
        def test_impl(df):
            return df.fillna(5.0)

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_fillna_str1(self):
        def test_impl(df):
            return df.fillna("dd")

        df = pd.DataFrame({"A": ["aa", "b", None, "ccc"]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(11.0, inplace=True)
            return A

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df2))

    def test_df_reset_index1(self):
        def test_impl(df):
            return df.reset_index(drop=True)

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_reset_index_inplace1(self):
        def test_impl():
            df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
            df.reset_index(drop=True, inplace=True)
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_df_dropna1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna2(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    @unittest.skip("pending index support in dropna()")
    def test_df_dropna_inplace1(self):
        # TODO: fix error when no df is returned
        def test_impl(df):
            df.dropna(inplace=True)
            return df

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df2)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna_str1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 4.0, 1.0],
                "B": ["aa", "b", None, "ccc"],
                "C": [np.nan, ["AA", "A"], ["B"], ["CC", "D"]],
            }
        )
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_drop1(self):
        def test_impl(df):
            return df.drop(columns=["A"])

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_drop_inplace2(self):
        # test droping after setting the column
        def test_impl(df):
            df2 = df[["A", "B"]]
            df2["D"] = np.ones(3)
            df2.drop(columns=["D"], inplace=True)
            return df2

        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_drop_inplace1(self):
        def test_impl(df):
            df.drop("A", axis=1, inplace=True)
            return df

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df2))

    def test_isin_df1(self):
        def test_impl(df, df2):
            return df.isin(df2)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        df2 = pd.DataFrame({"A": np.arange(n), "C": np.arange(n) ** 2})
        df2.A[n // 2 :] = n
        pd.testing.assert_frame_equal(bodo_func(df, df2), test_impl(df, df2))

    @unittest.skip("needs dict typing in Numba")
    def test_isin_dict1(self):
        def test_impl(df):
            vals = {"A": [2, 3, 4], "C": [4, 5, 6]}
            return df.isin(vals)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_isin_list1(self):
        def test_impl(df):
            vals = [2, 3, 4]
            return df.isin(vals)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_append1(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        df2 = pd.DataFrame({"A": np.arange(n), "C": np.arange(n) ** 2})
        df2.A[n // 2 :] = n
        pd.testing.assert_frame_equal(bodo_func(df, df2), test_impl(df, df2))

    def test_append2(self):
        def test_impl(df, df2, df3):
            return df.append([df2, df3], ignore_index=True)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        df2 = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        df2.A[n // 2 :] = n
        df3 = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        pd.testing.assert_frame_equal(bodo_func(df, df2, df3), test_impl(df, df2, df3))

    def test_concat_columns1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2], axis=1)

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([4, 5])
        S2 = pd.Series([6.0, 7.0])
        # TODO: support int as column name
        pd.testing.assert_frame_equal(
            bodo_func(S1, S2), test_impl(S1, S2).rename(columns={0: "0", 1: "1"})
        )

    def test_var_rename(self):
        # tests df variable replacement in untyped_pass where inlining
        # can cause extra assignments and definition handling errors
        # TODO: inline freevar
        def test_impl():
            df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
            # TODO: df['C'] = [5,6,7]
            df["C"] = np.ones(3)
            return inner_get_column(df)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_names=False)


if __name__ == "__main__":
    unittest.main()
