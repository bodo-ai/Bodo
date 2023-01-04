# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime
import os
import random
import re
import shutil

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    reduce_sum,
)
from bodo.utils.typing import BodoError

_pivot_df1 = pd.DataFrame(
    {
        "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
        "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
        "C": [
            "small",
            "large",
            "large",
            "small",
            "small",
            "large",
            "small",
            "small",
            "large",
        ],
        "D": [1, 2, 2, 6, 3, 4, 5, 6, 9],
    }
)


def test_pivot_distributed_metadata(memory_leak_check):
    """
    Test that the DataFrame returned by df.pivot
    has its distribution information appended.
    """

    @bodo.jit
    def test_pivot():
        df = pd.DataFrame(
            {"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)}
        )
        return df.pivot(index="A", columns="C", values="B")

    @bodo.jit
    def test_gatherv(df):
        return bodo.gatherv(df)

    df = test_pivot()
    # df should have distirbuted metadata. If not, gatherv will raise a warning
    # that df is not distributed.
    with pytest.warns(None) as record:
        test_gatherv(df)
    assert len(record) == 0


@pytest.mark.parametrize(
    "add_args",
    [
        # Run with output columns known at compile time.
        {"pivots": {"pt": ["small", "large"]}},
        # Run with output columns unknown at compile time
        None,
    ],
)
def test_pivot_random_int_count_sum_prod_min_max(add_args, memory_leak_check):
    """Since the pivot can have missing values for keys (unlike groupby
    for which every rows has a matching key) integer columns are converted
    to nullable int bool"""

    def f1(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="count")
        return pt

    def f2(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        return pt

    def f3(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="prod")
        return pt

    def f4(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="max")
        return pt

    def f5(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="min")
        return pt

    def f6(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="first")
        return pt

    random.seed(5)
    n = 30
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) for _ in range(n)]
    df = pd.DataFrame({"A": list_A, "C": list_C, "D": list_D})
    check_func(
        f1,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f2,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f3,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f4,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f5,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f6,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    # Add an extra column to verify the selcted indices are correct.
    df = pd.DataFrame({"A": list_A, "C": list_C, "B": list_C, "D": list_D})
    check_func(
        f6,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )


@pytest.mark.slow
def test_pivot_table_count_date_index(memory_leak_check):
    """Check that DatetimeDateArray can be used as an index.
    See #2238."""

    def f1(df):
        pt = df.pivot_table(index="date_only", columns="C", values="D", aggfunc="count")
        return pt

    def f2(df):
        pt = pd.pivot_table(
            df, index="date_only", columns="C", values="D", aggfunc="count"
        )
        return pt

    random.seed(5)
    n = 20
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) + 0.4 for _ in range(n)]
    df = pd.DataFrame({"A": list_A, "C": list_C, "D": list_D})
    df["date_only"] = np.array(
        [datetime.date(2020, 1, 1) + datetime.timedelta(i) for i in range(n)]
    )
    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(
        f1,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
        # TODO: Remove reset_index when exact index types match.
        reset_index=True,
    )
    check_func(
        f2,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
        # TODO: Remove reset_index when exact index types match.
        reset_index=True,
    )


def test_pivot_random_float_sum_max(memory_leak_check):
    """For floating point no need to convert to nullable since floats
    support nans by themselves"""

    def f1(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        return pt

    def f2(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="max")
        return pt

    random.seed(5)
    n = 30
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) + 0.4 for _ in range(n)]
    df = pd.DataFrame({"A": list_A, "C": list_C, "D": list_D})
    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(
        f1,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f2,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )


def test_pivot_random_int_mean_var_std(memory_leak_check):
    def f1(df):
        pt = df.pivot_table(index="id", columns="foo", values="value", aggfunc="mean")
        return pt

    def f2(df):
        pt = df.pivot_table(index="id", columns="foo", values="value", aggfunc="var")
        return pt

    def f3(df):
        pt = df.pivot_table(index="id", columns="foo", values="value", aggfunc="std")
        return pt

    random.seed(5)
    n = 200
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) for _ in range(n)]
    df = pd.DataFrame({"id": list_A, "foo": list_C, "value": list_D})
    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(
        f1,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    check_func(
        f2,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
        dist_test=False,
    )
    check_func(
        f3,
        (df,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        check_dtype=False,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )


@pytest.mark.smoke
def test_pivot(memory_leak_check):
    def test_impl(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        return pt

    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(
        test_impl,
        (_pivot_df1,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        set_columns_name_to_none=True,
        check_dtype=False,
        reorder_columns=True,
    )


def test_pivot_drop_column(datapath, memory_leak_check):
    """The specificity of this test is that we compute pt.small.values.sum().
    Therefore, the "large" column gets removed from the ouput by the compiler passes.
    The pivot_table code thus has to handle this.
    We replaced the pt.small.values.sum() by len(pt.small.values) in order the problem
    with sum of nullable_int_bool"""
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        res = len(pt.small.values)
        return res

    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(impl, (), additional_compiler_arguments=add_args)


@pytest.mark.skip("[BE-3188] Disabling since needs to use new pivot infrastructure")
def test_crosstab(memory_leak_check):
    def test_impl(df):
        pt = pd.crosstab(df.A, df.C)
        return pt

    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(
        test_impl,
        (_pivot_df1,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )
    # 0 entries in the crosstab
    pivot_values = {"pt": ["small", "large", "middle"]}
    add_args = {"pivots": pivot_values}
    list_A = ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"]
    list_C = [
        "small",
        "large",
        "large",
        "small",
        "small",
        "large",
        "small",
        "small",
        "middle",
    ]
    dfW = pd.DataFrame({"A": list_A, "C": list_C})
    check_func(
        test_impl,
        (dfW,),
        additional_compiler_arguments=add_args,
        sort_output=True,
        set_columns_name_to_none=True,
        reorder_columns=True,
    )


@pytest.mark.skip("[BE-3188] Disabling since needs to use new pivot infrastructure")
def test_crosstab_deadcolumn(datapath, memory_leak_check):
    """The specificity of this test is that we compute pt.small.values.sum().
    Therefore, the "large" column gets removed from the ouput by the compiler passes.
    The pivot_table code thus has to handle this"""
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = pd.crosstab(df.A, df.C)
        res = pt.small.values.sum()
        return res

    pivot_values = {"pt": ["small", "large"]}
    add_args = {"pivots": pivot_values}
    check_func(impl, (), additional_compiler_arguments=add_args)


@pytest.mark.skip("[BE-3188] Disabling since needs to use new pivot infrastructure")
def test_crosstab_invalid_types(memory_leak_check):
    """
    Tests that pd.crosstab produces a reasonable error message
    when index/column use unsupported types.
    """

    @bodo.jit
    def impl(index, columns):
        return pd.crosstab(index, columns)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4] * 3,
        }
    )
    with pytest.raises(
        BodoError, match=re.escape("'index' argument only supported for Series types")
    ):
        impl(df, df.A)
    with pytest.raises(
        BodoError, match=re.escape("'columns' argument only supported for Series types")
    ):
        impl(df.A, df)


def test_pivot_invalid_types(memory_leak_check):
    """
    Tests that pd.pivot and pd.pivot_table produces a reasonable error message
    when data is an unsupported types.
    """

    @bodo.jit
    def impl1(S):
        return pd.pivot(S, "A", "B", "C")

    @bodo.jit
    def impl2(S):
        return pd.pivot_table(S, "A", "B", "C")

    S = pd.Series([1, 2, 3, 4] * 3)
    with pytest.raises(
        BodoError, match=re.escape("'data' argument must be a DataFrame")
    ):
        impl1(S)
    with pytest.raises(
        BodoError, match=re.escape("'data' argument must be a DataFrame")
    ):
        impl2(S)


@pytest.mark.parametrize(
    "df",
    [
        # Basic DataFrame with NAs to insert
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # Integer column names
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [i for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # Timestamp values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [i for i in range(10)] * 100,
                "C": pd.Series(pd.date_range("1/1/2022", freq="H", periods=1000)),
            }
        ),
        # Nullable Integer Values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [i for i in range(10)] * 100,
                "C": pd.array(([i for i in range(9)] + [None]) * 100, dtype="Int32"),
            }
        ),
        # String Index
        pd.DataFrame(
            {
                "A": [str(i) for i in range(1000)],
                "B": [i for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # String values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [i for i in range(10)] * 100,
                "C": [str(i) for i in range(1000, 2000)],
            }
        ),
    ],
)
def test_pivot_basic(df, memory_leak_check):
    """
    Checks basic support for DataFrame.pivot on various datatypes.
    """
    # Test pivot unboxing
    def impl1(df):
        return df.pivot(index="A", columns="B", values="C")

    # Test that len still works
    def impl2(df):
        res = df.pivot(index="A", columns="B", values="C")
        return len(res)

    # Test that a list works.
    def impl3(df):
        res = df.pivot(index=["A"], columns=["B"], values=["C"])
        return len(res)

    # We don't capture the name field of the columns index,
    # so we set check_names=False.

    # Pivot produces nullable float values instead of nullable int
    # values for the data. As a result, since we don't support Float64,
    # we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl1,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )
    check_func(impl2, (df,), check_names=False, check_dtype=False, sort_output=True)
    check_func(impl3, (df,), check_names=False, check_dtype=False, sort_output=True)


@pytest.mark.parametrize(
    "df",
    [
        # Basic DataFrame with NAs to insert
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": [str(i) for i in range(1000)],
                "C": np.arange(1000, 2000),
            }
        ),
        # Integer column names
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": np.arange(1000),
                "C": np.arange(1000, 2000),
            }
        ),
        # Timestamp values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": np.arange(1000),
                "C": pd.Series(pd.date_range("1/1/2022", freq="H", periods=1000)),
            }
        ),
        # Nullable Integer Values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": np.arange(1000),
                "C": pd.array(([i for i in range(9)] + [None]) * 100, dtype="Int32"),
            }
        ),
        # String Index
        pd.DataFrame(
            {
                "A": [str(i) for i in range(1000)],
                "B": np.arange(1000),
                "C": np.arange(1000, 2000),
            }
        ),
        # String values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": np.arange(1000),
                "C": [str(i) for i in range(1000, 2000)],
            }
        ),
    ],
)
def test_pivot_empty(df, memory_leak_check):
    """
    Tests support for DataFrame.pivot on various
    types when each rank will contain at least 1
    empty column.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values="C")

    # We don't capture the name field of the columns index,
    # so we set check_names=False.

    # Pivot produces nullable float values instead of nullable int
    # values for the data. As a result, since we don't support Float64,
    # we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # Basic DataFrame with NAs to insert
        pd.DataFrame(
            {
                "A": [i for i in range(3)] * 5,
                "B": [str(i) for i in range(5)] * 3,
                "C": np.arange(1000, 1015),
            }
        ),
        # Integer column names
        pd.DataFrame(
            {
                "A": [i for i in range(3)] * 5,
                "B": [i for i in range(5)] * 3,
                "C": np.arange(1000, 1015),
            }
        ),
        # Timestamp values
        pd.DataFrame(
            {
                "A": [i for i in range(3)] * 5,
                "B": [i for i in range(5)] * 3,
                "C": pd.Series(pd.date_range("1/1/2022", freq="H", periods=15)),
            }
        ),
        # Nullable Integer Values
        pd.DataFrame(
            {
                "A": [i for i in range(3)] * 5,
                "B": [i for i in range(5)] * 3,
                "C": pd.array(
                    [i for i in range(11)] + [None, None, None, None], dtype="Int32"
                ),
            }
        ),
        # String Index
        pd.DataFrame(
            {
                "A": [str(i) for i in range(3)] * 5,
                "B": [i for i in range(5)] * 3,
                "C": np.arange(1000, 1015),
            }
        ),
        # String values
        pd.DataFrame(
            {
                "A": [i for i in range(3)] * 5,
                "B": [i for i in range(5)] * 3,
                "C": [str(i) for i in range(1000, 1015)],
            }
        ),
    ],
)
def test_pivot_full(df, memory_leak_check):
    """
    Tests support for DataFrame.pivot on various
    types when there won't be NAs due to missing values.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values="C")

    # We don't capture the name field of the columns index,
    # so we set check_names=False.

    # Pivot produces nullable float values instead of nullable int
    # values for the data. As a result, since we don't support Float64,
    # we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # Basic case
        pd.DataFrame(
            {
                "A": pd.array(
                    [i for i in range(996)] + [None, None, None, None], dtype="Int32"
                ),
                "B": [i for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # Empty case
        pd.DataFrame(
            {
                "A": pd.array(
                    [i for i in range(996)] + [None, None, None, None], dtype="Int32"
                ),
                "B": np.arange(1000),
                "C": np.arange(1000, 2000),
            }
        ),
        # Full case
        pd.DataFrame(
            {
                "A": pd.array([0, 1, None] * 5, dtype="Int32"),
                "B": [i for i in range(5)] * 3,
                "C": np.arange(1000, 1015),
            }
        ),
    ],
)
def test_pivot_na_index(df, memory_leak_check):
    """
    Tests support for DataFrame.pivot on various
    types when one of the index values is NA.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values="C")

    # We don't capture the name field of the columns index,
    # so we set check_names=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
        # Use py_output because Bodo keeps the values an int64, but Pandas
        # uses a float
        py_output=impl(df).astype("Int64"),
    )


@pytest.mark.parametrize(
    "df",
    [
        # Normal case
        pd.DataFrame(
            {
                "A": [i for i in range(5)] * 4,
                "B": [i for i in range(2)] * 10,
                "C": np.arange(20),
            }
        ),
        # String data
        pd.DataFrame(
            {
                "A": [i for i in range(5)] * 4,
                "B": [i for i in range(2)] * 10,
                "C": [str(i) for i in range(20)],
            }
        ),
    ],
)
# TODO: Enable memory_leak_check. Disabled because we raise an exception
def test_pivot_repeat_value(df):
    """
    Tests that DataFrame.pivot raises an exception
    when an (index, column) pair is repeated more
    than once.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values="C")

    err_msg = re.escape(
        "DataFrame.pivot(): 'index' contains duplicate entries for the same output column"
    )
    with pytest.raises(ValueError, match=err_msg):
        bodo.jit(impl)(df)


@pytest.mark.parametrize(
    "df",
    [
        # (string, string) multi-index columns
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                "B": [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                    "three",
                    "seven",
                ],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                    "medium",
                    "medium",
                ],
                "D": [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
        # (int64, string) multi-index columns
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                0: [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                    "three",
                    "seven",
                ],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                    "medium",
                    "medium",
                ],
                1: [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
        # (string, int64) multi-index columns
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                "B": [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                    "three",
                    "seven",
                ],
                "C": [
                    4,
                    35,
                    35,
                    4,
                    4,
                    35,
                    4,
                    4,
                    35,
                    11,
                    11,
                ],
                "D": [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
        # (int64, int64) multi-index columns
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                0: [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                    "three",
                    "seven",
                ],
                "C": [
                    4,
                    35,
                    35,
                    4,
                    4,
                    35,
                    4,
                    4,
                    35,
                    11,
                    11,
                ],
                1: [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
    ],
)
def test_pivot_table_multiple_values_numeric(df, memory_leak_check):
    """
    Test for using > 1 value column with pivot_table where each column
    holds numeric data. Each value passed has the same exact data and
    differs only in the types tested for the new column names.
    """

    def impl(df):
        return df.pivot_table(index="A", columns="C", values=None, aggfunc="count")

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


def test_pivot_table_multiple_values_string(memory_leak_check):
    """
    Test for using > 1 value column with pivot_table where each column
    holds string data.
    """

    def impl(df):
        return df.pivot_table(
            index="A", columns="C", values=["D", "B"], aggfunc="first"
        )

    df = pd.DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "f",
                "g",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "two",
                "one",
                "one",
                "two",
                "two",
                "three",
                "seven",
            ],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
                "medium",
                "medium",
            ],
            "D": [
                "A",
                "B",
                "B",
                "fdwfefw",
                "#424",
                "fewfe",
                "#$@5234",
                "fdwfefw",
                "r2r324f23ce",
                "fdwfefw",
                "#$@5234",
            ],
        }
    )

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # String values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": [str(i) for i in range(2000, 3000)],
                "B": [i for i in range(10)] * 100,
                "C": [str(i) for i in range(1000, 2000)],
            }
        ),
        # Numeric values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": np.arange(2000, 3000),
                "B": [i for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
    ],
)
def test_pivot_multiple_values(df, memory_leak_check):
    """
    Test running DataFrame.pivot() with multiple values
    of various types.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values=["C", "D"])

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # (string, int64)
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": [str(i) for i in range(2000, 3000)],
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # (int64, float64)
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": np.arange(2000, 3000),
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000).astype(np.float64),
            }
        ),
    ],
)
def test_pivot_diff_value_types(df, memory_leak_check):
    """
    Test running DataFrame.pivot() with multiple values
    of differing types.
    """

    def impl(df):
        return df.pivot(index="A", columns="B", values=["C", "D"])

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # (string, int64)
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                "B": [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                    "three",
                    "seven",
                ],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                    "medium",
                    "medium",
                ],
                "D": [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
        # (int64, float64)
        pd.DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "f",
                    "g",
                ],
                "B": [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 7],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                    "medium",
                    "medium",
                ],
                "D": [1.6, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
            }
        ),
    ],
)
def test_pivot_table_diff_value_types(df, memory_leak_check):
    """
    Test running DataFrame.pivot_table() with multiple values
    of differing types.
    """

    def impl(df):
        return df.pivot_table(
            index="A", columns="C", values=["D", "B"], aggfunc="first"
        )

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


def test_pd_pivot_multi_values(memory_leak_check):
    """
    Test pivot and pivot_table with multiple values using the top
    level pandas functions.
    """

    def impl1(df):
        return pd.pivot_table(
            df, index="A", columns="C", values=["D", "B"], aggfunc="first"
        )

    def impl2(df):
        return pd.pivot(df, index="A", columns="B", values=["C", "D"])

    df = pd.DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "f",
                "g",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "two",
                "one",
                "one",
                "two",
                "two",
                "three",
                "seven",
            ],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
                "medium",
                "medium",
            ],
            "D": [1, 2, 2, 6, 3, 4, 5, 6, 9, 6, 5],
        }
    )

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl1,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )

    df = pd.DataFrame(
        {
            "A": np.arange(1000),
            "D": np.arange(2000, 3000),
            "B": [str(i) for i in range(10)] * 100,
            "C": np.arange(1000, 2000).astype(np.float64),
        }
    )

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl2,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "C": np.arange(1000),
                "B": [str(i) for i in range(2000, 3000)],
                "D": [i for i in range(10)] * 100,
                "A": [str(i) for i in range(1000, 2000)],
            }
        ),
        # Numeric values
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": np.arange(2000, 3000),
                "B": [i for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # (string, int64)
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": [str(i) for i in range(2000, 3000)],
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # (int64, float64)
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": np.arange(2000, 3000),
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000).astype(np.float64),
            }
        ),
    ]
)
def pivot_dataframes(request):
    return request.param


def test_pivot_multiple_index(pivot_dataframes, memory_leak_check):
    """
    Test running DataFrame.pivot() with multiple index
    values.
    """

    def impl(df):
        return df.pivot(index=["A", "B"], columns="D", values="C")

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (pivot_dataframes,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


def test_pivot_table_multiple_index(pivot_dataframes, memory_leak_check):
    """
    Test running DataFrame.pivot_table() with multiple index
    values.
    """

    def impl(df):
        return df.pivot_table(
            index=["A", "B"], columns="D", values="C", aggfunc="first"
        )

    # Pivot may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (pivot_dataframes,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # Range index handling
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": [str(i) for i in range(2000, 3000)],
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            }
        ),
        # Index with a name
        pd.DataFrame(
            {
                "A": np.arange(1000),
                "D": [str(i) for i in range(2000, 3000)],
                "B": [str(i) for i in range(10)] * 100,
                "C": np.arange(1000, 2000),
            },
            index=pd.Index([i for i in range(500)] * 2, name="my_index_name"),
        ),
    ],
)

# https://dev.azure.com/bodo-inc/Bodo/_test/analytics?definitionId=5&contextType=build
# test_pivot_index_none[df0] on average takes 13.27 min, or 796.2 seconds
@pytest.mark.timeout(1000)
def test_pivot_index_none(df, memory_leak_check):
    """
    Test running DataFrame.pivot() with index=None and values=None.
    """

    def impl(df):
        return df.pivot(index=None, columns="D", values=None)

    # Pivot Table may produce different nullable data, so we set check_dtype=False.

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.skip
def test_pivot_table_index_none(memory_leak_check):
    """
    Test running DataFrame.pivot_table() with index=None and values=None.
    """

    def impl(df):
        return df.pivot_table(index=None, columns="D", values=None, aggfunc="first")

    # Pivot Table may produce different nullable data, so we set check_dtype=False.
    df = pd.DataFrame(
        {
            "A": np.arange(1000),
            "D": [str(i) for i in range(2000, 3000)],
            "B": [str(i) for i in range(10)] * 100,
            "C": np.arange(1000, 2000),
        }
    )

    # sort_output becuase row order isn't maintained by pivot.
    # reorder_columns because the column order is consistent but not defined.
    check_func(
        impl,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        # Integer values
        pd.DataFrame(
            {
                "A": np.arange(10),
                "B": ["teq", "b", "ce", "32", "re2"] * 2,
                "C": np.arange(10),
            }
        ),
        # String values
        pd.DataFrame(
            {
                "A": np.arange(10),
                "B": ["teq", "b", "ce", "32", "re2"] * 2,
                "C": [str(i) for i in range(10)],
            }
        ),
    ],
)
def test_pivot_to_parquet(df, memory_leak_check):
    """
    Tests calling to_parquet on the output of DataFrame.pivot()
    without requiring an intermediate move to Python.
    """
    import pyarrow.parquet as pq

    output_filename = "bodo_temp.pq"

    def impl(df, fname):
        new_df = df.pivot(index="A", columns="B", values="C")
        new_df.to_parquet(fname)

    py_output = df.pivot(index="A", columns="B", values="C")
    # Test with replicated and distributed data
    for distributed in (True, False):
        if distributed:
            df_in = _get_dist_arg(df)
            compiler_kwargs = {"distributed": ["df"]}
        else:
            df_in = df
            compiler_kwargs = {}
        bodo.jit(**compiler_kwargs)(impl)(df_in, output_filename)
        bodo.barrier()
        # Read the data on rank 0 and compare
        passed = 1
        if bodo.get_rank() == 0:
            try:
                result = pd.read_parquet(output_filename)
                # Reorder the columns since this can't be done in _test_equals_guard.
                result.sort_index(axis=1, inplace=True)
                passed = _test_equal_guard(
                    result,
                    py_output,
                    sort_output=True,
                    check_names=False,
                    check_dtype=False,
                )
                bodo_table = pq.read_table(output_filename)
                # Check the metadata
                assert bodo_table.schema.pandas_metadata.get("index_columns") == [
                    py_output.index.name
                ]
                cols_metadata = bodo_table.schema.pandas_metadata.get("columns")
                # Generate expected typenames for Int64 and string
                pd_type = "int64" if py_output.b.dtype == np.float64 else "unicode"
                np_type = "Int64" if py_output.b.dtype == np.float64 else "object"
                total_columns = py_output.columns.to_list() + [py_output.index.name]
                for col_metadata in cols_metadata:
                    assert col_metadata["name"] in total_columns, "Name doesn't match"
                    assert (
                        col_metadata["field_name"] in total_columns
                    ), "field_name doesn't match"
                    if col_metadata["name"] == py_output.index.name:
                        # Index is always an Integer column.
                        expected_pd_type = "int64"
                        expected_np_type = "int64"
                    else:
                        expected_pd_type = pd_type
                        expected_np_type = np_type
                    assert (
                        col_metadata["pandas_type"] == expected_pd_type
                    ), "pandas_type doesn't match"
                    assert (
                        col_metadata["numpy_type"] == expected_np_type
                    ), "numpy_type doesn't match"
                    assert col_metadata["metadata"] is None, "metadata doesn't match"
            except Exception:
                passed = 0
            finally:
                if distributed:
                    shutil.rmtree(output_filename)
                else:
                    os.remove(output_filename)
        n_passed = reduce_sum(passed)
        data_dist = "distributed" if distributed else "replicated"
        assert (
            n_passed == bodo.get_size()
        ), f"Output doesn't match Pandas with {data_dist} data"
        bodo.barrier()


def test_pivot_table_dict_encoded(memory_leak_check):
    """
    Tests support for df.pivot_table with dictionary
    encoded string columns.

    This won't output a final DataFrame with dictionary encoded
    arrays because the groupby operation will remove any
    dictionary encoding in the reduction. When dictionary encoding
    is supported with groupby pivot may need further updates.
    """

    temp_filename = "pivot_temp.pq"
    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": [str(i) for i in range(1000)] * 10,
                "D": [str(i) for i in range(2000, 3000)] * 10,
                "B": [str(i) for i in range(10)] * 1000,
                "C": [str(i) for i in range(1000, 2000)] * 10,
            }
        )
        df.to_parquet(temp_filename)
    bodo.barrier()

    def impl():
        df = pd.read_parquet(temp_filename, _bodo_read_as_dict=["A", "B", "C", "D"])
        return df.pivot_table(index=["B", "C"], columns="D", values="A", aggfunc="min")

    try:
        py_output = pd.read_parquet(temp_filename)
        py_output = py_output.pivot_table(
            index=["B", "C"], columns="D", values="A", aggfunc="min"
        )
        check_func(
            impl,
            (),
            py_output=py_output,
            check_names=False,
            check_dtype=False,
            sort_output=True,
            reorder_columns=True,
        )
    finally:
        if bodo.get_rank() == 0:
            os.remove(temp_filename)
        bodo.barrier()


def test_pivot_dict_encoded(memory_leak_check):
    """
    Tests support for df.pivot_table with dictionary
    encoded string columns.

    This won't output a final DataFrame with dictionary encoded
    arrays because the groupby operation will remove any
    dictionary encoding in the reduction. When dictionary encoding
    is supported with groupby pivot may need further updates.
    """

    temp_filename = "pivot_temp.pq"
    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": [str(i) for i in range(100)] * 1000,
                "D": [str(i) for i in range(200000, 300000)],
                "B": [str(i) for i in range(10)] * 1000 * 10,
                "C": [str(i) for i in range(100, 200)] * 1000,
            }
        )
        df.to_parquet(temp_filename)
    bodo.barrier()

    def impl():
        df = pd.read_parquet(temp_filename, _bodo_read_as_dict=["A", "B", "C", "D"])
        return df.pivot(index=["D", "C"], columns="A", values="B")

    try:
        py_output = pd.read_parquet(temp_filename)
        py_output = py_output.pivot(index=["D", "C"], columns="A", values="B")
        check_func(
            impl,
            (),
            py_output=py_output,
            check_names=False,
            check_dtype=False,
            sort_output=True,
            reorder_columns=True,
            dist_test=False,
        )
    finally:
        if bodo.get_rank() == 0:
            os.remove(temp_filename)
        bodo.barrier()


def test_pivot_table_non_ascii(memory_leak_check):
    """
    Tests DataFrame.pivot_table with string values for index,
    columns, and values that contain non-ascii strings.
    """

    def impl1(df):
        return df.pivot_table(index="C", columns="B", values="A", aggfunc="min")

    def impl2(df):
        return df.pivot_table(
            index=["C", "D"], columns="B", values=["A", "E"], aggfunc="min"
        )

    df = pd.DataFrame(
        {
            "A": ["Õabf3e", "Õ", "Õ32r23", "Õr3", "Õe32"] * 20,
            "B": ["россия очень, холодная страна", "россия очень"] * 50,
            "C": [
                "🢇🄐,🏈𠆶💑😅",
                "한국,가,고싶다ㅠ",
                "مرحبا, العالم ، هذا هو بودو",
                "🢇🄐,",
                "한국,가",
                "🢇🄐💑😅",
                "한국싶다ㅠ",
                "مرحبا, ",
                "🢇,",
                ",가",
            ]
            * 10,
            "D": ["Õfv43", "Õ3323"] * 50,
            "E": ["한국,가,고싶다ㅠ"] * 100,
        }
    )
    check_func(
        impl1,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )
    check_func(
        impl2,
        (df,),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reorder_columns=True,
    )
