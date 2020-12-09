# Copyright (C) 2019 Bodo Inc. All rights reserved.
import datetime
import random

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func

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


def test_pivot_random_int_count_sum_prod_min_max(memory_leak_check):
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

    random.seed(5)
    n = 30
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) for _ in range(n)]
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


@pytest.mark.slow
def test_pivot_table_count_date_index(memory_leak_check):
    """Check that DatetimeDateArray can be used as an index.
    See #2238."""

    def f1(df):
        pt = df.pivot_table(index="date_only", columns="C", values="D", aggfunc="count")
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
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="mean")
        return pt

    def f2(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="var")
        return pt

    def f3(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="std")
        return pt

    random.seed(5)
    n = 200
    n_keyA = 10
    list_A = [str(random.randint(10, 10 + n_keyA)) for _ in range(n)]
    list_C = [random.choice(["small", "large"]) for _ in range(n)]
    list_D = [random.randint(1, 1000) for _ in range(n)]
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


@pytest.mark.smoke
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
