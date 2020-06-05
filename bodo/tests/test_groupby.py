# Copyright (C) 2019 Bodo Inc. All rights reserved.
import random
import string
import datetime
import pandas as pd
import numpy as np
import bodo
from decimal import Decimal
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    check_parallel_coherency,
    check_func_type_extent,
    convert_list_string_decimal_columns,
    get_start_end,
    check_func,
)
import pytest


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [2, 1, np.nan, 1, 2, 2, 1],
                "B": [-8, 2, 3, 1, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": pd.Series(np.full(7, np.nan), dtype="Int64"),
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2.1, -1.5, 0.0, -1.5, 2.1, 2.1, 1.5],
                "B": [-8.3, np.nan, 3.8, 1.3, 5.4, np.nan, -7.0],
                "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3, -3.7],
            }
        ),
    ]
)
def test_df(request):
    return request.param


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [2, 1, np.nan, 1, 2, 2, 1],
                "B": [-8, 2, 3, 1, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2, 1, 1, 1, 2, 2, 1],
                "B": [-8, np.nan, 3, np.nan, 5, 6, 7],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
        pd.DataFrame(
            {
                "A": [2.1, -1.5, 0.0, -1.5, 2.1, 2.1, 1.5],
                "B": [-8.3, np.nan, 3.8, 1.3, 5.4, np.nan, -7.0],
                "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3, -3.7],
            }
        ),
    ]
)
def test_df_int_no_null(request):
    """
    Testing data for functions that does not support nullable integer columns
    with nulls only 

    Ideally, all testing function using test_df_int_no_null as inputs
    should support passing tests with test_df
    """
    return request.param


@pytest.mark.slow
def test_nullable_int():
    def impl(df):
        A = df.groupby("A").sum()
        return A

    def impl_select_colB(df):
        A = df.groupby("A")["B"].sum()
        return A

    def impl_select_colE(df):
        A = df.groupby("A")["E"].sum()
        return A

    def impl_select_colH(df):
        A = df.groupby("A")["H"].sum()
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="Int8"
            ),
            "C": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="Int16"
            ),
            "D": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="Int32"
            ),
            "E": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="Int64"
            ),
            "F": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="UInt8"
            ),
            "G": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="UInt16"
            ),
            "H": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="UInt32"
            ),
            "I": pd.Series(
                np.array([np.nan, 8, 2, np.nan, np.nan, np.nan, 20]), dtype="UInt64"
            ),
        }
    )

    check_func(impl, (df,), sort_output=True, reset_index=True)
    # pandas 1.0 has a regression here: output is int64 instead of Int8
    # so we disable check_dtype
    check_func(
        impl_select_colB, (df,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(impl_select_colE, (df,), sort_output=True, reset_index=True)
    # pandas 1.0 has a regression here: output is int64 instead of UInt32
    check_func(
        impl_select_colH, (df,), sort_output=True, reset_index=True, check_dtype=False
    )


@pytest.mark.parametrize(
    "df_null",
    [
        pd.DataFrame(
            {"A": [2, 1, 1, 1], "B": pd.Series(np.full(4, np.nan), dtype="Int64")}
        ),
        pd.DataFrame({"A": [1, 1, 1, 1], "B": pd.Series([1, 2, 3, 4], dtype="Int64")}),
    ],
)
def test_return_type_nullable_cumsum_cumprod(df_null):
    """
    Test Groupby when one row is a nullable-int-bool.
    A current problem is that cumsum/cumprod with pandas return an array of float for Int64
    in input. That is why we put check_dtype=False here.
    """

    def impl1(df):
        df2 = df.groupby("A")["B"].agg(("cumsum", "cumprod"))
        return df2

    def impl2(df):
        df2 = df.groupby("A")["B"].cumsum()
        return df2

    check_func(impl1, (df_null,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (df_null,), sort_output=True, reset_index=True, check_dtype=False)


def test_all_null_keys():
    """
    Test Groupby when all rows have null keys (returns empty dataframe)
    """

    def impl(df):
        A = df.groupby("A").count()
        return A

    df = pd.DataFrame(
        {"A": pd.Series(np.full(7, np.nan), dtype="Int64"), "B": [2, 1, 1, 1, 2, 2, 1]}
    )

    check_func(impl, (df,), sort_output=True, reset_index=True)


udf_in_df = pd.DataFrame(
    {
        "A": [2, 1, 1, 1, 2, 2, 1],
        "B": [-8, 2, 3, 1, 5, 6, 7],
        "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
    }
)


def test_agg():
    """
    Test Groupby.agg(): one user defined func and all cols
    """

    def impl(df):
        A = df.groupby("A").agg(lambda x: x.max() - x.min())
        return A

    # check_dtype=False since Bodo returns float for Series.min/max. TODO: fix min/max
    check_func(
        impl, (udf_in_df,), sort_output=True, reset_index=True, check_dtype=False
    )


def test_sum_string():
    def impl(df):
        A = df.groupby("A").sum()
        return A

    df1 = pd.DataFrame({"A": [1, 1, 1, 2], "B": ["a", "b", "c", "d"]})
    check_func(impl, (df1,), sort_output=True, reset_index=True)


def test_random_decimal_sum_min_max_last():
    """We do not have decimal as index. Therefore we have to use as_index=False
    """

    def impl1(df):
        df_ret = df.groupby("A", as_index=False).nunique()
        return df_ret["B"].copy()

    def impl2(df):
        A = df.groupby("A", as_index=False).last()
        return A

    def impl3(df):
        A = df.groupby("A", as_index=False)["B"].first()
        return A

    def impl4(df):
        A = df.groupby("A", as_index=False)["B"].count()
        return A

    def impl5(df):
        A = df.groupby("A", as_index=False).max()
        return A

    def impl6(df):
        A = df.groupby("A", as_index=False).min()
        return A

    def impl7(df):
        A = df.groupby("A", as_index=False)["B"].mean()
        return A

    def impl8(df):
        A = df.groupby("A", as_index=False)["B"].median()
        return A

    def impl9(df):
        A = df.groupby("A", as_index=False)["B"].var()
        return A

    # We need to drop column A because the column A is replaced by std(A)
    # in pandas due to a pandas bug.
    def impl10(df):
        A = df.groupby("A", as_index=False)["B"].std()
        return A.drop(columns="A")

    def compute_random_decimal(opt, n):
        e_list = []

        def random_str1():
            e_str1 = str(1 + random.randint(1, 8))
            e_str2 = str(1 + random.randint(1, 8))
            return e_str1 + "." + e_str2

        def random_str2():
            klen1 = random.randint(1, 7)
            klen2 = random.randint(1, 7)
            e_str1 = "".join([str(1 + random.randint(1, 8)) for _ in range(klen1)])
            e_str2 = "".join([str(1 + random.randint(1, 8)) for _ in range(klen2)])
            esign = "" if random.randint(1, 2) == 1 else "-"
            return esign + e_str1 + "." + e_str2

        for _ in range(n):
            if opt == 1:
                e_str = random_str1()
            if opt == 2:
                e_str = random_str2()
            e_list.append(Decimal(e_str))
        return pd.Series(e_list)

    random.seed(5)
    n = 10
    df1 = pd.DataFrame(
        {"A": compute_random_decimal(1, n), "B": compute_random_decimal(2, n)}
    )

    # Direct checks for which pandas has equivalent functions.
    check_func(impl1, (df1,), sort_output=True, reset_index=True)
    check_func(impl2, (df1,), sort_output=True, reset_index=True)
    check_func(impl3, (df1,), sort_output=True, reset_index=True)
    check_func(impl4, (df1,), sort_output=True, reset_index=True)
    check_func(impl5, (df1,), sort_output=True, reset_index=True)
    check_func(impl6, (df1,), sort_output=True, reset_index=True)

    # For mean/median/var/std we need to map the types.
    check_func_type_extent(impl7, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(impl8, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(impl9, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(impl10, (df1,), sort_output=True, reset_index=True)


def test_random_string_sum_min_max_first_last():
    def impl1(df):
        A = df.groupby("A").sum()
        return A

    def impl2(df):
        A = df.groupby("A").min()
        return A

    def impl3(df):
        A = df.groupby("A").max()
        return A

    def impl4(df):
        A = df.groupby("A").first()
        return A

    def impl5(df):
        A = df.groupby("A").last()
        return A

    def random_dataframe(n):
        random.seed(5)
        eList_A = []
        eList_B = []
        for i in range(n):
            len_str = random.randint(1, 10)
            val_A = random.randint(1, 10)
            val_B = "".join(random.choices(string.ascii_uppercase, k=len_str))
            eList_A.append(val_A)
            eList_B.append(val_B)
        return pd.DataFrame({"A": eList_A, "B": eList_B})

    df1 = random_dataframe(100)
    check_func(impl1, (df1,), sort_output=True, reset_index=True)
    check_func(impl2, (df1,), sort_output=True, reset_index=True)
    check_func(impl3, (df1,), sort_output=True, reset_index=True)
    check_func(impl4, (df1,), sort_output=True, reset_index=True)
    check_func(impl5, (df1,), sort_output=True, reset_index=True)


def test_groupby_missing_entry():
    """The columns which cannot be processed cause special problems as they are
    sometimes dropped instead of failing
    """

    def test_drop_sum(df):
        return df.groupby("A").sum()

    def test_drop_count(df):
        return df.groupby("A").count()

    df1 = pd.DataFrame(
        {"A": [3, 2, 3], "B": pd.date_range("2017-01-03", periods=3), "C": [3, 1, 2]}
    )
    df2 = pd.DataFrame({"A": [3, 2, 3], "B": ["aa", "bb", "cc"], "C": [3, 1, 2]})
    df3 = pd.DataFrame({"A": [3, 2, 3], "B": ["aa", "bb", "cc"]})
    check_func(test_drop_sum, (df1,), sort_output=True)
    check_func(test_drop_sum, (df2,), sort_output=True)
    check_func(test_drop_sum, (df3,), sort_output=True)
    check_func(test_drop_count, (df1,), sort_output=True)
    check_func(test_drop_count, (df2,), sort_output=True)
    check_func(test_drop_count, (df3,), sort_output=True)


def test_agg_str_key():
    """
    Test Groupby.agg() with string keys
    """

    def impl(df):
        A = df.groupby("A").agg(lambda x: x.sum())
        return A

    df = pd.DataFrame(
        {"A": ["AA", "B", "B", "B", "AA", "AA", "B"], "B": [-8, 2, 3, 1, 5, 6, 7],}
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_agg_series_input():
    """
    Test Groupby.agg(): make sure input to UDF is a Series, not Array
    """

    def impl(df):
        # using `count` since Arrays don't support it
        A = df.groupby("A").agg(lambda x: x.count())
        return A

    # check_dtype=False since Pandas returns float64 for count sometimes for some reason
    check_func(
        impl, (udf_in_df,), sort_output=True, reset_index=True, check_dtype=False
    )


def test_agg_bool_expr():
    """
    Test Groupby.agg(): make sure boolean expressions work (#326)
    """

    def impl(df):
        return df.groupby("A")["B"].agg(lambda x: ((x == "A") | (x == "B")).sum())

    df = pd.DataFrame({"A": [1, 2, 1, 2] * 2, "B": ["A", "B", "C", "D"] * 2})
    check_func(impl, (df,), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df_index",
    [
        pd.DataFrame(
            {
                "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
                "B": [1, 2, 3, 2, 1, 1, 1],
                "C": [3, 5, 6, 5, 4, 4, 3],
            },
            index=[-1, 2, -3, 0, 4, 5, 2],
        ),
        pd.DataFrame(
            {
                "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
                "B": [1, 2, 3, 2, 1, 1, 1],
                "C": [3, 5, 6, 5, 4, 4, 3],
            },
            index=["a", "b", "c", "d", "e", "f", "g"],
        ),
        pd.DataFrame(
            {
                "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
                "B": [1, 2, 3, 2, 1, 1, 1],
                "C": [3, 5, 6, 5, 4, 4, 3],
            }
        ),
    ],
)
def test_cumsum_index_preservation(df_index):
    def test_impl_basic(df1):
        df2 = df1.groupby("B").cumsum()
        return df2

    def test_impl_both(df1):
        df2 = df1.groupby("B")["C"].agg(("cumprod", "cumsum"))
        return df2

    def test_impl_all(df1):
        df2 = df1.groupby("B").agg(
            {"A": ["cumprod", "cumsum"], "C": ["cumprod", "cumsum"]}
        )
        return df2

    check_func(
        test_impl_basic,
        (df_index,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    check_func(
        test_impl_both,
        (df_index,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    check_func(
        test_impl_all,
        (df_index,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


def test_cumsum_random_index():
    def test_impl(df1):
        df2 = df1.groupby("B").cumsum()
        return df2

    def get_random_dataframe_A(n):
        eListA = []
        eListB = []
        for i in range(n):
            eValA = random.randint(1, 10)
            eValB = random.randint(1, 10)
            eListA.append(eValA)
            eListB.append(eValB)
        return pd.DataFrame({"A": eListA, "B": eListB})

    def get_random_dataframe_B(n):
        eListA = []
        eListB = []
        eListC = []
        for i in range(n):
            eValA = random.randint(1, 10)
            eValB = random.randint(1, 10)
            eValC = random.randint(1, 10) + 20
            eListA.append(eValA)
            eListB.append(eValB)
            eListC.append(eValC)
        return pd.DataFrame({"A": eListA, "B": eListB}, index=eListC)

    def get_random_dataframe_C(n):
        eListA = []
        eListB = []
        eListC = []
        for i in range(n):
            eValA = random.randint(1, 10)
            eValB = random.randint(1, 10)
            eValC = chr(random.randint(ord("a"), ord("z")))
            eListA.append(eValA)
            eListB.append(eValB)
            eListC.append(eValC)
        return pd.DataFrame({"A": eListA, "B": eListB}, index=eListC)

    random.seed(5)
    df1 = get_random_dataframe_A(100)
    df2 = get_random_dataframe_B(100)
    df3 = get_random_dataframe_C(100)

    check_func(test_impl, (df1,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(test_impl, (df2,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(test_impl, (df3,), sort_output=True, reset_index=True, check_dtype=False)


def test_sum_max_min_list_string_random():
    """Tests for columns being a list of strings.
    We have to use as_index=False since list of strings are mutable
    and index are immutable so cannot be an index"""

    def test_impl1(df1):
        df2 = df1.groupby("A", as_index=False).sum()
        return df2

    def test_impl2(df1):
        df2 = df1.groupby("A", as_index=False).max()
        return df2

    def test_impl3(df1):
        df2 = df1.groupby("A", as_index=False).min()
        return df2

    def test_impl4(df1):
        df2 = df1.groupby("A", as_index=False).first()
        return df2

    def test_impl5(df1):
        df2 = df1.groupby("A", as_index=False).last()
        return df2

    def test_impl6(df1):
        df2 = df1.groupby("A", as_index=False).count()
        return df2

    def test_impl7(df1):
        df2 = df1.groupby("A", as_index=False)["B"].agg(("sum", "min", "max", "last"))
        return df2

    def test_impl8(df1):
        df2 = df1.groupby("A", as_index=False).nunique()
        return df2

    random.seed(5)

    def rand_col_l_str(n):
        e_list_list = []
        for _ in range(n):
            if random.random() < 0.1:
                e_ent = np.nan
            else:
                e_ent = []
                for _ in range(random.randint(1, 2)):
                    k = random.randint(1, 3)
                    val = "".join(random.choices(["A", "B", "C"], k=k))
                    e_ent.append(val)
            e_list_list.append(e_ent)
        return e_list_list

    n = 10
    df1 = pd.DataFrame({"A": rand_col_l_str(n), "B": rand_col_l_str(n)})

    def check_fct(the_fct, df1, select_col_comparison):
        bodo_fct = bodo.jit(the_fct)
        # Computing images via pandas and pandas but applying the merging of columns
        df1_merge = convert_list_string_decimal_columns(df1)
        df2_merge_preA = the_fct(df1_merge)
        df2_merge_A = df2_merge_preA[select_col_comparison]
        df2_merge_preB = convert_list_string_decimal_columns(bodo_fct(df1))
        df2_merge_B = df2_merge_preB[select_col_comparison]
        # Now comparing the results.
        list_col_names = df2_merge_A.columns.to_list()
        df2_merge_A_sort = df2_merge_A.sort_values(by=list_col_names).reset_index(
            drop=True
        )
        df2_merge_B_sort = df2_merge_B.sort_values(by=list_col_names).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(
            df2_merge_A_sort, df2_merge_B_sort, check_dtype=False
        )
        # Now doing the parallel check
        check_parallel_coherency(the_fct, (df1,), sort_output=True, reset_index=True)

    # For nunique, we face the problem of difference of formatting between nunique
    # in Bodo and in Pandas.
    check_func_type_extent(test_impl1, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(test_impl2, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(test_impl3, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(test_impl4, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(test_impl5, (df1,), sort_output=True, reset_index=True)
    check_func_type_extent(test_impl6, (df1,), sort_output=True, reset_index=True)

    # For test_impl7, we have an error in as_index=False function, that is:
    # df1.groupby("A", as_index=False)["B"].agg(("sum", "min", "max"))
    #
    # The problem is that pandas does it in a way that we consider erroneous.
    check_fct(test_impl7, df1, ["sum", "min", "max", "last"])

    # For test_impl8 we face the problem that pandas returns a wrong column
    # for the A. multiplicities are given (always 1) instead of the values.
    check_fct(test_impl8, df1, ["B"])


def test_groupby_datetime_miss():
    """Testing the groupby with columns having datetime with missing entries
    TODO: need to support the cummin/cummax cases after pandas is corrected"""

    def test_impl1(df):
        A = df.groupby("A", as_index=False).min()
        return A

    def test_impl2(df):
        A = df.groupby("A", as_index=False).max()
        return A

    def test_impl3(df):
        A = df.groupby("A").first()
        return A

    def test_impl4(df):
        A = df.groupby("A").last()
        return A

    random.seed(5)

    def get_small_list(shift, elen):
        small_list_date = []
        for _ in range(elen):
            e_year = random.randint(shift, shift + 20)
            e_month = random.randint(1, 12)
            e_day = random.randint(1, 28)
            small_list_date.append(datetime.datetime(e_year, e_month, e_day))
        return small_list_date

    def get_random_entry(small_list):
        if random.random() < 0.2:
            return "NaT"
        else:
            pos = random.randint(0, len(small_list) - 1)
            return small_list[pos]

    n_big = 100
    col_a = []
    col_b = []
    small_list_a = get_small_list(1940, 5)
    small_list_b = get_small_list(1920, 20)
    for idx in range(n_big):
        col_a.append(get_random_entry(small_list_a))
        col_b.append(get_random_entry(small_list_b))
    df1 = pd.DataFrame({"A": pd.Series(col_a), "B": pd.Series(col_b)})

    check_func(
        test_impl1, (df1,), sort_output=True, check_dtype=False, reset_index=True
    )
    check_func(
        test_impl2, (df1,), sort_output=True, check_dtype=False, reset_index=True
    )
    # TODO: solve the bug below. We should not need to have a reset_index=True
    check_func(test_impl3, (df1,), sort_output=True, reset_index=True)
    check_func(test_impl4, (df1,), sort_output=True, reset_index=True)


def test_agg_as_index_fast():
    """
    Test Groupby.agg() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        A = df.groupby("A", as_index=False).agg(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl1, (df,), sort_output=True, check_dtype=False, reset_index=True)


@pytest.mark.slow
def test_agg_as_index():
    """
    Test Groupby.agg() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl2(df):
        A = df.groupby("A", as_index=False)["B"].agg(lambda x: x.max() - x.min())
        return A

    def impl3(df):
        A = df.groupby("A", as_index=False)["B"].agg({"B": "sum"})
        return A

    def impl3b(df):
        A = df.groupby(["A", "B"], as_index=False)["C"].agg({"C": "sum"})
        return A

    def impl4(df):
        def id1(x):
            return (x <= 2).sum()

        def id2(x):
            return (x > 2).sum()

        A = df.groupby("A", as_index=False)["B"].agg((id1, id2))
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    # disabled because this doesn't work in pandas 1.0 (looks like a bug)
    # check_func(impl2, (df,), sort_output=True, check_dtype=False)
    check_func(impl3, (df,), sort_output=True, reset_index=True)
    check_func(impl3b, (df,), sort_output=True, reset_index=True)

    # for some reason pandas does not make index a column with impl4:
    # https://github.com/pandas-dev/pandas/issues/25011
    pandas_df = impl4(df)
    pandas_df.reset_index(inplace=True)  # convert A index to column
    pandas_df = pandas_df.sort_values(by="A").reset_index(drop=True)
    bodo_df = bodo.jit(impl4)(df)
    bodo_df = bodo_df.sort_values(by="A").reset_index(drop=True)
    pd.testing.assert_frame_equal(pandas_df, bodo_df)


def test_agg_select_col_fast():
    """
    Test Groupby.agg() with explicitly select one (str)column
    """

    def impl_str(df):
        A = df.groupby("A")["B"].agg(lambda x: (x == "a").sum())
        return A

    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )

    check_func(impl_str, (df_str,), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_agg_select_col():
    """
    Test Groupby.agg() with explicitly select one column
    """

    def impl_num(df):
        A = df.groupby("A")["B"].agg(lambda x: x.max() - x.min())
        return A

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].agg(lambda x: x.max() - x.min())
        return A

    df_int = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    df_float = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2]}
    )
    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )
    check_func(
        impl_num, (df_int,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(
        impl_num, (df_float,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(test_impl, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_agg_no_parfor():
    """
    Test Groupby.agg(): simple UDF with no parfor
    """

    def impl1(df):
        A = df.groupby("A").agg(lambda x: 1)
        return A

    def impl2(df):
        A = df.groupby("A").agg(lambda x: len(x))
        return A

    check_func(
        impl1, (udf_in_df,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(
        impl2, (udf_in_df,), sort_output=True, reset_index=True, check_dtype=False
    )


def test_agg_len_mix():
    """
    Test Groupby.agg(): use of len() in a UDF mixed with another parfor
    """

    def impl(df):
        A = df.groupby("A").agg(lambda x: x.sum() / len(x))
        return A

    check_func(
        impl, (udf_in_df,), sort_output=True, reset_index=True, check_dtype=False
    )


def test_agg_multi_udf():
    """
    Test Groupby.agg() multiple user defined functions
    """

    def impl(df):
        def id1(x):
            return (x <= 2).sum()

        def id2(x):
            return (x > 2).sum()

        return df.groupby("A")["B"].agg((id1, id2))

    def impl2(df):
        def id1(x):
            return (x <= 2).sum()

        def id2(x):
            return (x > 2).sum()

        return df.groupby("A")["B"].agg(("var", id1, id2, "sum"))

    def impl3(df):
        return df.groupby("A")["B"].agg(
            (lambda x: x.max() - x.min(), lambda x: x.max() + x.min())
        )

    def impl4(df):
        return df.groupby("A")["B"].agg(("cumprod", "cumsum"))

    df = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    check_func(impl, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)
    # check_dtype=False since Bodo returns float for Series.min/max. TODO: fix min/max
    check_func(impl3, (df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl4, (df,), sort_output=True, reset_index=True)


def test_aggregate():
    """
    Test Groupby.aggregate(): one user defined func and all cols
    """

    def impl(df):
        A = df.groupby("A").aggregate(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl, (df,), sort_output=True, reset_index=True, check_dtype=False)


def test_aggregate_as_index():
    """
    Test Groupby.aggregate() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        A = df.groupby("A", as_index=False).aggregate(lambda x: x.max() - x.min())
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2],
        }
    )

    check_func(impl1, (df,), sort_output=True, check_dtype=False, reset_index=True)


def test_aggregate_select_col():
    """
    Test Groupby.aggregate() with explicitly select one column
    """

    def impl_num(df):
        A = df.groupby("A")["B"].aggregate(lambda x: x.max() - x.min())
        return A

    def impl_str(df):
        A = df.groupby("A")["B"].aggregate(lambda x: (x == "a").sum())
        return A

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].aggregate(lambda x: x.max() - x.min())
        return A

    df_int = pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1], "B": [1, 2, 3, 4, 5, 6, 7]})
    df_float = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": [1.2, 2.4, np.nan, 2.2, 5.3, 3.3, 7.2]}
    )
    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )
    check_func(
        impl_num, (df_int,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(
        impl_num, (df_float,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(
        impl_str, (df_str,), sort_output=True, reset_index=True, check_dtype=False
    )
    check_func(test_impl, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_groupby_agg_const_dict():
    """
    Test groupy.agg with function spec passed as constant dictionary
    """

    def impl(df):
        df2 = df.groupby("A")[["B", "C"]].agg({"B": "count", "C": "sum"})
        return df2

    def impl2(df):
        df2 = df.groupby("A").agg({"B": "count", "C": "sum"})
        return df2

    def impl3(df):
        df2 = df.groupby("A").agg({"B": "median"})
        return df2

    def impl4(df):
        df2 = df.groupby("A").agg({"B": ["median"]})
        return df2

    def impl5(df):
        df2 = df.groupby("A").agg({"D": "nunique", "B": "median", "C": "var"})
        return df2

    def impl6(df):
        df2 = df.groupby("A").agg({"B": ["median", "nunique"]})
        return df2

    def impl7(df):
        df2 = df.groupby("A").agg({"B": ["count", "var", "prod"], "C": ["std", "sum"]})
        return df2

    def impl8(df):
        df2 = df.groupby("A", as_index=False).agg(
            {"B": ["count", "var", "prod"], "C": ["std", "sum"]}
        )
        return df2

    def impl9(df):
        df2 = df.groupby("A").agg({"B": ["count", "var", "prod"], "C": "std"})
        return df2

    def impl10(df):
        df2 = df.groupby("A").agg({"B": ["count", "var", "prod"], "C": ["std"]})
        return df2

    def impl11(df):
        df2 = df.groupby("A").agg(
            {"B": ["count", "median", "prod"], "C": ["nunique", "sum"]}
        )
        return df2

    def impl12(df):
        def id1(x):
            return (x >= 2).sum()

        df2 = df.groupby("D").agg({"B": "var", "A": id1, "C": "sum"})
        return df2

    def impl13(df):
        df2 = df.groupby("D").agg({"B": lambda x: x.max() - x.min(), "A": "sum"})
        return df2

    def impl14(df):
        df2 = df.groupby("A").agg(
            {
                "D": lambda x: (x == "BB").sum(),
                "B": lambda x: x.max() - x.min(),
                "C": "sum",
            }
        )
        return df2

    def impl15(df):
        df2 = df.groupby("A").agg({"B": "cumsum", "C": "cumprod"})
        return df2

    # reuse a complex dict to test typing transform for const dict removal
    def impl16(df):
        d = {"B": [lambda a: a.sum(), "mean"]}
        df1 = df.groupby("A").agg(d)
        df2 = df.groupby("C").agg(d)
        return df1, df2

    # reuse and return a const dict to test typing transform
    def impl17(df):
        d = {"B": "sum"}
        df1 = df.groupby("A").agg(d)
        df2 = df.groupby("C").agg(d)
        return df1, df2, d

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "D": ["AA", "B", "BB", "B", "AA", "AA", "B"],
            "B": [-8.1, 2.1, 3.1, 1.1, 5.1, 6.1, 7.1],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)
    check_func(impl3, (df,), sort_output=True, reset_index=True)
    check_func(impl4, (df,), sort_output=True, reset_index=True)
    check_func(impl5, (df,), sort_output=True, reset_index=True)
    check_func(impl6, (df,), sort_output=True, reset_index=True)
    check_func(impl7, (df,), sort_output=True, reset_index=True)
    check_func(impl8, (df,), sort_output=True, reset_index=True)
    check_func(impl9, (df,), sort_output=True, reset_index=True)
    check_func(impl10, (df,), sort_output=True, reset_index=True)
    check_func(impl11, (df,), sort_output=True, reset_index=True)
    check_func(impl12, (df,), sort_output=True, reset_index=True)
    check_func(impl13, (df,), sort_output=True, reset_index=True)
    check_func(impl14, (df,), sort_output=True, reset_index=True)
    check_func(impl15, (df,), sort_output=True, reset_index=True)
    # can't use check_func since lambda name in MultiIndex doesn't match Pandas
    # TODO: fix lambda name
    # check_func(impl16, (df,), sort_output=True, reset_index=True)
    bodo.jit(impl16)(df)  # just check for compilation errors
    # TODO: enable is_out_distributed after fixing gatherv issue for tuple output
    check_func(impl17, (df,), sort_output=True, reset_index=True, dist_test=False)


def g(x):
    return (x == "a").sum()


@pytest.mark.slow
def test_agg_global_func():
    """
    Test Groupby.agg() with a global function as UDF
    """

    def impl_str(df):
        A = df.groupby("A")["B"].agg(g)
        return A

    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["a", "b", "c", "c", "b", "c", "a"]}
    )

    check_func(impl_str, (df_str,), sort_output=True, reset_index=True)


def test_count():
    """
    Test Groupby.count()
    """

    def impl1(df):
        A = df.groupby("A").count()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").count()
        return A

    df_int = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7],
            "C": [1.1, 2.4, 3.1, -1.9, 2.3, 3.0, -2.4],
        }
    )
    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_bool = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [True, np.nan, False, True, np.nan, False, False],
            "C": [True, True, False, True, True, False, False],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df_int,), sort_output=True, reset_index=True)
    check_func(impl1, (df_str,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl1, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_count_select_col():
    """
    Test Groupby.count() with explicitly select one column
    TODO: after groupby.count() properly ignores nulls, adds np.nan to df_str
    """

    def impl1(df):
        A = df.groupby("A")["B"].count()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].count()
        return A

    df_int = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7],
            "C": [1.1, 2.4, 3.1, -1.9, 2.3, 3.0, -2.4],
        }
    )
    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_bool = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [True, np.nan, False, True, np.nan, False, False],
            "C": [True, True, False, True, True, False, False],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df_int,), sort_output=True, reset_index=True)
    check_func(impl1, (df_str,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl1, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df_med",
    [
        pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]}),
        pd.DataFrame({"A": [1, 2, 2, 1, 1], "B": [1, 5, 4, 4, 3]}),
        pd.DataFrame({"A": [1, 1, 1, 1, 1], "B": [1, 2, 3, 4, np.nan]}),
    ],
)
def test_median_simple(df_med):
    """
    Test Groupby.median() with a single entry.
    """

    def impl1(df):
        A = df.groupby("A")["B"].median()
        return A

    check_func(impl1, (df_med,), sort_output=True, reset_index=True)


def test_median_large_random_numpy():
    """
    Test Groupby.median() with a large random numpy vector
    """

    def get_random_array(n, sizlen):
        elist = []
        for i in range(n):
            eval = random.randint(1, sizlen)
            if eval == 1:
                eval = None
            elist.append(eval)
        return np.array(elist, dtype=np.float64)

    def impl1(df):
        A = df.groupby("A")["B"].median()
        return A

    random.seed(5)
    nb = 100
    df1 = pd.DataFrame({"A": get_random_array(nb, 10), "B": get_random_array(nb, 100)})
    check_func(impl1, (df1,), sort_output=True, reset_index=True)


def test_median_nullable_int_bool():
    """
    Test Groupby.median() with a large random sets of nullable_int_bool
    """

    def impl1(df):
        df2 = df.groupby("A")["B"].median()
        return df2

    nullarr = pd.Series([1, 2, 3, 4, None, 1, 2], dtype="UInt16")
    df1 = pd.DataFrame({"A": [1, 1, 1, 1, 1, 2, 2], "B": nullarr})
    check_func(impl1, (df1,), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df_uniq",
    [
        pd.DataFrame(
            {"A": [2, 1, 1, 1, 2, 2, 1], "B": [-8, np.nan, 3, 1, np.nan, 6, 7]}
        ),
        pd.DataFrame(
            {
                "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
                "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
            }
        ),
    ],
)
def test_nunique_select_col(df_uniq):
    """
    Test Groupby.nunique() with explicitly selected one column. Boolean are broken in pandas so the
    test is removed.
    TODO: Implementation of Boolean test when pandas is corrected.
    """

    def impl1(df):
        A = df.groupby("A")["B"].nunique()
        return A

    def impl2(df):
        A = df.groupby("A")["B"].nunique(dropna=True)
        return A

    def impl3(df):
        A = df.groupby("A")["B"].nunique(dropna=False)
        return A

    check_func(impl1, (df_uniq,), sort_output=True, reset_index=True)
    check_func(impl2, (df_uniq,), sort_output=True, reset_index=True)
    check_func(impl3, (df_uniq,), sort_output=True, reset_index=True)


def test_nunique_select_col_missing_keys():
    """
    Test Groupby.nunique() with explicitly select one column. Some keys are missing
    for this test
    """

    def impl1(df):
        A = df.groupby("A")["B"].nunique()
        return A

    df_int = pd.DataFrame(
        {"A": [np.nan, 1, np.nan, 1, 2, 2, 1], "B": [-8, np.nan, 3, 1, np.nan, 6, 7]}
    )
    df_str = pd.DataFrame(
        {
            "A": [np.nan, "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
        }
    )
    check_func(impl1, (df_int,), sort_output=True, reset_index=True)
    check_func(impl1, (df_str,), sort_output=True, reset_index=True)


def test_filtered_count():
    """
    Test Groupby.count() with filtered dataframe
    """

    def test_impl(df, cond):
        df2 = df[cond]
        c = df2.groupby("A").count()
        return df2, c

    bodo_func = bodo.jit(test_impl)
    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, np.nan, 5, 6, 7],
            "C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    cond = df.A > 1
    res = test_impl(df, cond)
    h_res = bodo_func(df, cond)
    pd.testing.assert_frame_equal(res[0], h_res[0])
    pd.testing.assert_frame_equal(res[1], h_res[1])


def test_as_index_count():
    """
    Test Groupby.count() on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).count()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["C"].count()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, np.nan, 5, 6, 7],
            "C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_single_col_reset_index(test_df):
    def impl1(df):
        A = df.groupby("A")["B"].sum().reset_index()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)


def test_nonvar_column_names():
    """Test column names that cannot be variable names to make sure groupby code
    generation sanitizes variable names properly.
    """

    def impl1(df):
        A = df.groupby("A: A")["B: B"].sum()
        return A

    df = pd.DataFrame(
        {
            "A: A": [2, 1, 1, 1, 2, 2, 1],
            "B: B": [-8, 2, 3, np.nan, 5, 6, 7],
            "C: C": [2, 3, -1, 1, 2, 3, -1],
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)


def test_cumsum_large_random_numpy():
    def get_random_array(n, sizlen):
        elist = []
        for i in range(n):
            eval = random.randint(1, sizlen)
            if eval == 1:
                eval = None
            elist.append(eval)
        return np.array(elist, dtype=np.float64)

    def impl1(df):
        A = df.groupby("A")["B"].cumsum()
        return A

    def impl2(df):
        A = df.groupby("A")["B"].cumsum(skipna=True)
        return A

    def impl3(df):
        A = df.groupby("A")["B"].cumsum(skipna=False)
        return A

    random.seed(5)
    nb = 100
    df1 = pd.DataFrame({"A": get_random_array(nb, 10), "B": get_random_array(nb, 100)})
    check_func(impl1, (df1,), sort_output=True, reset_index=True)
    check_func(impl2, (df1,), sort_output=True, reset_index=True)
    check_func(impl3, (df1,), sort_output=True, reset_index=True)


def test_cummin_cummax_large_random_numpy():
    def get_random_array(n, sizlen):
        elist = []
        for i in range(n):
            eval = random.randint(1, sizlen)
            if eval == 1:
                eval = None
            elist.append(eval)
        return np.array(elist, dtype=np.float64)

    def impl1(df):
        A = df.groupby("A")["B"].agg(("cummin", "cummax"))
        return A

    def impl2(df):
        A = df.groupby("A").cummin()
        return A

    def impl3(df):
        A = df.groupby("A").cummax()
        return A

    def impl4(df):
        A = df.groupby("A")["B"].cummin()
        return A

    def impl5(df):
        A = df.groupby("A")["B"].cummax()
        return A

    def impl6(df):
        A = df.groupby("A").agg({"B": "cummin"})
        return A

    # The as_index option has no bearing for cumulative operations but better be safe than sorry.
    def impl7(df):
        A = df.groupby("A", as_index=True)["B"].cummin()
        return A

    # ditto
    def impl8(df):
        A = df.groupby("A", as_index=False)["B"].cummin()
        return A

    random.seed(5)
    nb = 100
    df1 = pd.DataFrame({"A": get_random_array(nb, 10), "B": get_random_array(nb, 100)})
    check_func(impl1, (df1,), sort_output=True, reset_index=True)
    check_func(impl2, (df1,), sort_output=True, reset_index=True)
    check_func(impl3, (df1,), sort_output=True, reset_index=True)
    check_func(impl4, (df1,), sort_output=True, reset_index=True)
    check_func(impl5, (df1,), sort_output=True, reset_index=True)
    check_func(impl6, (df1,), sort_output=True, reset_index=True)
    check_func(impl7, (df1,), sort_output=True, reset_index=True)
    check_func(impl8, (df1,), sort_output=True, reset_index=True)


def test_groupby_cumsum_simple():
    """
    Test Groupby.cumsum(): a simple case
    """

    def impl(df):
        df2 = df.groupby("A")["B"].cumsum()
        return df2

    df1 = pd.DataFrame({"A": [1, 1, 1, 1, 1], "B": [1, 2, 3, 4, 5]})
    check_func(impl, (df1,), sort_output=True, reset_index=True)


def test_groupby_cumprod_simple():
    """
    Test Groupby.cumprod(): a simple case
    """

    def impl(df):
        df2 = df.groupby("A")["B"].cumprod()
        return df2

    df1 = pd.DataFrame({"A": [1, 1, 1, 1, 1], "B": [1, 2, 3, 4, 5]})
    check_func(impl, (df1,), sort_output=True, reset_index=True)


def test_groupby_cumsum():
    """
    Test Groupby.cumsum()
    """

    def impl1(df):
        df2 = df.groupby("A").cumsum(skipna=False)
        return df2

    def impl2(df):
        df2 = df.groupby("A").cumsum(skipna=True)
        return df2

    df1 = pd.DataFrame(
        {
            "A": [0, 1, 3, 2, 1, 0, 4, 0, 2, 0],
            "B": [-8, np.nan, 3, 1, np.nan, 6, 7, 3, 1, 2],
            "C": [-8, 2, 3, 1, 5, 6, 7, 3, 1, 2],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "B": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    df3 = pd.DataFrame(
        {
            "A": [0.3, np.nan, 3.5, 0.2, np.nan, 3.3, 0.2, 0.3, 0.2, 0.2],
            "B": [-1.1, 1.1, 3.2, 1.1, 5.2, 6.8, 7.3, 3.4, 1.2, 2.4],
            "C": [-8.1, 2.3, 5.3, 1.1, 0.5, 4.6, 1.7, 4.3, -8.1, 5.3],
        }
    )
    check_func(impl1, (df1,), sort_output=True, reset_index=True)
    check_func(impl1, (df2,), sort_output=True, reset_index=True)
    check_func(impl1, (df3,), sort_output=True, reset_index=True)
    check_func(impl2, (df1,), sort_output=True, reset_index=True)
    check_func(impl2, (df2,), sort_output=True, reset_index=True)
    check_func(impl2, (df3,), sort_output=True, reset_index=True)


def test_groupby_multi_intlabels_cumsum_int():
    """
    Test Groupby.cumsum() on int columns
    multiple labels for 'by'
    """

    def impl(df):
        df2 = df.groupby(["A", "B"])["C"].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 1, -8, 1, 5, 1, 7],
            "C": [3, np.nan, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_groupby_multi_labels_cumsum_multi_cols():
    """
    Test Groupby.cumsum() 
    multiple labels for 'by', multiple cols to cumsum
    """

    def impl(df):
        df2 = df.groupby(["A", "B"])[["C", "D"]].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, 5, 6, 5, 4, 4, 3],
            "D": [3.1, 1.1, 6.0, np.nan, 4.0, np.nan, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_groupby_as_index_cumsum():
    """
    Test Groupby.cumsum() on groupby() as_index=False
    for both dataframe and series returns
    TODO: add np.nan to "A" after groupby null keys are properly ignored
          for cumsum
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).cumsum()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["C"].cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [3.0, 1.0, 4.1, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, np.nan, 6, 5, 4, 4, 3],
            "D": [3.1, 1.1, 6.0, np.nan, 4.0, np.nan, 3],
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_cumsum_all_nulls_col():
    """
    Test Groupby.cumsum() on column with all null entries
    TODO: change by to "A" after groupby null keys are properly ignored
          for cumsum
    """

    def impl(df):
        df2 = df.groupby("B").cumsum()
        return df2

    df = pd.DataFrame(
        {
            "A": [np.nan, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0],
            "B": [1, 2, 3, 2, 1, 1, 1],
            "C": [3, 5, 6, 5, 4, 4, 3],
            "D": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_max(test_df):
    """
    Test Groupby.max()
    """

    def impl1(df):
        A = df.groupby("A").max()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").max()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,))


def test_max_one_col(test_df):
    """
    Test Groupby.max() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].max()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].max()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    # seems like Pandas 1.0 has a regression and returns float64 for Int64 in this case
    check_dtype = True
    if pd.Int64Dtype() in test_df.dtypes.to_list():
        check_dtype = False
    check_func(
        impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=check_dtype
    )
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,))


def test_groupby_as_index_max():
    """
    Test max on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).max()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].max()
        return df2

    check_func(impl1, (11,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_max_datetime():
    """
    Test Groupby.max() on datetime column
    for both dataframe and series returns
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).max()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["B"].max()
        return df2

    df = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_mean(test_df):
    """
    Test Groupby.mean()
    """

    def impl1(df):
        A = df.groupby("A").mean()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").mean()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_mean_one_col(test_df):
    """
    Test Groupby.mean() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].mean()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].mean()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_groupby_as_index_mean():
    """
    Test mean on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).mean()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].mean()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False, reset_index=True)
    check_func(impl2, (11,), sort_output=True, check_dtype=False, reset_index=True)


def test_min(test_df):
    """
    Test Groupby.min()
    """

    def impl1(df):
        A = df.groupby("A").min()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").min()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_min_one_col(test_df):
    """
    Test Groupby.min() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].min()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].min()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    # seems like Pandas 1.0 has a regression and returns float64 for Int64 in this case
    check_dtype = True
    if pd.Int64Dtype() in test_df.dtypes.to_list():
        check_dtype = False
    check_func(
        impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=check_dtype
    )
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_groupby_as_index_min():
    """
    Test min on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).min()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].min()
        return df2

    check_func(impl1, (11,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_min_datetime():
    """
    Test Groupby.min() on datetime column
    for both dataframe and series returns
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).min()
        return df2

    def impl2(df):
        df2 = df.groupby("A", as_index=False)["B"].min()
        return df2

    df = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_prod(test_df):
    """
    Test Groupby.prod()
    """

    def impl1(df):
        A = df.groupby("A").prod()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").prod()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            # This column is disabled because pandas removes it
            # from output. This could be a bug in pandas. TODO: enable when it
            # is fixed
            # "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_prod_one_col(test_df):
    """
    Test Groupby.prod() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].prod()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].prod()
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "C": [True, np.nan, False, True, np.nan, False, False, True],
            "B": [True, True, False, True, True, False, False, False],
        }
    )

    # seems like Pandas 1.0 has a regression and returns float64 for Int64 in this case
    check_dtype = True
    if pd.Int64Dtype() in test_df.dtypes.to_list():
        check_dtype = False
    check_func(
        impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=check_dtype
    )
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_groupby_as_index_prod():
    """
    Test prod on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).prod()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].prod()
        return df2

    check_func(impl1, (11,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_first_last(test_df):
    """
    Test Groupby.first() and Groupby.last()
    """

    def impl1(df):
        A = df.groupby("A").first()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").first()
        return A

    def impl3(df):
        A = df.groupby("A").last()
        return A

    def impl4(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").last()
        return A

    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )
    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl1, (df_str,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl1, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)
    check_func(impl3, (test_df,), sort_output=True, reset_index=True)
    check_func(impl3, (df_str,), sort_output=True, reset_index=True)
    check_func(impl3, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl3, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl4, (11,), sort_output=True, reset_index=True)


def test_first_last_one_col(test_df):
    """
    Test Groupby.first() and Groupby.last() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].first()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].first()
        return A

    def impl3(df):
        A = df.groupby("A")["B"].last()
        return A

    def impl4(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].last()
        return A

    df_str = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", np.nan, "bb", "aa", np.nan, "ggg", "rr"],
            "C": ["cc", "aa", "aa", "bb", "vv", "cc", "cc"],
        }
    )
    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    df_dt = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
    )

    # seems like Pandas 1.0 has a regression and returns float64 for Int64 in this case
    check_dtype = True
    if pd.Int64Dtype() in test_df.dtypes.to_list():
        check_dtype = False
    check_func(
        impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=check_dtype
    )
    check_func(impl1, (df_str,), sort_output=True, reset_index=True)
    check_func(impl1, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl1, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)
    check_func(
        impl3, (test_df,), sort_output=True, reset_index=True, check_dtype=check_dtype
    )
    check_func(impl3, (df_str,), sort_output=True, reset_index=True)
    check_func(impl3, (df_bool,), sort_output=True, reset_index=True)
    check_func(impl3, (df_dt,), sort_output=True, reset_index=True)
    check_func(impl4, (11,), sort_output=True, reset_index=True)


def test_groupby_as_index_first_last():
    """
    Test first and last on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).first()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].first()
        return df2

    def impl3(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).last()
        return df2

    def impl4(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].last()
        return df2

    check_func(impl1, (11,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)
    check_func(impl3, (11,), sort_output=True, reset_index=True)
    check_func(impl4, (11,), sort_output=True, reset_index=True)


def test_std(test_df_int_no_null):
    """
    Test Groupby.std()
    """

    def impl1(df):
        # NOTE: pandas fails here if one of the data columns is Int64 with all
        # nulls. That is why this test uses test_df_int_no_null
        A = df.groupby("A").std()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").std()
        return A

    check_func(
        impl1,
        (test_df_int_no_null,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_std_one_col(test_df):
    """
    Test Groupby.std() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].std()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].std()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_groupby_as_index_std():
    """
    Test std on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).std()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].std()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False, reset_index=True)
    check_func(impl2, (11,), sort_output=True, check_dtype=False, reset_index=True)


def test_sum(test_df):
    """
    Test Groupby.sum()
    """

    def impl1(df):
        A = df.groupby("A").sum()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").sum()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_sum_one_col(test_df):
    """
    Test Groupby.sum() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].sum()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].sum()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_select_col_attr():
    """
    Test Groupby with column selected using getattr instead of getitem
    """

    def impl(df):
        A = df.groupby("A").B.sum()
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_groupby_as_index_sum():
    """
    Test sum on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).sum()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].sum()
        return df2

    check_func(impl1, (11,), sort_output=True, reset_index=True)
    check_func(impl2, (11,), sort_output=True, reset_index=True)


def test_groupby_multi_intlabels_sum():
    """
    Test df.groupby() multiple labels of string columns
    and Groupy.sum() on integer column
    """

    def impl(df):
        A = df.groupby(["A", "C"])["B"].sum()
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_groupby_multi_key_to_index():
    """
    Make sure df.groupby() with multiple keys creates a MultiIndex index in output
    """

    def impl(df):
        A = df.groupby(["A", "C"])["B"].sum()
        return A

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    # not using check_func(... sort_output=True) since it drops index, but we need to
    # make sure proper index is being created
    # TODO: avoid dropping index in check_func(... sort_output=True) when indexes are
    # supported properly for various APIs
    pd.testing.assert_series_equal(
        bodo.jit(impl)(df).sort_index(), impl(df).sort_index()
    )


def test_groupby_multi_strlabels():
    """
    Test df.groupby() multiple labels of string columns
    with as_index=False, and Groupy.sum() on integer column
    """

    def impl(df):
        df2 = df.groupby(["A", "B"], as_index=False)["C"].sum()
        return df2

    df = pd.DataFrame(
        {
            "A": ["aa", "b", "b", "b", "aa", "aa", "b"],
            "B": ["ccc", "a", "a", "aa", "ccc", "ggg", "a"],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl, (df,), sort_output=True, reset_index=True)


def test_groupby_multiselect_sum():
    """
    Test groupy.sum() on explicitly selected columns using a tuple and using a constant
    list (#198)
    """

    def impl1(df):
        df2 = df.groupby("A")["B", "C"].sum()
        return df2

    def impl2(df):
        df2 = df.groupby("A")[["B", "C"]].sum()
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_agg_multikey_parallel():
    """
    Test groupby multikey with distributed df
    """

    def test_impl(df):
        A = df.groupby(["A", "C"])["B"].sum()
        return A.sum()

    bodo_func = bodo.jit(distributed_block=["df"])(test_impl)
    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    start, end = get_start_end(len(df))
    h_res = bodo_func(df.iloc[start:end])
    p_res = test_impl(df)
    assert h_res == p_res


def test_var(test_df):
    """
    Test Groupby.var()
    """

    def impl1(df):
        A = df.groupby("A").var()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A").var()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_var_one_col(test_df):
    """
    Test Groupby.var() with one column selected
    """

    def impl1(df):
        A = df.groupby("A")["B"].var()
        return A

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        A = df.groupby("A")["B"].var()
        return A

    check_func(impl1, (test_df,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (11,), sort_output=True, reset_index=True, check_dtype=False)


def test_groupby_as_index_var():
    """
    Test var on groupby() as_index=False
    for both dataframe and series returns
    """

    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False).var()
        return df2

    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.arange(n)})
        df2 = df.groupby("A", as_index=False)["B"].var()
        return df2

    check_func(impl1, (11,), sort_output=True, check_dtype=False, reset_index=True)
    check_func(impl2, (11,), sort_output=True, check_dtype=False, reset_index=True)


def test_const_list_inference():
    """
    Test passing non-const list that can be inferred as constant to groupby()
    """

    def impl1(df):
        return df.groupby(["A"] + ["B"]).sum()

    def impl2(df):
        return df.groupby(list(set(df.columns) - set(["A", "C"]))).sum()

    # test df schema change by setting a column
    def impl3(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        df["D"] = 4
        return df.groupby("D").sum()

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": ["a", "b", "c", "c", "b", "c", "a"],
            "C": [1, 3, 1, 2, -4, 0, 5],
        }
    )

    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)
    check_func(impl3, (11,), sort_output=True, reset_index=True)


# global key list for groupby() testing
g_keys = ["A", "B"]


def test_global_list():
    """
    Test passing a global list to groupby()
    """

    # freevar key list for groupby() testing
    f_keys = ["A", "B"]

    # global case
    def impl1(df):
        return df.groupby(g_keys).sum()

    # freevar case
    def impl2(df):
        return df.groupby(f_keys).sum()

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": ["a", "b", "c", "c", "b", "c", "a"],
            "C": [1, 3, 1, 2, -4, 0, 5],
        }
    )

    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_literal_args():
    """
    Test forcing groupby() key list and as_index to be literals if jit arguments
    """

    # 'by' arg
    def impl1(df, keys):
        return df.groupby(keys).sum()

    # both 'by' and 'as_index'
    def impl2(df, keys, as_index):
        return df.groupby(by=keys, as_index=as_index).sum()

    # computation on literal arg
    def impl3(df, keys, as_index):
        return df.groupby(by=keys + ["B"], as_index=as_index).sum()

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [1, 3, 3, 1, 3, 1, 3],
            "C": [1, 3, 1, 2, -4, 0, 5],
        }
    )

    check_func(impl1, (df, ["A", "B"]), sort_output=True, reset_index=True)
    check_func(impl2, (df, "A", False), sort_output=True, reset_index=True)
    check_func(impl2, (df, ["A", "B"], True), sort_output=True, reset_index=True)
    check_func(impl3, (df, ["A"], True), sort_output=True, reset_index=True)


def test_schema_change():
    """
    Test df schema change for groupby() to make sure errors are not thrown
    """

    # schema change in dict agg case
    def impl1(df):
        df["AA"] = np.arange(len(df))
        return df.groupby(["A"]).agg({"AA": "sum", "B": "count"})

    # schema change for groupby object getitem
    def impl2(df):
        df["AA"] = np.arange(len(df))
        return df.groupby(["A"]).AA.sum()

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": ["a", "b", "c", "c", "b", "c", "a"],
            "C": [1, 3, 1, 2, -4, 0, 5],
        }
    )

    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


# ------------------------------ pivot, crosstab ------------------------------ #


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


def test_pivot():
    def test_impl(df):
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        return (pt.small.values, pt.large.values)

    def test_impl2(df):
        pt = df.pivot_table(index="A", columns="C", values="D")
        return (pt.small.values, pt.large.values)

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(test_impl)
    assert set(bodo_func(_pivot_df1)[0]) == set(test_impl(_pivot_df1)[0])
    assert set(bodo_func(_pivot_df1)[1]) == set(test_impl(_pivot_df1)[1])

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(test_impl2)
    assert set(bodo_func(_pivot_df1)[0]) == set(test_impl2(_pivot_df1)[0])
    assert set(bodo_func(_pivot_df1)[1]) == set(test_impl2(_pivot_df1)[1])


def test_pivot_parallel(datapath):
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = df.pivot_table(index="A", columns="C", values="D", aggfunc="sum")
        res = pt.small.values.sum()
        return res

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(impl)
    assert bodo_func() == impl()


def test_crosstab():
    def test_impl(df):
        pt = pd.crosstab(df.A, df.C)
        return (pt.small.values, pt.large.values)

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(test_impl)
    assert set(bodo_func(_pivot_df1)[0]) == set(test_impl(_pivot_df1)[0])
    assert set(bodo_func(_pivot_df1)[1]) == set(test_impl(_pivot_df1)[1])


def test_crosstab_parallel(datapath):
    fname = datapath("pivot2.pq")

    def impl():
        df = pd.read_parquet(fname)
        pt = pd.crosstab(df.A, df.C)
        res = pt.small.values.sum()
        return res

    bodo_func = bodo.jit(pivots={"pt": ["small", "large"]})(impl)
    assert bodo_func() == impl()
