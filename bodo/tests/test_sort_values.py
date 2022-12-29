# Copyright (C) 2022 Bodo Inc. All rights reserved.

"""Test sort_values opration as called as df.sort_values()
   The C++ implementation uses the timsort which is a stable sort algorithm.
   Therefore, in the test we use mergesort, which guarantees that the equality
   tests can be made sensibly.
   ---
   The alternative is to use reset_index=True so that possible difference in sorting
   would be eliminated.
"""

import os
import random
import re
import string

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    check_func,
    check_parallel_coherency,
    gen_nonascii_list,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_array_struct_list_int,
    gen_random_arrow_list_list_decimal,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    gen_random_decimal_array,
    gen_random_list_string_array,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError, BodoWarning


@pytest.fixture(
    params=[
        # int series, float, and bool columns
        # TODO: change to "A": pd.Series([1, 8, 4, np.nan, 3], dtype="Int32")
        # after string column with nans is properly sorted
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series([1, 8, 4, 10, 3], dtype="Int32"),
                    "B": [1.1, np.nan, 4.2, 3.1, -1.3],
                    "C": [True, False, False, np.nan, True],
                },
                range(0, 5, 1),
            ),
            marks=pytest.mark.skip,
            # TODO: remove skip mark after remove none as index, PR #407
        ),
        # uint8, float32 dtypes, datetime index
        pd.DataFrame(
            {
                "A": np.array([1, 8, 4, 0, 3], dtype=np.uint8),
                "B": pd.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype="Float32"),
            },
            pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
        ),
        # bool list, numpy array
        # TODO: change to "A": [True, False, False, np.nan, True])
        # after string column with nans is properly sorted
        # and a Series(bool list) test too
        pytest.param(
            pd.DataFrame(
                {
                    "A": [True, False, False, True, True],
                    "B": np.array([1, 0, 4, -100, 11], dtype=np.int64),
                }
            ),
            marks=pytest.mark.skip,
            # TODO: remove skip mark after boolean shuffle properly handled
        ),
        # string and int columns, float index
        # TODO: change to "A": ["AA", np.nan, "", "D", "GG"]
        # after string column with nans is properly sorted
        # and a Series(str list) test too
        pd.DataFrame(
            {
                "A": ["AA", "AA", "", "D", "GG", "B", "ZZ", "K2", "F123"],
                "B": [1, 8, 4, -1, 2, 11, 3, 19, 14],
            },
            [-2.1, 0.1, 1.1, 7.1, 9.0, 1.2, -3.0, -1.2, 0.2],
        ),
        # TODO: parallel range index with start != 0 and stop != 1
        # datetime columns, int index
        pd.DataFrame(
            {
                "A": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
                "B": pd.date_range(start="2013-09-04", end="2013-09-29", periods=5),
                "C": [1.1, np.nan, 4.2, 3.1, -1.3],
                "D": pd.array([1.1, None, 4.2, 3.1, -1.3], "Float64"),
            },
            [-2, 1, 3, 5, 9],
        ),
        # Categorical columns (all with ordered=True)
        pd.DataFrame(
            {
                # Make sure there are no duplicates for consistent, comparable results
                "A": pd.Categorical(["AA", "BB", "", "C", None], ordered=True),
                "B": pd.Categorical([1, 2, 4, None, 5], ordered=True),
                "C": pd.Categorical(
                    pd.concat(
                        [
                            pd.Series(
                                pd.date_range(
                                    start="2/1/2015", end="2/24/2021", periods=4
                                )
                            ),
                            pd.Series(data=[None], index=[4]),
                        ]
                    ),
                    ordered=True,
                ),
                "D": pd.Categorical(
                    pd.concat(
                        [
                            pd.Series(pd.timedelta_range(start="1 day", periods=4)),
                            pd.Series(data=[None], index=[4]),
                        ]
                    ),
                    ordered=True,
                ),
            }
        ),
        # Binary Columns with nan
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        b"AA",
                        b"AA",
                        b"",
                        b"D",
                        np.nan,
                        b"B",
                        b"ZZ",
                        np.nan,
                        b"F123",
                    ],
                    "B": [
                        b"jkasdf",
                        b"asdfas",
                        np.nan,
                        b"D",
                        np.nan,
                        b"asdgas",
                        b"sdga",
                        b"sdaladnc",
                        b"sdasdan",
                    ],
                    "C": [
                        b"hjksda",
                        b"sdvnds",
                        b"",
                        b"asdjgka",
                        b"",
                        b"Basasd",
                        b"asldfasdf",
                        b"asdjflas",
                        b"sasdal",
                    ],
                },
            ),
            id="binary_df",
        )
        # TODO: timedelta
    ]
)
def df_value(request):
    return request.param


@pytest.mark.slow
def test_sort_datetime_missing(is_slow_run, memory_leak_check):
    """Test the datetime for missing entries"""

    def test_impl1(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="first", kind="mergesort"
        )
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(
            by="A", ascending=False, na_position="first", kind="mergesort"
        )
        return df2

    def test_impl3(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort"
        )
        return df2

    def test_impl4(df1):
        df2 = df1.sort_values(
            by="A", ascending=False, na_position="last", kind="mergesort"
        )
        return df2

    len_period = 400
    list_date = pd.date_range(start="2000-01-01", periods=len_period)
    np.random.seed(5)
    e_list = []
    for idx in range(len_period):
        if np.random.random() < 0.2:
            e_ent = "NaT"
        else:
            e_ent = list_date[idx]
        e_list.append(e_ent)

    df1 = pd.DataFrame({"A": e_list})

    check_func(
        test_impl1,
        (df1,),
    )
    if not is_slow_run:
        return
    check_func(
        test_impl2,
        (df1,),
    )
    check_func(
        test_impl3,
        (df1,),
    )
    check_func(
        test_impl4,
        (df1,),
    )


@pytest.mark.smoke
def test_single_col(memory_leak_check):
    """
    sorts a dataframe that has only one column
    """
    fname = os.path.join("bodo", "tests", "data", "kde.parquet")

    def test_impl():
        df = pd.read_parquet(fname)
        df.sort_values("points", inplace=True)
        res = df.points.values
        return res

    check_func(
        test_impl,
        (),
    )


@pytest.mark.slow
def test_sort_values_val(memory_leak_check):
    """
    Test sort_values(): with just 1 column
    return value is a list(i.e. without columns)
    """

    def impl(df):
        return df.sort_values(by=3, kind="mergesort")[3].values

    n = 10
    df = pd.DataFrame({3: np.arange(n) + 1.0, "B": np.arange(n) + 1})
    check_func(impl, (df,))


@pytest.mark.slow
def test_sort_values_tuple_keys(memory_leak_check):
    """
    Test sort_values() where column names are tuples
    """

    def impl1(df):
        df2 = df.groupby("A", as_index=False).agg({"B": ["sum", "mean"]})
        return df2.sort_values(by="A")

    def impl2(df):
        df2 = df.groupby("A", as_index=False).agg({"B": ["sum", "mean"]})
        return df2.sort_values(by=("A", ""))

    n = 10
    df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
    check_func(impl1, (df,), check_dtype=False, dist_test=False)
    check_func(impl2, (df,), check_dtype=False, dist_test=False)


@pytest.mark.slow
def test_sort_values_1col(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column
    """

    def impl(df):
        return df.sort_values(by="A", kind="mergesort")

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_sort_values_1col_ascending(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column and ascending=True
    """

    def impl(df):
        return df.sort_values(by="A", kind="mergesort", ascending=True)

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_sort_values_1col_inplace(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column
    """

    def impl(df):
        df.sort_values(by="A", kind="mergesort", inplace=True)
        return df

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    # inplace sort not supported for dict-encoded string arrays
    check_func(impl, (df_value,), use_dict_encoded_strings=False)


@pytest.mark.slow
def test_sort_values_2col(df_value, memory_leak_check):
    """
    Test sort_values(): with 2 columns
    """

    def impl(df):
        return df.sort_values(by=["A", "B"], kind="mergesort", ascending=[True, False])

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_sort_values_2col_inplace(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column
    """

    def impl(df):
        df.sort_values(
            by=["A", "B"], kind="mergesort", ascending=[True, False], inplace=True
        )
        return df

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    # inplace sort not supported for dict-encoded string arrays
    check_func(impl, (df_value,), use_dict_encoded_strings=False)


@pytest.mark.slow
def test_sort_values_str(memory_leak_check):
    """
    Test sort_values():
    dataframe has int column, and str column with nans
    sort over int columm
    """

    def test_impl(df):
        return df.sort_values(by="A", kind="mergesort")

    def _gen_df_str(n):
        str_vals = []
        for _ in range(n):
            # store NA with 30% chance
            if random.random() < 0.3:
                str_vals.append(np.nan)
                continue

            k = random.randint(1, 10)
            k2 = random.randint(1, 10)

            nonascii_val = " ".join(random.sample(gen_nonascii_list(k2), k2))
            val = nonascii_val.join(
                random.choices(string.ascii_uppercase + string.digits, k=k)
            )

            str_vals.append(val)

        A = np.random.randint(0, 1000, n)
        df = pd.DataFrame({"A": A, "B": str_vals}).drop_duplicates("A")
        return df

    random.seed(5)
    np.random.seed(3)
    # seeds should be the same on different processors for consistent input
    n = 17  # 1211
    df = _gen_df_str(n)
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_sort_values_binary(memory_leak_check):
    """
    Test sort_values():
    dataframe has int column, and binary column with nans
    sort over int columm
    """

    def test_impl(df):
        return df.sort_values(by="A", kind="mergesort")

    def _gen_df_binary(n):
        bytes_vals = []
        for _ in range(n):
            # store NA with 30% chance
            if np.random.randint(0, 10) < 3:
                bytes_vals.append(np.nan)
                continue

            val = bytes(np.random.randint(1, 100))
            bytes_vals.append(val)

        A = np.random.randint(0, 1000, n)
        df = pd.DataFrame({"A": A, "B": bytes_vals}).drop_duplicates("A")
        return df

    np.random.seed(3)
    # seeds should be the same on different processors for consistent input
    n = 17
    df = _gen_df_binary(n)
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_sort_values_1col_long_int_list(memory_leak_check):
    """
    Test sort_values(): with 1 longer int column
    """

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by="A", ascending=False, kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = []
        for i in range(n):
            eVal = i * i % 34
            eListA.append(eVal)
        return pd.DataFrame({"A": eListA})

    n = 10
    check_func(test_impl1, (get_quasi_random(n),))
    check_func(test_impl2, (get_quasi_random(n),))


@pytest.mark.slow
def test_sort_values_2col_long_np(memory_leak_check):
    """
    Test sort_values(): with just 2 longer int columns
    """

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by=["A", "B"], kind="mergesort")
        return df2

    def get_quasi_random(n):
        eListA = np.array([0] * n, dtype=np.uint64)
        eListB = np.array([0] * n, dtype=np.uint64)
        for i in range(n):
            eValA = i * i % 34
            eValB = i * i * i % 34
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})

    n = 100
    check_func(test_impl1, (get_quasi_random(n),))
    check_func(test_impl2, (get_quasi_random(n),))


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_sort_values_1col_np_array(dtype, memory_leak_check):
    """
    Test sort_values(): with just one column
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_dtype(n, dtype):
        eListA = np.array([0] * n, dtype=dtype)
        for i in range(n):
            eVal = i * i % 34
            eListA[i] = eVal
        return pd.DataFrame({"A": eListA})

    n = 100
    check_func(
        test_impl,
        (get_quasi_random_dtype(n, dtype),),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype1, dtype2",
    [
        (np.int8, np.int16),
        (np.uint8, np.int32),
        (np.int16, np.float64),
        (np.uint16, np.float32),
        ("Float32", "Float64"),
    ],
)
@pytest.mark.slow
def test_sort_values_2col_pd_array(dtype1, dtype2, memory_leak_check):
    """
    Test sort_values(): with two columns, two types
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_dtype(n, dtype1, dtype2):
        eListA = pd.array([0] * n, dtype=dtype1)
        eListB = pd.array([0] * n, dtype=dtype2)
        for i in range(n):
            eValA = i * i % 34
            eValB = i * (i - 1) % 23
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})

    n = 1000
    check_func(
        test_impl,
        (get_quasi_random_dtype(n, dtype1, dtype2),),
    )


@pytest.mark.parametrize(
    "n, len_str", [pytest.param(1000, 2, marks=pytest.mark.slow), (100, 1), (300, 2)]
)
def test_sort_values_strings_constant_length(n, len_str, memory_leak_check):
    """
    Test sort_values(): with 1 column and strings of constant length
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            val = "".join(random.choices(string.ascii_uppercase, k=len_str))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    random.seed(5)
    check_func(
        test_impl,
        (get_random_strings_array(n, len_str),),
    )


@pytest.mark.parametrize(
    "n, len_str", [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (10, 30)]
)
def test_sort_values_strings_variable_length(n, len_str, memory_leak_check):
    """
    Test sort_values(): with 1 column and strings of variable length all of character A.
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_var_length_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            k = random.randint(1, len_str)
            val = "A" * k
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    random.seed(5)
    df1 = get_random_var_length_strings_array(n, len_str)
    check_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_str",
    [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (100, 30)],
)
def test_sort_values_strings(n, len_str, memory_leak_check):
    """
    Test sort_values(): with 1 column and strings of variable length and variable characters.
    with some entries assigned to missing values
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        str_vals = []
        for _ in range(n):
            prob = random.randint(1, 10)
            if prob == 1:
                val = np.nan
            else:
                k = random.randint(1, len_str)
                k2 = len_str - k

                nonascii_val = " ".join(random.sample(gen_nonascii_list(k2), k2))
                val = nonascii_val.join(random.choices(string.ascii_uppercase, k=k))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    random.seed(5)
    df1 = get_random_strings_array(n, len_str)
    check_func(test_impl, (df1,))


def test_sort_random_values_binary():
    """
    Test sort_values(): with 1 column of random binary values with
    some entries assigned to missing values
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_bin_df(n):
        bin_vals = []
        for _ in range(n):
            prob = np.random.randint(1, 10)
            if prob == 1:
                val = np.nan
            else:
                val = bytes(np.random.randint(1, 100))

            bin_vals.append(val)
        df = pd.DataFrame({"A": bin_vals})
        return df

    np.random.seed(5)
    df1 = get_random_bin_df(100)
    check_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_siz", [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (10, 30)]
)
def test_sort_values_two_columns_nan(n, len_siz, memory_leak_check):
    """Test with two columns with some NaN entries, sorting over one column"""

    def test_impl1(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort", axis=0
        )
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="first", kind="mergesort", axis=0
        )
        return df2

    def test_impl3(df1):
        df2 = df1.sort_values(
            by="A", ascending=False, na_position="last", kind="mergesort", axis=0
        )
        return df2

    def test_impl4(df1):
        df2 = df1.sort_values(
            by="A", ascending=False, na_position="first", kind="mergesort", axis=0
        )
        return df2

    def get_random_column(n, n_row):
        str_vals = []
        for _ in range(n):
            prob = random.randint(1, 10)
            if prob == 1:
                val = np.nan
            else:
                val = random.randint(1, len_siz)
            str_vals.append(val)
        return str_vals

    def get_random_dataframe_two_columns(n, len_siz):
        df = pd.DataFrame(
            {"A": get_random_column(n, len_siz), "B": get_random_column(n, len_siz)}
        )
        return df

    random.seed(5)
    df1 = get_random_dataframe_two_columns(n, len_siz)
    check_func(
        test_impl1,
        (df1,),
    )
    check_func(
        test_impl2,
        (df1,),
    )
    check_func(
        test_impl3,
        (df1,),
    )
    check_func(
        test_impl4,
        (df1,),
    )


def test_sort_values_na_position_list(memory_leak_check):
    """Test with two columns with some NaN entries, sorting over both using different
    nulls first/last values"""

    def test_impl1(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=True, na_position=["last", "first"], axis=0
        )
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=True, na_position=["first", "last"], axis=0
        )
        return df2

    def test_impl3(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=False, na_position=["last", "first"], axis=0
        )
        return df2

    def test_impl4(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=False, na_position=["first", "last"], axis=0
        )
        return df2

    n = 100
    len_siz = 4

    def get_random_column(n, n_row):
        str_vals = []
        for _ in range(n):
            prob = random.randint(1, 5)
            if prob == 1:
                val = np.nan
            else:
                val = random.randint(1, len_siz)
            str_vals.append(val)
        return str_vals

    def get_random_dataframe_two_columns(n, len_siz):
        df = pd.DataFrame(
            {"A": get_random_column(n, len_siz), "B": get_random_column(n, len_siz)}
        )
        return df

    random.seed(5)
    df1 = get_random_dataframe_two_columns(n, len_siz)
    # Pandas can't support multiple na_position values, so we use py_output
    # Here we always sort by column A and for column B we replace with NA with
    # a value that matches where FIRST LAST would place null values
    def create_py_output(df, ascending, na_position_list):
        df_copy = df.copy(deep=True)
        if ascending:
            if na_position_list[1] == "first":
                na_value = -1
            else:
                na_value = len_siz + 1
        else:
            if na_position_list[1] == "first":
                na_value = len_siz + 1
            else:
                na_value = -1

        df_copy["B"] = df_copy["B"].fillna(na_value)
        output_df = df_copy.sort_values(
            by=["A", "B"], ascending=ascending, na_position=na_position_list[0]
        )
        # Restore NaN
        output_df["B"][output_df["B"] == na_value] = np.nan
        return output_df

    check_func(
        test_impl1, (df1,), py_output=create_py_output(df1, True, ["last", "first"])
    )
    check_func(
        test_impl2, (df1,), py_output=create_py_output(df1, True, ["first", "last"])
    )
    check_func(
        test_impl3, (df1,), py_output=create_py_output(df1, False, ["last", "first"])
    )
    check_func(
        test_impl4, (df1,), py_output=create_py_output(df1, False, ["first", "last"])
    )


@pytest.mark.slow
def test_sort_values_by_index(memory_leak_check):
    """Sorting with a non-trivial index"""

    def test_impl1(df1):
        df2 = df1.sort_values("index_name")
        return df2

    df1 = pd.DataFrame({"A": [1, 2, 2]}, index=[2, 1, 0])
    df1.index.name = "index_name"
    check_func(test_impl1, (df1,), sort_output=False)


@pytest.mark.slow
def test_sort_values_bool_list(memory_leak_check):
    """Test of NaN values for the sorting with vector of ascending"""

    def test_impl1(df1):
        df2 = df1.sort_values(by=["B", "A"], kind="mergesort", axis=0)
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by=["A", "B"], ascending=True, kind="mergesort", axis=0)
        return df2

    def test_impl3(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=[True, True], kind="mergesort", axis=0
        )
        return df2

    def test_impl4(df1):
        df2 = df1.sort_values(by=["A", "B"], ascending=False, kind="mergesort", axis=0)
        return df2

    def test_impl5(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=[False, False], kind="mergesort", axis=0
        )
        return df2

    def test_impl6(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=[True, False], kind="mergesort", axis=0
        )
        return df2

    def test_impl7(df1):
        df2 = df1.sort_values(
            by=["A", "B"], ascending=[False, True], kind="mergesort", axis=0
        )
        return df2

    df1 = pd.DataFrame(
        {
            "A": [2, np.nan, 7, np.nan, -1, -4, np.nan, 1, 2],
            "B": [3, 6, 0, 1, 2, -4, 7, 7, 2],
        }
    )
    check_func(test_impl1, (df1,), sort_output=False)
    check_func(test_impl2, (df1,), sort_output=False)
    check_func(test_impl3, (df1,), sort_output=False)
    check_func(test_impl4, (df1,), sort_output=False)
    check_func(test_impl5, (df1,), sort_output=False)
    check_func(test_impl6, (df1,), sort_output=False)
    check_func(test_impl7, (df1,), sort_output=False)


@pytest.mark.slow
def test_sort_values_nullable_int_array(memory_leak_check):
    """Test of NaN values for the sorting for a nullable int bool array"""

    def test_impl(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort", axis=0
        )
        return df2

    nullarr = pd.array([13, None, 17], dtype="UInt16")
    df1 = pd.DataFrame({"A": nullarr})
    check_func(test_impl, (df1,))


@pytest.mark.slow
def test_sort_with_nan_entries(memory_leak_check):
    """Test of the dataframe with nan entries"""

    def impl1(df):
        return df.sort_values(by="A", kind="mergesort")

    df1 = pd.DataFrame({"A": ["AA", np.nan, "", "D", "GG"]})
    df2 = pd.DataFrame({"A": [1, 8, 4, np.nan, 3]})
    df3 = pd.DataFrame({"A": pd.array([1, 2, None, 3], dtype="UInt16")})
    df4 = pd.DataFrame({"A": pd.Series([1, 8, 4, np.nan, 3], dtype="Int32")})
    df5 = pd.DataFrame({"A": pd.Series(["AA", np.nan, "", "D", "GG"])})
    check_func(impl1, (df1,), sort_output=False, check_typing_issues=False)
    check_func(impl1, (df2,), sort_output=False)
    check_func(impl1, (df3,), sort_output=False)
    check_func(impl1, (df4,), sort_output=False)
    check_func(impl1, (df5,), sort_output=False, check_typing_issues=False)


def test_sort_values_list_inference(memory_leak_check):
    """
    Test constant list inference in sort_values()
    """

    def impl(df):
        return df.sort_values(
            by=list(set(df.columns) - set(["B", "C"])), kind="mergesort"
        )

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2],
            "C": np.arange(6),
        }
    )
    check_func(impl, (df,))


def test_sort_values_key_rm_dead(memory_leak_check):
    """
    Make sure dead column elimination works for sort key outputs
    """

    def impl(df):
        return df.sort_values(by=["A", "C", "E"])[["A", "D"]]

    def impl2(df):
        return df.sort_values(by=["C", "A", "E"])[["A", "D"]]

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2],
            "C": np.arange(6),
            "D": [
                "¿abc¡Y tú, quién te crees?",
                "ÕÕÕú¡úú,úũ¿ééé",
                "россия очень, холодная страна",
                np.nan,
                "مرحبا, العالم ، هذا هو بودو",
                "Γειά σου ,Κόσμε",
            ],
            "E": np.arange(6, dtype=np.int32) * np.int32(10),
        }
    )
    check_func(impl, (df,))
    check_func(impl2, (df,))

    # make sure dead keys are detected properly
    sort_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    sort_func(df)
    fir = sort_func.overloads[sort_func.signatures[0]].metadata["preserved_ir"]

    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.sort.Sort):
                # dead column is inside the live table in case of table format
                assert stmt.dead_var_inds == {1}
                assert stmt.dead_key_var_inds == {2, 4}


def test_sort_values_rm_dead(memory_leak_check):
    """
    Make sure dead Sort IR nodes are removed
    """

    def impl(df):
        df.sort_values(by=["A"])

    df = pd.DataFrame({"A": [1, 3, 2, 0, -1, 4], "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2]})

    # make sure there is no Sort node
    sort_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    sort_func(df)
    fir = sort_func.overloads[sort_func.signatures[0]].metadata["preserved_ir"]

    for block in fir.blocks.values():
        for stmt in block.body:
            assert not isinstance(stmt, bodo.ir.sort.Sort)


def test_sort_values_empty_df_key_rm_dead(memory_leak_check):
    """
    Test if sorting an empty DataFrame where the key is dead.
    Tests to make sure that we can index and access remaining
    columns after the sort operation.
    """

    def impl(df):
        df = df.sort_values(
            by="A",
            ascending=True,
            na_position="first",
        )

        return df["B"]

    df = pd.DataFrame(
        {
            "A": pd.Series([], dtype="datetime64[ns]"),
            "B": pd.Series([], dtype="string[pyarrow]"),
            "C": pd.Series([], dtype="Int64"),
        }
    )

    check_func(impl, (df,))

    # make sure dead keys are detected properly
    sort_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    sort_func(df)
    fir = sort_func.overloads[sort_func.signatures[0]].metadata["preserved_ir"]

    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.sort.Sort):
                # dead column is inside the live table in case of table format
                assert stmt.dead_var_inds == {2}
                assert stmt.dead_key_var_inds == {0}


def test_sort_values_len_only(memory_leak_check):
    """
    Make sure len() works when all columns are dead
    """

    def impl(df):
        df2 = df.sort_values(by=["A"])
        return len(df2)

    df = pd.DataFrame({"A": [1, 3, 2, 0, -1, 4], "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2]})
    check_func(impl, (df,))


def test_sort_values_index_only(memory_leak_check):
    """
    Make sure sort works if returning only the Index (table is dead in table format
    case)
    """

    def impl(df):
        df2 = df.sort_values(by=["A"])
        return df2.index

    df = pd.DataFrame({"A": [1, 3, 2, 0, -1, 4], "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2]})
    check_func(impl, (df,))


def test_sort_values_unknown_cats(memory_leak_check):
    """
    Make sure categorical arrays with unknown categories work
    """

    def impl(df):
        df["A"] = df.A.astype("category")
        df["B"] = df.B.astype("category")
        df.index = pd.Categorical(df.index)
        df2 = df.sort_values(by=["A"])
        return df2

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": ["a1", "a3", "b1", "b4", "a1", "b1"],
            "C": [1.1, 2.2, 3.3, 4.1, -1.1, -0.1],
        },
        index=["a1", "a2", "a3", "a4", "a5", "a6"],
    )
    check_func(impl, (df,))


@pytest.mark.slow
def test_sort_values_force_literal(memory_leak_check):
    """
    Test forcing JIT args to be literal if required by sort_values()
    """

    def impl(df, by, na_position):
        return df.sort_values(by=by, kind="mergesort", na_position=na_position)

    def impl2(df, by, asc, na_position):
        return df.sort_values(
            by=by, kind="mergesort", ascending=asc, na_position=na_position
        )

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": [1.2, 3.4, np.nan, 2.2, 3.1, -1.2],
            "C": np.arange(6),
        }
    )
    check_func(impl, (df, ["B"], "first"))
    check_func(impl, (df, "B", "first"))
    check_func(impl2, (df, ["B", "C"], [False, True], "first"))


@pytest.mark.slow
def test_sort_values_input_boundaries(memory_leak_check):
    """
    Test sort_values() with redistribution boundaries passed in manually
    """

    @bodo.jit(distributed=["df"])
    def impl(df, A):
        bounds = bodo.libs.distributed_api.get_chunk_bounds(A)
        return df.sort_values(by="A", _bodo_chunk_bounds=bounds)

    # create data chunks on different processes and check expected output
    rank = bodo.get_rank()
    n_pes = bodo.get_size()
    if n_pes > 3:
        return
    if n_pes == 1:
        df = pd.DataFrame({"A": np.array([2, 11, 3, 2], np.int64), "B": [4, 3, 2, 1]})
        A = np.array([2, 3, 11])
        out = df.sort_values("A").reset_index(drop=True)
    elif n_pes == 2:
        if rank == 0:
            df = pd.DataFrame({"A": np.array([4, 11, 8], np.int64), "B": [3, 1, 2]})
            A = np.array([3, 6])
            out = pd.DataFrame({"A": np.array([3, 4], np.int64), "B": [4, 3]})
        else:
            df = pd.DataFrame({"A": np.array([10, 3], np.int64), "B": [5, 4]})
            A = np.array([7, 9, 11], np.int64)
            out = pd.DataFrame({"A": np.array([8, 10, 11], np.int64), "B": [2, 5, 1]})
    elif n_pes == 3:
        if rank == 0:
            df = pd.DataFrame({"A": np.array([8, 4], np.int64), "B": [1, 2]})
            A = np.array([2, 3, 5])
            out = pd.DataFrame({"A": np.array([0, 4], np.int64), "B": [4, 2]})
        elif rank == 1:
            df = pd.DataFrame({"A": np.array([11, 0], np.int64), "B": [3, 4]})
            A = np.array([6, 7, 8])
            out = pd.DataFrame({"A": np.array([6, 8], np.int64), "B": [6, 1]})
        # rank 2
        else:
            df = pd.DataFrame({"A": np.array([9, 6], np.int64), "B": [5, 6]})
            A = np.array([9, 11])
            out = pd.DataFrame({"A": np.array([9, 11], np.int64), "B": [5, 3]})

    pd.testing.assert_frame_equal(impl(df, A).reset_index(drop=True), out)

    # test empty chunk corner cases
    if n_pes == 1:
        df = pd.DataFrame({"A": np.array([], np.int64), "B": np.array([], np.float64)})
        A = np.array([2, 3, 11])
        out = df.copy()
    elif n_pes == 2:
        if rank == 0:
            df = pd.DataFrame(
                {"A": np.array([3], np.int64), "B": np.array([1], np.float64)}
            )
            A = np.array([1])
            out = pd.DataFrame(
                {"A": np.array([], np.int64), "B": np.array([], np.float64)}
            )
        else:
            df = pd.DataFrame(
                {"A": np.array([], np.int64), "B": np.array([], np.float64)}
            )
            A = np.array([3])
            out = pd.DataFrame(
                {"A": np.array([3], np.int64), "B": np.array([1], np.float64)}
            )
    elif n_pes == 3:
        if rank == 0:
            df = pd.DataFrame(
                {"A": np.array([], np.int64), "B": np.array([], np.float64)}
            )
            A = np.array([3])
            out = pd.DataFrame(
                {"A": np.array([3], np.int64), "B": np.array([1], np.float64)}
            )
        elif rank == 1:
            df = pd.DataFrame(
                {"A": np.array([3, 5], np.int64), "B": np.array([1, 2], np.float64)}
            )
            A = np.array([4])
            out = pd.DataFrame(
                {"A": np.array([], np.int64), "B": np.array([], np.float64)}
            )
        else:
            df = pd.DataFrame(
                {"A": np.array([], np.int64), "B": np.array([], np.float64)}
            )
            A = np.array([5])
            out = pd.DataFrame(
                {"A": np.array([5], np.int64), "B": np.array([2], np.float64)}
            )

    pd.testing.assert_frame_equal(impl(df, A).reset_index(drop=True), out)

    # error checking unsupported array type
    with pytest.raises(BodoError, match=(r"only supported when there is a single key")):

        @bodo.jit(distributed=["df"])
        def impl(df, A):
            bounds = bodo.libs.distributed_api.get_chunk_bounds(A)
            return df.sort_values(by=["A", "B"], _bodo_chunk_bounds=bounds)

        df = pd.DataFrame(
            {"A": np.array([1], np.int64), "B": np.array([1.2], np.float64)}
        )
        A = np.array([5])
        impl(df, A)


@pytest.mark.slow
def test_list_string(memory_leak_check):
    """Sorting values by list of strings"""

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    random.seed(5)
    n = 100
    df1 = pd.DataFrame({"A": gen_random_list_string_array(2, n)})
    check_func(test_impl, (df1,))


@pytest.mark.slow
def test_list_string_missing(memory_leak_check):
    """Sorting values by list of strings"""

    def f(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    random.seed(5)
    n = 10
    df1 = pd.DataFrame({"A": gen_random_list_string_array(3, n)})
    check_func(f, (df1,), convert_columns_to_pandas=True)


@pytest.mark.slow
# TODO: add memory_leak_check
def test_list_string_arrow():
    """Sorting values by list of strings"""

    def f(df1):
        df2 = df1.sort_values(by=3, kind="mergesort")
        return df2

    def rand_col_l_str(n):
        e_list = []
        for _ in range(n):
            if random.random() < 0.1:
                e_ent = np.nan
            else:
                e_ent = []
                for _ in range(random.randint(1, 5)):
                    k = random.randint(1, 5)
                    val = "".join(random.choices(["A", "B", "C"], k=k))
                    e_ent.append(val)
            e_list.append(e_ent)
        return e_list

    random.seed(5)
    n = 1000
    list_rand = [random.randint(1, 30) for _ in range(n)]
    df1 = pd.DataFrame({3: list_rand, 6: rand_col_l_str(n)})

    check_func(f, (df1,))


@pytest.mark.slow
def test_sort_values_bytes_null(memory_leak_check):
    """
    Test sort_values(): for bytes keys with NULL char inside them to make sure value
    comparison can handle NULLs.
    """

    def impl(df):
        return df.sort_values(by="A")

    df = pd.DataFrame(
        {
            "A": [
                b"\x00abc",
                b"\x00\x00fds",
                b"\x00lkhs",
                b"asbc",
                b"qwer",
                b"zxcv",
                b"\x00pqw",
                b"\x00\x00asdfg",
                b"hiofgas",
            ],
            "B": np.arange(9),
        }
    )
    check_func(impl, (df,))


# ------------------------------ error checking ------------------------------ #


df = pd.DataFrame({"A": [-1, 3, -3, 0, -1], "B": ["a", "c", "b", "c", "b"]})


def test_sort_values_by_const_str_or_str_list(memory_leak_check):
    """
    Test sort_values(): 'by' is of type str or list of str
    """

    def impl1(df):
        return df.sort_values(by=None)

    def impl2(df):
        return df.sort_values(by=1)

    with pytest.raises(
        BodoError,
        match="'by' parameter only supports a constant column label or column labels",
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError,
        match=" invalid keys .* for by",
    ):
        bodo.jit(impl2)(df)


def test_sort_values_by_labels(memory_leak_check):
    """
    Test sort_values(): 'by' is a valid label or label lists
    """

    def impl1(df):
        return df.sort_values(by=["C"])

    def impl2(df):
        return df.sort_values(by=["B", "C"])

    msg = re.escape("sort_values(): invalid keys ['C'] for by")
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl2)(df)


def test_sort_values_axis_default(memory_leak_check):
    """
    Test sort_values(): 'axis' cannot be values other than integer value 0
    """

    def impl1(df):
        return df.sort_values(by=["A"], axis=1)

    def impl2(df):
        return df.sort_values(by=["A"], axis="1")

    def impl3(df):
        return df.sort_values(by=["A"], axis=None)

    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl3)(df)


def test_sort_values_ascending_bool(memory_leak_check):
    """
    Test sort_values(): 'ascending' must be of type bool
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], ascending=None)

    def impl2(df):
        return df.sort_values(by=["A"], ascending=2)

    def impl3(df, ascending):
        return df.sort_values(by=["A", "B"], ascending=ascending)

    def impl4(df):
        return df.sort_values(by=["A", "B"], ascending=[True])

    def impl5(df):
        return df.sort_values(by=["A", "B"], ascending=[True, False, True])

    with pytest.raises(
        BodoError, match="'ascending' parameter must be of type bool or list of bool"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'ascending' parameter must be of type bool or list of bool"
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl3)(df, True)
    with pytest.raises(
        BodoError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl4)(df)
    with pytest.raises(
        BodoError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl5)(df)


def test_sort_force_reshuffling(memory_leak_check):
    """By having only one key we guarantee that all rows will be put into just one bin.
    This gets us a very skewed partition and therefore triggers the reshuffling after sort"""

    def f(df):
        return df.sort_values(by=["A"], kind="mergesort")

    random.seed(5)
    n = 100
    list_A = [1] * n
    list_B = [random.randint(0, 10) for _ in range(n)]
    df = pd.DataFrame({"A": list_A, "B": list_B})
    check_func(f, (df,))


def test_sort_values_inplace_bool(memory_leak_check):
    """
    Test sort_values(): 'inplace' must be of type bool
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], inplace=None)

    def impl2(df):
        return df.sort_values(by="A", inplace=9)

    with pytest.raises(BodoError, match="'inplace' parameter must be of type bool"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="'inplace' parameter must be of type bool"):
        bodo.jit(impl2)(df)


def test_sort_values_kind_no_spec(memory_leak_check):
    """
    Test sort_values(): 'kind' should not be specified by users
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], kind=None)

    def impl2(df):
        return df.sort_values(by=["A"], kind="mergesort")

    def impl3(df):
        return df.sort_values(by=["A"], kind=2)

    with pytest.warns(
        BodoWarning, match="specifying sorting algorithm is not supported"
    ):
        bodo.jit(impl1)(df)
    with pytest.warns(
        BodoWarning, match="specifying sorting algorithm is not supported"
    ):
        bodo.jit(impl2)(df)
    with pytest.warns(
        BodoWarning, match="specifying sorting algorithm is not supported"
    ):
        bodo.jit(impl3)(df)


def test_sort_values_na_position_no_spec(memory_leak_check):
    """
    Test sort_values(): 'na_position' should not be specified by users
    """

    def impl1(df):
        return df.sort_values(by=["A", "B"], na_position=None)

    def impl2(df):
        return df.sort_values(by=["A"], na_position="break")

    def impl3(df):
        return df.sort_values(by=["A"], na_position=0)

    with pytest.raises(
        BodoError, match="na_position parameter must be a literal constant"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="na_position should either be"):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError, match="na_position parameter must be a literal constant"
    ):
        bodo.jit(impl3)(df)


def test_inplace_sort_values_series(memory_leak_check):
    """
    Test sort_values(inplace=True): inplace not supported for Series.sort_values
    """

    def impl1(S):
        return S.sort_values(inplace=True)

    s = pd.Series([1, 8, 4, 10, 3])

    with pytest.raises(
        BodoError, match="inplace parameter only supports default value False"
    ):
        bodo.jit(impl1)(s)


def test_random_decimal(memory_leak_check):
    """Sorting a random decimal"""

    def f(df):
        return df.sort_values(by=["A"])

    random.seed(5)
    n = 50
    df1 = pd.DataFrame({"A": gen_random_decimal_array(2, n)})
    check_func(f, (df1,), convert_columns_to_pandas=True)


def test_sort_list_list(memory_leak_check):
    data = np.array(
        [
            [[[1, 2], [3]], [[2, None]]],
            [[[1, 2], [3]], [[2, 4]]],
            [[[3], [], [1, None, 4]]],
            [[[3], [42], [1, None, 4]]],
            None,
            [[[4, 5, 6], []], [[1]], [[1, 2]]],
            [[[4, 5, 6], [32]], [[1]], [[1, 2]]],
            [],
            [[[], [1]], None, [[1, 4]], []],
        ]
    )
    df = pd.DataFrame({"A": [8, 7, 6, 5, 1, 4, 2, 3, 0], "B": data})

    def f(df):
        df_ret = df.sort_values(by="A", ascending=True, na_position="first")
        return df_ret

    check_func(f, (df,))


def test_sort_values_nested_arrays_random():
    def f(df):
        df2 = df.sort_values(by="A")
        return df2

    random.seed(5)
    n = 1000
    df1 = pd.DataFrame({"A": gen_random_arrow_array_struct_int(10, n)})
    df2 = pd.DataFrame({"A": gen_random_arrow_array_struct_list_int(10, n)})
    df3 = pd.DataFrame({"A": gen_random_arrow_list_list_int(1, -0.1, n)})
    df4 = pd.DataFrame({"A": gen_random_arrow_struct_struct(10, n)})
    df5 = pd.DataFrame({"A": gen_random_arrow_list_list_decimal(2, -0.1, n)})
    check_parallel_coherency(f, (df1,))
    check_parallel_coherency(f, (df2,))
    check_parallel_coherency(f, (df3,))
    check_parallel_coherency(f, (df4,))
    check_parallel_coherency(f, (df5,))
