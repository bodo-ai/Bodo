"""Test sort_values operation as called as df.sort_values()
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
import traceback
from datetime import date, datetime
from decimal import Decimal

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo import BodoWarning
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    check_parallel_coherency,
    gen_nonascii_list,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_array_struct_list_int,
    gen_random_arrow_array_struct_string,
    gen_random_arrow_list_list_decimal,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_string,
    gen_random_arrow_struct_struct,
    gen_random_decimal_array,
    gen_random_list_string_array,
    is_bool_object_series,
    pytest_pandas,
)

pytestmark = pytest_pandas


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
                    "C": [True, False, False, None, True],
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
            pd.date_range(start="2018-04-24", end="2018-04-29", periods=5, unit="ns"),
        ),
        # bool list, numpy array
        # TODO: change to "A": [True, False, False, None, True])
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
                "A": pd.date_range(
                    start="2018-04-24", end="2018-04-29", periods=5, unit="ns"
                ),
                "B": pd.date_range(
                    start="2013-09-04", end="2013-09-29", periods=5, unit="ns"
                ),
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
                                    start="2/1/2015",
                                    end="2/24/2021",
                                    periods=4,
                                    unit="ns",
                                )
                            ),
                            pd.Series(data=[None], index=[4]),
                        ]
                    ).astype("datetime64[ns]"),
                    ordered=True,
                ),
                "D": pd.Categorical(
                    pd.concat(
                        [
                            pd.Series(
                                pd.timedelta_range(start="1 day", periods=4, unit="ns")
                            ),
                            pd.Series(data=[None], index=[4]),
                        ]
                    ).astype("timedelta64[ns]"),
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
                        None,
                        b"B",
                        b"ZZ",
                        None,
                        b"F123",
                    ],
                    "B": [
                        b"jkasdf",
                        b"asdfas",
                        None,
                        b"D",
                        None,
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
        ),
        # TODO: timedelta
    ]
)
def df_value(request):
    return request.param


@pytest.mark.df_lib
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
    list_date = pd.date_range(start="2000-01-01", periods=len_period, unit="ns")
    np.random.seed(5)
    e_list = []
    for idx in range(len_period):
        if np.random.random() < 0.2:
            e_ent = pd.NaT
        else:
            e_ent = list_date[idx]
        e_list.append(e_ent)

    df1 = pd.DataFrame({"A": e_list})

    check_func(
        test_impl1,
        (df1,),
        reset_index=bodo.test_dataframe_library_enabled,
    )
    if not is_slow_run:
        return
    check_func(
        test_impl2,
        (df1,),
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        test_impl3,
        (df1,),
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        test_impl4,
        (df1,),
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.smoke
def test_single_col(memory_leak_check):
    """
    sorts a dataframe that has only one column
    """
    fname = os.path.join("bodo", "tests", "data", "kde.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
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

    check_func(impl, (df_value,), reset_index=bodo.test_dataframe_library_enabled)


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
def test_sort_values_1col_inplace(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column
    """
    import bodo.decorators  # noqa

    if bodo.hiframes.boxing._use_dict_str_type:
        pytest.skip("not supported for dict string type")

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
def test_sort_values_2col_inplace(df_value, memory_leak_check):
    """
    Test sort_values(): with just 1 column
    """
    import bodo.decorators  # noqa

    if bodo.hiframes.boxing._use_dict_str_type:
        pytest.skip("not supported for dict string type")

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


@pytest.mark.df_lib
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
                str_vals.append(None)
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
    check_func(test_impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
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
                bytes_vals.append(None)
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
    check_func(test_impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
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
    check_func(
        test_impl1,
        (get_quasi_random(n),),
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        test_impl2,
        (get_quasi_random(n),),
        reset_index=bodo.test_dataframe_library_enabled,
    )


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
    check_func(
        test_impl1,
        (get_quasi_random(n),),
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        test_impl2,
        (get_quasi_random(n),),
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.df_lib
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
        reset_index=bodo.test_dataframe_library_enabled,
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
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.df_lib
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
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.df_lib
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
    check_func(test_impl, (df1,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
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
                val = None
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
    check_func(test_impl, (df1,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
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
                val = None
            else:
                val = bytes(np.random.randint(1, 100))

            bin_vals.append(val)
        df = pd.DataFrame({"A": bin_vals})
        return df

    np.random.seed(5)
    df1 = get_random_bin_df(100)
    check_func(test_impl, (df1,), reset_index=bodo.test_dataframe_library_enabled)


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
        output_df.loc[output_df["B"] == na_value, "B"] = np.nan
        return output_df

    check_func(
        test_impl1,
        (df1,),
        py_output=create_py_output(df1, True, ["last", "first"]),
        check_dtype=False,
    )
    check_func(
        test_impl2,
        (df1,),
        py_output=create_py_output(df1, True, ["first", "last"]),
        check_dtype=False,
    )
    check_func(
        test_impl3,
        (df1,),
        py_output=create_py_output(df1, False, ["last", "first"]),
        check_dtype=False,
    )
    check_func(
        test_impl4,
        (df1,),
        py_output=create_py_output(df1, False, ["first", "last"]),
        check_dtype=False,
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


@pytest.mark.df_lib
@pytest.mark.slow
def test_sort_with_nan_entries(memory_leak_check):
    """Test of the dataframe with nan entries"""

    def impl1(df):
        return df.sort_values(by="A", kind="mergesort")

    df1 = pd.DataFrame({"A": ["AA", None, "", "D", "GG"]})
    df2 = pd.DataFrame({"A": [1, 8, 4, np.nan, 3]})
    df3 = pd.DataFrame({"A": pd.array([1, 2, None, 3], dtype="UInt16")})
    df4 = pd.DataFrame({"A": pd.Series([1, 8, 4, np.nan, 3], dtype="Int32")})
    df5 = pd.DataFrame({"A": pd.Series(["AA", None, "", "D", "GG"])})
    check_func(
        impl1,
        (df1,),
        sort_output=False,
        check_typing_issues=False,
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        impl1,
        (df2,),
        sort_output=False,
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        impl1,
        (df3,),
        sort_output=False,
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        impl1,
        (df4,),
        sort_output=False,
        reset_index=bodo.test_dataframe_library_enabled,
    )
    check_func(
        impl1,
        (df5,),
        sort_output=False,
        check_typing_issues=False,
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.df_lib
def test_sort_values_list_inference(memory_leak_check):
    """
    Test constant list inference in sort_values()
    """

    def impl(df):
        return df.sort_values(by=list(set(df.columns) - {"B", "C"}), kind="mergesort")

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2],
            "C": np.arange(6),
        }
    )
    check_func(impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


def test_sort_values_key_rm_dead(memory_leak_check):
    """
    Make sure dead column elimination works for sort key outputs
    """
    from bodo.tests.utils_jit import DeadcodeTestPipeline

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
                None,
                "مرحبا, العالم ، هذا هو بودو",
                "Γειά σου ,Κόσμε",
            ],
            "E": np.arange(6, dtype=np.int32) * np.int32(10),
        }
    )
    check_func(impl, (df,), reset_index=bodo.test_dataframe_library_enabled)
    check_func(impl2, (df,), reset_index=bodo.test_dataframe_library_enabled)

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
    from bodo.tests.utils_jit import DeadcodeTestPipeline

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
    from bodo.tests.utils_jit import DeadcodeTestPipeline

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


@pytest.mark.df_lib
def test_sort_values_len_only(memory_leak_check):
    """
    Make sure len() works when all columns are dead
    """

    def impl(df):
        df2 = df.sort_values(by=["A"])
        return len(df2)

    df = pd.DataFrame({"A": [1, 3, 2, 0, -1, 4], "B": [1.2, 3.4, 0.1, 2.2, 3.1, -1.2]})
    check_func(impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


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
    check_func(impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
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
    check_func(
        impl, (df, ["B"], "first"), reset_index=bodo.test_dataframe_library_enabled
    )
    check_func(
        impl, (df, "B", "first"), reset_index=bodo.test_dataframe_library_enabled
    )
    check_func(
        impl2,
        (df, ["B", "C"], [False, True], "first"),
        reset_index=bodo.test_dataframe_library_enabled,
    )


@pytest.mark.slow
def test_sort_values_input_boundaries(memory_leak_check):
    """
    Test sort_values() with redistribution boundaries passed in manually
    """
    from bodo.utils.typing import BodoError

    @bodo.jit(distributed=["df", "A"])
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

        @bodo.jit(distributed=["df", "A"])
        def impl(df, A):
            bounds = bodo.libs.distributed_api.get_chunk_bounds(A)
            return df.sort_values(by=["A", "B"], _bodo_chunk_bounds=bounds)

        df = pd.DataFrame(
            {"A": np.array([1], np.int64), "B": np.array([1.2], np.float64)}
        )
        A = np.array([5])
        impl(df, A)


@pytest.fixture(
    params=[
        ## Edge Cases
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": [0, 6, 7, 7] * 10,
                            "B": [10, 7, 7, 8] * 10,
                        }
                    )
                ],
                [
                    np.array([0, 5, 10], dtype=np.int64),
                ],
                float("-inf"),
                float("inf"),
            ),
            id="edge_case",
        ),
        ## Very simple test
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.arange(0, 100, 5, dtype=np.int64),
                            "B": np.arange(0, 200, 10, dtype=np.int64),
                        }
                    ),
                ],
                [
                    np.array([0, 100, 200], dtype=np.int64),
                ],
                float("-inf"),
                float("inf"),
            ),
            id="simple",
        ),
        ## Test that it works in the empty table case
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.arange(0, dtype=np.int64),
                            "B": np.arange(0, dtype=np.int64),
                        }
                    ),
                ],
                [
                    np.array([0, 100, 200], dtype=np.int64),
                ],
                float("-inf"),
                float("inf"),
            ),
            id="empty",
        ),
        ## Integers: Data x bounds ([10, NA], [NA, NA], [10, 20], [20, 20])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.arange(0, 100, 5, dtype=np.int64),
                            "B": np.arange(0, 200, 10, dtype=np.int64),
                        }
                    ),
                ],
                [
                    np.array([-10, 20], dtype=np.int64),
                    np.array([10, 20], dtype=np.int64),
                    np.array([20, 20], dtype=np.int64),
                    np.array([20, 2000], dtype=np.int64),
                ],
                float("-inf"),
                float("inf"),
            ),
            id="int64",
        ),
        ## Nullable Integers: Data (with nulls, without nulls)  x bounds ([10, NA], [NA, NA], [10, 20], [20, 20])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": pd.array(
                                [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 4, 8, 12, 16],
                                dtype="Int64",
                            ),
                            "B": pd.array(
                                [pd.NA, pd.NA, 20, 24, 28, 32, 36, pd.NA, pd.NA, pd.NA],
                                dtype="Int64",
                            ),
                        }
                    ),
                ],
                [
                    pd.array([10, 20], dtype="Int64"),
                    pd.array([10, pd.NA], dtype="Int64"),
                    pd.array([pd.NA, pd.NA], dtype="Int64"),
                    pd.array([20, 20], dtype="Int64"),
                ],
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
            ),
            id="Int64",
        ),
        ## Floats: Data (with NAs, without NAs)  x bounds ([10.5, NA], [NA, NA], [10, 20.4], [10, 20], [20, 20])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.arange(0, 20, 4, dtype=np.float64),
                            "B": np.arange(20, 40, 4, dtype=np.float64),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": np.concatenate(
                                (
                                    np.array([np.nan] * 5, dtype=np.float64),
                                    np.arange(0, 20, 4, dtype=np.float64),
                                )
                            ),
                            "B": np.concatenate(
                                (
                                    np.array([np.nan] * 2, dtype=np.float64),
                                    np.arange(20, 40, 4, dtype=np.float64),
                                    np.array([np.nan] * 3, dtype=np.float64),
                                )
                            ),
                        }
                    ),
                ],
                [
                    np.array([10.5, np.nan], dtype=np.float64),
                    np.array([np.nan, np.nan], dtype=np.float64),
                    np.array([10, 20.4], dtype=np.float64),
                    np.array([10, 20], dtype=np.float64),
                    np.array([20, 20], dtype=np.float64),
                ],
                float("-inf"),
                float("inf"),
            ),
            id="float64",
        ),
        ## Date: Data (with nulls, without nulls)  x bounds ([D1, NA], [NA, NA], [D1, D2], [D2, D2])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": pd.date_range(
                                start="2016-01-01",
                                end="2022-12-12",
                                periods=15,
                                unit="ns",
                            ).date,
                            "B": pd.date_range(
                                start="2018-01-01",
                                end="2024-12-12",
                                periods=15,
                                unit="ns",
                            ).date,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": np.concatenate(
                                (
                                    np.array([pd.NA] * 5),
                                    pd.date_range(
                                        start="2016-01-01",
                                        end="2022-12-12",
                                        periods=15,
                                        unit="ns",
                                    ).date,
                                )
                            ),
                            "B": np.concatenate(
                                (
                                    np.array([pd.NA] * 3),
                                    pd.date_range(
                                        start="2016-01-01",
                                        end="2022-12-12",
                                        periods=15,
                                        unit="ns",
                                    ).date,
                                    np.array([pd.NA] * 2),
                                )
                            ),
                        }
                    ),
                ],
                [
                    pd.array(
                        [date(2018, 12, 12), None], dtype=pd.ArrowDtype(pa.date32())
                    ),
                    pd.array(
                        [date(2018, 12, 12), pd.NA], dtype=pd.ArrowDtype(pa.date32())
                    ),
                    pd.array([pd.NA, pd.NA], dtype=pd.ArrowDtype(pa.date32())),
                    pd.array([None, None], dtype=pd.ArrowDtype(pa.date32())),
                    pd.array(
                        [date(2018, 12, 12), date(2019, 12, 12)],
                        dtype=pd.ArrowDtype(pa.date32()),
                    ),
                    pd.array(
                        [date(2019, 12, 12), date(2019, 12, 12)],
                        dtype=pd.ArrowDtype(pa.date32()),
                    ),
                ],
                date.min,
                date.max,
            ),
            id="date",
        ),
        ## Datetime: Data (with NAs, without NAs)  x bounds ([T1, NA], [NA, NA], [T1, T2], [T2, T2])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.array(
                                pd.date_range(
                                    start="2016-01-01",
                                    end="2022-12-12",
                                    periods=15,
                                    unit="ns",
                                ),
                                dtype="datetime64[ns]",
                            ),
                            "B": np.array(
                                pd.date_range(
                                    start="2018-01-01",
                                    end="2024-12-12",
                                    periods=15,
                                    unit="ns",
                                ),
                                dtype="datetime64[ns]",
                            ),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": np.concatenate(
                                (
                                    np.array([None] * 5, dtype="datetime64[ns]"),
                                    np.array(
                                        pd.date_range(
                                            start="2016-01-01",
                                            end="2022-12-12",
                                            periods=15,
                                            unit="ns",
                                        ),
                                        dtype="datetime64[ns]",
                                    ),
                                ),
                                dtype="datetime64[ns]",
                            ),
                            "B": np.concatenate(
                                (
                                    np.array([None] * 3, dtype="datetime64[ns]"),
                                    np.array(
                                        pd.date_range(
                                            start="2016-01-01",
                                            end="2022-12-12",
                                            periods=15,
                                            unit="ns",
                                        ),
                                        dtype="datetime64[ns]",
                                    ),
                                    np.array([None] * 2, dtype="datetime64[ns]"),
                                ),
                                dtype="datetime64[ns]",
                            ),
                        }
                    ),
                ],
                [
                    np.array(
                        [datetime(2018, 12, 12, 15, 23, 45), None],
                        dtype="datetime64[ns]",
                    ),
                    np.array([None, None], dtype="datetime64[ns]"),
                    np.array(
                        [
                            datetime(2018, 12, 12, 15, 23, 45),
                            datetime(2019, 12, 12, 21, 34, 21),
                        ],
                        dtype="datetime64[ns]",
                    ),
                    np.array(
                        [
                            datetime(2019, 12, 12, 21, 34, 21),
                            datetime(2019, 12, 12, 21, 34, 21),
                        ],
                        dtype="datetime64[ns]",
                    ),
                ],
                pd.Timestamp.min,
                pd.Timestamp.max,
            ),
            id="datetime",
        ),
        ## Timedelta: Data (with NAs, without NAs)  x bounds ([T1, NA], [NA, NA], [T1, T2], [T2, T2])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.array(
                                pd.timedelta_range(
                                    start="1 day", end="500 day", periods=15, unit="ns"
                                ),
                                dtype="timedelta64[ns]",
                            ),
                            "B": np.array(
                                pd.timedelta_range(
                                    start="10 days",
                                    end="300 day",
                                    periods=15,
                                    unit="ns",
                                ),
                                dtype="timedelta64[ns]",
                            ),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": np.concatenate(
                                (
                                    np.array([None] * 5, dtype="timedelta64[ns]"),
                                    np.array(
                                        pd.timedelta_range(
                                            start="1 day",
                                            end="500 day",
                                            periods=15,
                                            unit="ns",
                                        ),
                                        dtype="timedelta64[ns]",
                                    ),
                                ),
                                dtype="timedelta64[ns]",
                            ),
                            "B": np.concatenate(
                                (
                                    np.array([None] * 2, dtype="timedelta64[ns]"),
                                    np.array(
                                        pd.timedelta_range(
                                            start="10 days",
                                            end="300 day",
                                            periods=15,
                                            unit="ns",
                                        ),
                                        dtype="timedelta64[ns]",
                                    ),
                                    np.array([None] * 3, dtype="timedelta64[ns]"),
                                ),
                                dtype="timedelta64[ns]",
                            ),
                        }
                    ),
                ],
                [
                    np.array([pd.Timedelta("5 day"), None], dtype="timedelta64[ns]"),
                    np.array([None, None], dtype="timedelta64[ns]"),
                    np.array(
                        [pd.Timedelta("5 day"), pd.Timedelta("600 day")],
                        dtype="timedelta64[ns]",
                    ),
                    np.array(
                        [pd.Timedelta("400 day"), pd.Timedelta("400 day")],
                        dtype="timedelta64[ns]",
                    ),
                ],
                pd.Timedelta.min,
                pd.Timedelta.max,
            ),
            id="timedelta",
        ),
        ## Time: Data (with NAs, without NAs)  x bounds ([T1, NA], [NA, NA], [T1, T2], [T2, T2])
        pytest.param(
            lambda: (
                [
                    pd.DataFrame(
                        {
                            "A": [
                                bodo.types.Time(12, 0),
                                bodo.types.Time(1, 1, 3),
                                bodo.types.Time(2),
                                bodo.types.Time(12, 0),
                                bodo.types.Time(6, 7, 13),
                                bodo.types.Time(2),
                                bodo.types.Time(17, 1, 3),
                            ],
                            "B": [
                                bodo.types.Time(15, 10),
                                bodo.types.Time(1, 2, 3),
                                bodo.types.Time(5),
                                bodo.types.Time(12, 10),
                                bodo.types.Time(6, 7, 13),
                                bodo.types.Time(20),
                                bodo.types.Time(19, 1, 10),
                            ],
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": [
                                None,
                                None,
                                None,
                                None,
                                None,
                                bodo.types.Time(12, 0),
                                bodo.types.Time(1, 1, 3),
                                bodo.types.Time(2),
                                bodo.types.Time(12, 0),
                                bodo.types.Time(6, 7, 13),
                                bodo.types.Time(2),
                                bodo.types.Time(17, 1, 3),
                            ],
                            "B": [
                                None,
                                None,
                                None,
                                bodo.types.Time(15, 10),
                                bodo.types.Time(1, 2, 3),
                                bodo.types.Time(5),
                                bodo.types.Time(12, 10),
                                bodo.types.Time(6, 7, 13),
                                bodo.types.Time(20),
                                bodo.types.Time(19, 1, 10),
                                None,
                                None,
                            ],
                        }
                    ),
                ],
                [
                    pd.array(
                        [bodo.types.Time(4, 10, 45), None],
                        pd.ArrowDtype(pa.time64("ns")),
                    ),
                    pd.array([None, None], pd.ArrowDtype(pa.time64("ns"))),
                    pd.array(
                        [bodo.types.Time(4, 10, 45), bodo.types.Time(18, 14, 59)],
                        pd.ArrowDtype(pa.time64("ns")),
                    ),
                    pd.array(
                        [bodo.types.Time(18, 10, 45), bodo.types.Time(18, 10, 45)],
                        pd.ArrowDtype(pa.time64("ns")),
                    ),
                ],
                bodo.types.Time(0, 0, 0, 0, 0),
                bodo.types.Time(23, 59, 59, 999, 999),
            ),
            id="time",
        ),
        ## Decimals: Data (with NAs, without NAs)  x bounds ([10.5, NA], [NA, NA], [10, 20.5], [10, 20], [20, 20])
        pytest.param(
            (
                [
                    pd.DataFrame(
                        {
                            "A": np.array(
                                [
                                    Decimal(d)
                                    for d in np.arange(0, 20, 4, dtype=np.float64)
                                ]
                            ),
                            "B": np.array(
                                [
                                    Decimal(d)
                                    for d in np.arange(20, 40, 4, dtype=np.float64)
                                ]
                            ),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "A": np.concatenate(
                                (
                                    np.array([None] * 5),
                                    np.array(
                                        [
                                            Decimal(d)
                                            for d in np.arange(
                                                0, 20, 4, dtype=np.float64
                                            )
                                        ]
                                    ),
                                )
                            ),
                            "B": np.concatenate(
                                (
                                    np.array([None] * 2),
                                    np.array(
                                        [
                                            Decimal(d)
                                            for d in np.arange(
                                                20, 40, 4, dtype=np.float64
                                            )
                                        ]
                                    ),
                                    np.array([None] * 3),
                                )
                            ),
                        }
                    ),
                ],
                [
                    pd.array(
                        [Decimal(10.5), None],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
                    ),
                    pd.array([None, None], dtype=pd.ArrowDtype(pa.decimal128(38, 18))),
                    pd.array(
                        [Decimal(10.0), Decimal(20.5)],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
                    ),
                    pd.array(
                        [Decimal(10.0), Decimal(20.0)],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
                    ),
                    pd.array(
                        [Decimal(20.0), Decimal(20.0)],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
                    ),
                ],
                # Using Decimal("-Infinity") and Decimal("Infinity") leads to unrelated issues.
                # For our purposes, just using large values should be sufficient.
                Decimal(-99999999),
                Decimal(99999999),
            ),
            id="decimal",
        ),
    ],
)
def sort_args(request):
    """Lazily evaluate args to avoid importing compiler on test collection."""
    import bodo.decorators  # noqa

    args = request.param

    return args() if callable(args) else args


@pytest.mark.skip("TODO: fix for Pandas 3")
@pytest.mark.skipif(bodo.get_size() > 3, reason="Only implemented for up to 3 ranks.")
def test_sort_table_for_interval_join(sort_args, memory_leak_check):
    """
    Test the sort_table_for_interval_join for multiple data types.
    Tests both the point (n_keys=1) and interval (n_keys=2) cases.
    """
    from bodo.tests.utils_jit import reduce_sum

    dfs, bounds_list, min_val, max_val = sort_args

    @bodo.jit(distributed=["df"])
    def impl_point(df, bounds):
        return df.sort_values(
            by="A",
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    @bodo.jit(distributed=["df"])
    def impl_interval(df, bounds):
        return df.sort_values(
            by=["A", "B"],
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    myrank = bodo.get_rank()
    n_pes = bodo.get_size()

    def check_correctness_point(in_df, out, bounds, myrank, n_pes):
        left_bound = bounds[myrank - 1] if myrank > 0 else min_val
        right_bound = bounds[myrank] if myrank < (n_pes - 1) else max_val
        # We treat NAs as +inf.
        if pd.isna(left_bound):
            left_bound = max_val
        if pd.isna(right_bound):
            right_bound = max_val
        if (left_bound == right_bound) and (myrank != 0):
            # If the bound of this rank is the same as that of the previous rank,
            # all points should have gone to the previous rank and this rank should
            # be empty (except if it's rank 0, since the data has to go somewhere in
            # case all bounds are the same).
            assert out.shape[0] == 0
        else:
            for _, row in out.iterrows():
                val = row["A"]
                if pd.isna(val):
                    val = max_val
                assert left_bound < val <= right_bound

        # Verify that it's sorted:
        sorted_py = out.sort_values(by="A")
        pd.testing.assert_series_equal(
            out["A"].reset_index(drop=True), sorted_py["A"].reset_index(drop=True)
        )

        # Verify that it has all the expected rows:
        out_len_g = reduce_sum(out.shape[0])
        assert out_len_g == in_df.shape[0], (
            f"Expected total length to be the same before and after sorting the points. Expected: {in_df.shape[0]}. Got: {out_len_g}"
        )

    def check_correctness_interval(
        in_df: pd.DataFrame,
        out: pd.DataFrame,
        bounds: np.ndarray,
        myrank: int,
        n_pes: int,
    ):
        left_bound = bounds[myrank - 1] if myrank > 0 else min_val
        right_bound = bounds[myrank] if myrank < (n_pes - 1) else max_val
        # We treat NAs as +inf.
        if pd.isna(left_bound):
            left_bound = max_val
        if pd.isna(right_bound):
            right_bound = max_val

        for i, _ in out.iterrows():
            # Note that this will replace the values in-place in the
            # out dataframe, which is what we want.
            if pd.isna(out.loc[i, "B"]):
                out.loc[i, "B"] = max_val
            if pd.isna(out.loc[i, "A"]):
                out.loc[i, "A"] = max_val
            assert out["A"][i] <= out["B"][i], (
                f"Interval sort should skip bad rows. Found ({out['A'][i]}, {out['B'][i]})."
            )
            assert left_bound <= out["B"][i], f"Expected {left_bound} <= {out['B'][i]}"
            assert out["A"][i] <= right_bound, (
                f"Expected {out['A'][i]} <= {right_bound}"
            )

        # Verify that it's sorted correctly:
        sorted_py = out.sort_values(by=["A", "B"])
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True),
            sorted_py.reset_index(drop=True),
            check_dtype=False,
        )

        # Verify that all the rows that are expected to be sent to this rank,
        # indeed are.
        for i, _ in in_df.iterrows():
            if pd.isna(in_df.loc[i, "B"]):
                in_df.loc[i, "B"] = max_val
            if pd.isna(in_df.loc[i, "A"]):
                in_df.loc[i, "A"] = max_val
        exp_df = in_df[
            (in_df["A"] <= in_df["B"])  # Remove bad intervals
            & (in_df["A"] <= right_bound)
            & (in_df["B"] >= left_bound)
        ]

        exp_df = exp_df.sort_values(by=["A", "B"])
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True), exp_df.reset_index(drop=True), check_dtype=False
        )

    for df in dfs:
        # Set the seed so the shuffle is consistent across ranks
        np.random.seed(1024)
        df = df.sample(frac=1)

        for bounds in bounds_list:
            # Create a copy since we modify it during the tests
            in_df = df.copy(deep=True)
            # Make bounds the correct length
            bounds = bounds[: (n_pes - 1)]

            point_sort_out = impl_point(_get_dist_arg(in_df), bounds)
            passed = 1
            try:
                check_correctness_point(in_df, point_sort_out, bounds, myrank, n_pes)
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                passed = 0
            n_passed = reduce_sum(passed)
            assert n_passed == n_pes

            interval_sort_out = impl_interval(_get_dist_arg(in_df), bounds)
            passed = 1
            try:
                check_correctness_interval(
                    in_df, interval_sort_out, bounds, myrank, n_pes
                )
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                passed = 0
            n_passed = reduce_sum(passed)
            assert n_passed == n_pes


def test_sort_for_interval_join_err_checking():
    """
    Tests that simple compile time checks are enforced when using
    _bodo_interval_sort = True.
    """
    from bodo.utils.typing import BodoError

    @bodo.jit(distributed=["df"])
    def impl(df, by, bounds):
        return df.sort_values(
            by=by,
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    # 1. _bodo_chunk_bounds not provided with _bodo_interval_sort
    df = pd.DataFrame({"A": np.arange(10)})
    bounds = None

    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): _bodo_chunk_bounds with at most 2 keys must be provided when _bodo_interval_sort=True",
    ):
        impl(_get_dist_arg(df), "A", bounds)

    # 2. Number of keys is >2
    df = pd.DataFrame(
        {
            "A": np.arange(10, dtype=np.int64),
            "B": np.arange(10, 20, dtype=np.int64),
            "C": np.arange(20, 30, dtype=np.int64),
        }
    )
    bounds = np.array([0, 10, 20], dtype=np.int64)
    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): When using _bodo_interval_sort, you must specify at most 2 keys",
    ):
        impl(_get_dist_arg(df), ["A", "B", "C"], bounds)

    # 3. ascending is not true (both singular and list case)
    @bodo.jit(distributed=["df"])
    def impl2(df, by, bounds):
        return df.sort_values(
            by=by,
            ascending=False,
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): 'ascending' parameter must be true when using _bodo_interval_sort",
    ):
        impl2(_get_dist_arg(df), ["A", "B"], bounds)

    @bodo.jit(distributed=["df"])
    def impl3(df, by, bounds):
        return df.sort_values(
            by=by,
            ascending=[True, False],
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): Every value in 'ascending' must be true when using _bodo_interval_sort",
    ):
        impl3(_get_dist_arg(df), ["A", "B"], bounds)

    # 4. na_position is not 'last' (both singular and list case)
    @bodo.jit(distributed=["df"])
    def impl4(df, by, bounds):
        return df.sort_values(
            by=by,
            na_position="first",
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): na_position must be 'last' when using _bodo_interval_sort",
    ):
        impl4(_get_dist_arg(df), ["A", "B"], bounds)

    @bodo.jit(distributed=["df"])
    def impl5(df, by, bounds):
        return df.sort_values(
            by=by,
            na_position=["last", "first"],
            _bodo_chunk_bounds=bounds,
            _bodo_interval_sort=True,
        )

    with pytest.raises(
        BodoError,
        match=r"sort_values\(\): Every value in na_position must be 'last' when using _bodo_interval_sort",
    ):
        impl5(_get_dist_arg(df), ["A", "B"], bounds)


@pytest.mark.df_lib
@pytest.mark.slow
def test_list_string(memory_leak_check):
    """Sorting values by list of strings"""

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    random.seed(5)
    n = 100
    df1 = pd.DataFrame({"A": gen_random_list_string_array(2, n)})
    check_func(test_impl, (df1,), reset_index=bodo.test_dataframe_library_enabled)


@pytest.mark.df_lib
@pytest.mark.slow
def test_list_string_missing(memory_leak_check):
    """Sorting values by list of strings"""

    def f(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    random.seed(5)
    n = 10
    df1 = pd.DataFrame({"A": gen_random_list_string_array(3, n)})
    check_func(
        f,
        (df1,),
        convert_columns_to_pandas=True,
        reset_index=bodo.test_dataframe_library_enabled,
    )


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
                e_ent = None
            else:
                e_ent = []
                for _ in range(random.randint(1, 5)):
                    k = random.randint(1, 5)
                    val = "".join(random.choices(["A", "B", "C"], k=k))
                    e_ent.append(val)
            e_list.append(e_ent)
        return pd.Series(
            e_list, dtype=pd.ArrowDtype(pa.large_list(pa.large_string()))
        ).values

    random.seed(5)
    n = 1000
    list_rand = [random.randint(1, 30) for _ in range(n)]
    df1 = pd.DataFrame({3: list_rand, 6: rand_col_l_str(n)})

    check_func(f, (df1,))


@pytest.mark.df_lib
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
    check_func(impl, (df,), reset_index=bodo.test_dataframe_library_enabled)


# ------------------------------ error checking ------------------------------ #


df = pd.DataFrame({"A": [-1, 3, -3, 0, -1], "B": ["a", "c", "b", "c", "b"]})


def test_sort_values_by_const_str_or_str_list(memory_leak_check):
    """
    Test sort_values(): 'by' is of type str or list of str
    """
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

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


@pytest.mark.df_lib
def test_sort_force_reshuffling(memory_leak_check):
    """By having only one key we guarantee that all rows will be put into just one bin.
    This gets us a very skewed partition and therefore triggers the reshuffling after sort
    """

    def f(df):
        return df.sort_values(by=["A"], kind="mergesort")

    random.seed(5)
    n = 100
    list_A = [1] * n
    list_B = [random.randint(0, 10) for _ in range(n)]
    df = pd.DataFrame({"A": list_A, "B": list_B})
    check_func(f, (df,), reset_index=bodo.test_dataframe_library_enabled)


def test_sort_values_inplace_bool(memory_leak_check):
    """
    Test sort_values(): 'inplace' must be of type bool
    """
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

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
    check_func(
        f,
        (df1,),
        convert_columns_to_pandas=True,
        convert_to_nullable_float=False,
    )


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
        ],
        object,
    )
    df = pd.DataFrame({"A": [8, 7, 6, 5, 1, 4, 2, 3, 0], "B": data})

    def f(df):
        df_ret = df.sort_values(by="A", ascending=True, na_position="first")
        return df_ret

    check_func(f, (df,))


@pytest.mark.skip(reason="Nested Arrays are experimental.")
def test_sort_values_nested_arrays_random(memory_leak_check):
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


@pytest.fixture()
def nested_df(request):
    rng = np.random.default_rng(42)
    df = request.param
    df["A"] = rng.permutation(df["A"])
    return df


@pytest.mark.parametrize(
    "nested_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_array_struct_int(10, 1000),
                }
            ),
            id="struct_int",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_array_struct_list_int(10, 1000),
                }
            ),
            id="struct_list_int",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_list_list_int(1, -0.1, 1000),
                }
            ),
            id="list_list_int",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_struct_struct(10, 1000),
                }
            ),
            id="struct_struct",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_list_list_decimal(2, -0.1, 1000),
                }
            ),
            id="list_list_decimal",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_array_struct_int(10, 1000, True),
                }
            ),
            id="map_int",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_array_struct_list_int(10, 1000, True),
                }
            ),
            id="map_list_int",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_struct_struct(10, 1000, True),
                }
            ),
            id="map_map",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_list_string_array(1, 1000),
                }
            ),
            id="array_string",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_struct_string(10, 1000),
                }
            ),
            id="struct_string",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(1000),
                    "B": gen_random_arrow_array_struct_string(10, 3, 1000),
                }
            ),
            id="array_struct_string",
        ),
    ],
)
def test_sort_nested_arrays_passthrough_random(nested_df, memory_leak_check):
    """Test sort for tables with nested array data"""

    def f(df):
        df2 = df.sort_values(by="A")
        return df2

    check_func(f, (nested_df,))


def test_sort_values_nested_arr_dict(memory_leak_check):
    """Make sure sort works for array(array) input with dictionary data (see [BSE-1155])"""

    def impl(df):
        return df.sort_values(by="A")

    df1 = pd.DataFrame(
        {
            "A": [2, 1, 3],
            "B": pd.array(
                [["a1", None, "a2"], ["a3"], None],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
            ),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [2, 1, 3],
            "B": pd.array(
                [[["1", "2", "8"], ["3"]], [["2", None]], None],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.large_string()))),
            ),
        }
    )
    # TODO[BSE-1257]: support parallel sort
    check_func(impl, (df1,), only_seq=True)
    check_func(impl, (df2,), only_seq=True)
