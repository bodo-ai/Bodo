# Copyright (C) 2019 Bodo Inc. All rights reserved.

"""Test sort_values opration as called as df.sort_values()
   The C++ implementation uses the timsort which is a stable sort algorithm.
   Therefore, in the test we use mergesort, which guarantees that the equality
   tests can be made sensibly.
"""

import pandas as pd
import numpy as np
import bodo
import random
import string
import pytest
from bodo.tests.utils import check_func, is_bool_object_series, check_parallel_coherency
from bodo.utils.typing import BodoWarning, BodoError
import os

random.seed(5)
np.random.seed(3)


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
                "B": np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
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
            {"A": ["AA", "AA", "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
            [-2.1, 0.1, 1.1, 7.1, 9.0],
        ),
        # TODO: parallel range index with start != 0 and stop != 1
        # datetime columns, int index
        pd.DataFrame(
            {
                "A": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
                "B": pd.date_range(start="2013-09-04", end="2013-09-29", periods=5),
                "C": [1.1, np.nan, 4.2, 3.1, -1.3],
            },
            [-2, 1, 3, 5, 9],
        ),
        # TODO: timedelta
    ]
)
def df_value(request):
    return request.param


def test_sort_datetime_missing():
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
        if random.random() < 0.2:
            e_ent = "NaT"
        else:
            e_ent = list_date[idx]
        e_list.append(e_ent)

    df1 = pd.DataFrame({"A": e_list})

    check_func(
        test_impl1, (df1,),
    )
    check_func(
        test_impl2, (df1,),
    )
    check_func(
        test_impl3, (df1,),
    )
    check_func(
        test_impl4, (df1,),
    )


def test_single_col():
    """
    sorts a dataframe that has only one column
    modify bodo.ir.sort.MIN_SAMPLES to test sampling
    """
    fname = os.path.join("bodo", "tests", "data", "kde.parquet")

    def test_impl():
        df = pd.read_parquet(fname)
        df.sort_values("points", inplace=True)
        res = df.points.values
        return res

    save_min_samples = bodo.ir.sort.MIN_SAMPLES
    try:
        bodo.ir.sort.MIN_SAMPLES = 10
        check_func(
            test_impl, (),
        )
    finally:
        bodo.ir.sort.MIN_SAMPLES = save_min_samples  # restore global val


def test_sort_values_val():
    """
    Test sort_values(): with just 1 column\
    return value is a list(i.e. without columns)
    """

    def impl(df):
        return df.sort_values(by="A", kind="mergesort").A.values

    n = 10
    df = pd.DataFrame({"A": np.arange(n) + 1.0, "B": np.arange(n) + 1})
    check_func(impl, (df,))


def test_sort_values_1col(df_value):
    """
    Test sort_values(): with just 1 column
    """

    def impl(df):
        return df.sort_values(by="A", kind="mergesort")

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


def test_sort_values_1col_inplace(df_value):
    """
    Test sort_values(): with just 1 column
    """

    def impl(df):
        df.sort_values(by="A", kind="mergesort", inplace=True)
        return df

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


def test_sort_values_2col(df_value):
    """
    Test sort_values(): with 2 columns
    """

    def impl(df):
        return df.sort_values(by=["A", "B"], kind="mergesort", ascending=[True, False])

    if is_bool_object_series(df_value["A"]):
        check_func(impl, (df_value,), check_dtype=False)
        return

    check_func(impl, (df_value,))


def test_sort_values_2col_inplace(df_value):
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

    check_func(impl, (df_value,))


def test_sort_values_str():
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
            val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)

        A = np.random.randint(0, 1000, n)
        df = pd.DataFrame({"A": A, "B": str_vals}).drop_duplicates("A")
        return df

    # seeds should be the same on different processors for consistent input
    n = 17  # 1211
    df = _gen_df_str(n)
    check_func(test_impl, (df,))


def test_sort_values_1col_long_int_list():
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


def test_sort_values_2col_long_np():
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
@pytest.mark.slow
def test_sort_values_1col_np_array(dtype):
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
        test_impl, (get_quasi_random_dtype(n, dtype),),
    )


@pytest.mark.parametrize(
    "dtype1, dtype2",
    [
        (np.int8, np.int16),
        (np.uint8, np.int32),
        (np.int16, np.float64),
        (np.uint16, np.float32),
    ],
)
@pytest.mark.slow
def test_sort_values_2col_np_array(dtype1, dtype2):
    """
    Test sort_values(): with two columns, two types
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_quasi_random_dtype(n, dtype1, dtype2):
        eListA = np.array([0] * n, dtype=dtype1)
        eListB = np.array([0] * n, dtype=dtype2)
        for i in range(n):
            eValA = i * i % 34
            eValB = i * (i - 1) % 23
            eListA[i] = eValA
            eListB[i] = eValB
        return pd.DataFrame({"A": eListA, "B": eListB})

    n = 1000
    check_func(
        test_impl, (get_quasi_random_dtype(n, dtype1, dtype2),),
    )


@pytest.mark.parametrize(
    "n, len_str", [pytest.param(1000, 2, marks=pytest.mark.slow), (100, 1), (300, 2)]
)
def test_sort_values_strings_constant_length(n, len_str):
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

    check_func(
        test_impl, (get_random_strings_array(n, len_str),),
    )


@pytest.mark.parametrize(
    "n, len_str", [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (10, 30)]
)
def test_sort_values_strings_variable_length(n, len_str):
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

    df1 = get_random_var_length_strings_array(n, len_str)
    check_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_str",
    [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (10, 30), (100, 30)],
)
def test_sort_values_strings(n, len_str):
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
                val = "".join(random.choices(string.ascii_uppercase, k=k))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    df1 = get_random_strings_array(n, len_str)
    check_func(test_impl, (df1,))


@pytest.mark.parametrize(
    "n, len_siz", [(100, 30), pytest.param(1000, 10, marks=pytest.mark.slow), (10, 30)]
)
def test_sort_values_two_columns_nan(n, len_siz):
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

    df1 = get_random_dataframe_two_columns(n, len_siz)
    check_func(
        test_impl1, (df1,),
    )
    check_func(
        test_impl2, (df1,),
    )
    check_func(
        test_impl3, (df1,),
    )
    check_func(
        test_impl4, (df1,),
    )


def test_sort_values_by_index():
    """Sorting with a non-trivial index"""

    def test_impl1(df1):
        df2 = df1.sort_values("index_name")
        return df2

    df1 = pd.DataFrame({"A": [1, 2, 2]}, index=[2, 1, 0])
    df1.index.name = "index_name"
    check_func(test_impl1, (df1,), sort_output=False)


def test_sort_values_bool_list():
    """Test of NaN values for the sorting with vector of ascending"""

    def test_impl1(df1):
        df2 = df1.sort_values(by=["A", "B"], kind="mergesort", axis=0)
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


def test_sort_values_nullable_int_array():
    """Test of NaN values for the sorting for a nullable int bool array"""

    def test_impl(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort", axis=0
        )
        return df2

    nullarr = pd.array([13, None, 17], dtype="UInt16")
    df1 = pd.DataFrame({"A": nullarr})
    check_func(test_impl, (df1,))


def test_sort_with_nan_entries():
    """Test of the dataframe with nan entries"""

    def impl1(df):
        return df.sort_values(by="A", kind="mergesort")

    df1 = pd.DataFrame({"A": ["AA", np.nan, "", "D", "GG"]})
    df2 = pd.DataFrame({"A": [1, 8, 4, np.nan, 3]})
    df3 = pd.DataFrame({"A": pd.array([1, 2, None, 3], dtype="UInt16")})
    df4 = pd.DataFrame({"A": pd.Series([1, 8, 4, np.nan, 3], dtype="Int32")})
    df5 = pd.DataFrame({"A": pd.Series(["AA", np.nan, "", "D", "GG"])})
    check_func(impl1, (df1,), sort_output=False)
    check_func(impl1, (df2,), sort_output=False)
    check_func(impl1, (df3,), sort_output=False)
    check_func(impl1, (df4,), sort_output=False)
    check_func(impl1, (df5,), sort_output=False)


def test_sort_values_list_inference():
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


def test_sort_values_force_literal():
    """
    Test forcing JIT args to be literal if required by sort_values()
    """

    def impl(df, by, na_position):
        return df.sort_values(by=by, kind="mergesort", na_position=na_position)

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 0, -1, 4],
            "B": [1.2, 3.4, np.nan, 2.2, 3.1, -1.2],
            "C": np.arange(6),
        }
    )
    check_func(impl, (df, ["B"], "first"))


def test_list_string():
    """Sorting values by list of strings"""

    def test_impl(df1):
        df2 = df1.sort_values(by="A")
        return df2

    def rand_col_l_str(n):
        e_list = []
        for _ in range(n):
            if random.random() < -0.1:
                e_ent = np.nan
            else:
                e_ent = []
                for _ in range(random.randint(1, 2)):
                    k = random.randint(1, 3)
                    val = "".join(random.choices(["A", "B", "C"], k=k))
                    e_ent.append(val)
            e_list.append(e_ent)
        return e_list

    n = 100
    df1 = pd.DataFrame({"A": rand_col_l_str(n)})
    check_parallel_coherency(test_impl, (df1,), sort_output=True, reset_index=True)


# ------------------------------ error checking ------------------------------ #


df = pd.DataFrame({"A": [-1, 3, -3, 0, -1], "B": ["a", "c", "b", "c", "b"]})


def test_sort_values_by_const_str_or_str_list():
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
        match="'by' parameter only supports a constant column label or column labels",
    ):
        bodo.jit(impl2)(df)


def test_sort_values_by_labels():
    """
    Test sort_values(): 'by' is a valid label or label lists
    """

    def impl1(df):
        return df.sort_values(by=["C"])

    def impl2(df):
        return df.sort_values(by=["B", "C"])

    with pytest.raises(BodoError, match="invalid key .* for by"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="invalid key .* for by"):
        bodo.jit(impl2)(df)


def test_sort_values_axis_default():
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


def test_sort_values_ascending_bool():
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


def test_sort_values_inplace_bool():
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


def test_sort_values_kind_no_spec():
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


def test_sort_values_na_position_no_spec():
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


def test_inplace_sort_values_series():
    """
    Test sort_values(inplace=True): inplace not supported for Series.sort_values
    """

    def impl1(S):
        return S.sort_values(inplace=True)

    s = pd.Series([1, 8, 4, 10, 3])

    with pytest.raises(BodoError, match="'inplace' is not supported yet"):
        bodo.jit(impl1)(s)
