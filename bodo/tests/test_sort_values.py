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
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoWarning, BodoError


def test_sort_values_1col():
    """
    Test sort_values(): with just one column
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

    n = 100
    check_func(test_impl1, (get_quasi_random(n),), sort_output=False)
    check_func(test_impl2, (get_quasi_random(n),), sort_output=False)


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
    check_func(test_impl, (get_quasi_random_dtype(n, dtype),), sort_output=False)


def test_sort_values_2col():
    """
    Test sort_values(): with just one column
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
    check_func(test_impl1, (get_quasi_random(n),), sort_output=False)
    check_func(test_impl2, (get_quasi_random(n),), sort_output=False)


@pytest.mark.parametrize(
    "dtype1, dtype2",
    [
        (np.int8, np.int16),
        (np.uint8, np.int32),
        (np.int16, np.float64),
        (np.uint16, np.float32),
    ],
)
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
        test_impl, (get_quasi_random_dtype(n, dtype1, dtype2),), sort_output=False
    )


@pytest.mark.parametrize("n, len_str", [(1000, 2), (100, 1), (300, 2)])
def test_sort_values_strings_constant_length(n, len_str):
    """
    Test sort_values(): with 1 column and strings of constant length
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        random.seed(5)
        str_vals = []
        for _ in range(n):
            val = "".join(random.choices(string.ascii_uppercase, k=len_str))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    check_func(test_impl, (get_random_strings_array(n, len_str),), sort_output=False)


@pytest.mark.parametrize("n, len_str", [(100, 30), (1000, 10), (10, 30)])
def test_sort_values_strings_variable_length(n, len_str):
    """
    Test sort_values(): with 1 column and strings of variable length all of character A.
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_var_length_strings_array(n, len_str):
        random.seed(5)
        str_vals = []
        for _ in range(n):
            k = random.randint(1, len_str)
            val = "A" * k
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    check_func(
        test_impl, (get_random_var_length_strings_array(n, len_str),), sort_output=False
    )


def test_sort_values_string_list():
    """Test the list of string. Missing values in a string array"""

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort", na_position="last", axis=0)
        return df2

    def get_random_string_dataframe(n, len_str):
        random.seed(5)
        str_vals = []
        for _ in range(n):
            val1 = random.randint(1, len_str)
            if val1 == 1:
                val = np.nan
            else:
                val = "".join(random.choices(string.ascii_uppercase, k=val1))
            str_vals.append(val)
        df = pd.DataFrame({"A": str_vals})
        return df

    df1 = get_random_string_dataframe(100, 10)
    check_func(test_impl1, (df1,), sort_output=False)


@pytest.mark.parametrize("n, len_str", [(100, 30), (1000, 10), (10, 30), (100, 30)])
def test_sort_values_strings(n, len_str):
    """
    Test sort_values(): with 1 column and strings of variable length and variable characters.
    with some entries assigned to missing values
    """

    def test_impl(df1):
        df2 = df1.sort_values(by="A", kind="mergesort")
        return df2

    def get_random_strings_array(n, len_str):
        random.seed(5)
        str_vals = []
        for _ in range(n):
            prob = random.randint(1, 10)
            if prob == 1:
                val = np.nan
            else:
                k = random.randint(1, len_str)
                val = "".join(random.choices(string.ascii_uppercase, k=k))
            str_vals.append(val)
        str_valB = ["A", np.nan, "B", np.nan]
        df = pd.DataFrame({"A": str_valB})
        return df

    bodo_func = bodo.jit(test_impl)
    df1 = get_random_strings_array(n, len_str)
    pd.testing.assert_frame_equal(
        bodo_func(df1), test_impl(df1),
    )


#    TODO: Solve the bug in the check that makes the following fail
#    (problem of conversion of dataFrames)
#    check_func(
#        test_impl, (get_random_strings_array(n, len_str),), sort_output=False
#    )


@pytest.mark.parametrize("n, len_siz", [(100, 30), (1000, 10), (10, 30), (100, 30)])
def test_sort_values_two_columns_nan(n, len_siz):
    """Test with two columns with some NaN entries"""

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
        random.seed(5)
        df = pd.DataFrame(
            {"A": get_random_column(n, len_siz), "B": get_random_column(n, len_siz)}
        )
        return df

    df1 = get_random_dataframe_two_columns(n, len_siz)
    check_func(test_impl1, (df1,), sort_output=False)
    check_func(test_impl2, (df1,), sort_output=False)
    check_func(test_impl3, (df1,), sort_output=False)
    check_func(test_impl4, (df1,), sort_output=False)


def test_sort_values_simplest():
    """Simplest case of sort_values"""

    def test_impl1(df1):
        df2 = df1.sort_values("A")
        return df2

    df1 = pd.DataFrame({"A": [1, 2, 2]})
    check_func(test_impl1, (df1,), sort_output=False)


def test_sort_values_by_index():
    """Sorting with a non-trivial index"""

    def test_impl1(df1):
        df2 = df1.sort_values("index_name")
        return df2

    df1 = pd.DataFrame({"A": [1, 2, 2]}, index=[2, 1, 0])
    df1.index.name = "index_name"
    check_func(test_impl1, (df1,), sort_output=False)


def test_sort_values_nan_case():
    """Test of NaN values for the sorting with all possible values of ascending and na_position"""

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


def test_sort_values_nan_case_simple():
    """Test of NaN values for the sorting in a numpy array"""

    def test_impl(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort", axis=0
        )
        return df2

    df1 = pd.DataFrame({"A": [2, np.nan, 7, np.nan]})
    check_func(test_impl, (df1,), sort_output=False)


def test_sort_values_nullable_int_array():
    """Test of NaN values for the sorting for a nullable int bool array"""

    def test_impl(df1):
        df2 = df1.sort_values(
            by="A", ascending=True, na_position="last", kind="mergesort", axis=0
        )
        return df2

    nullarr = pd.array([13, None, 17], dtype="UInt16")
    df1 = pd.DataFrame({"A": nullarr})
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(
        bodo_func(df1), test_impl(df1),
    )


def test_sort_values_numpy_nan():
    """Test of the code without NaN values"""

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", kind="mergesort", axis=0)
        return df2

    df1 = pd.DataFrame({"A": [13, np.nan, 17]})
    check_func(test_impl1, (df1,), sort_output=False)


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


def test_sort_values_two_columns():
    """Test of the code without NaN values"""

    def test_impl1(df1):
        df2 = df1.sort_values(by=["A", "B"], kind="mergesort", axis=0)
        return df2

    df1 = pd.DataFrame(
        {"A": [2, -3, 7, 10, -1, -4, 0, 1, 2], "B": [3, 6, 0, 1, 2, -4, 7, 7, 2]}
    )
    check_func(test_impl1, (df1,), sort_output=False)


def test_sort_values_regular_case():
    """Test of the code without NaN values"""

    def test_impl1(df1):
        df2 = df1.sort_values(by="A", ascending=True, kind="mergesort", axis=0)
        return df2

    def test_impl2(df1):
        df2 = df1.sort_values(by="A", ascending=False, kind="mergesort", axis=0)
        return df2

    df1 = pd.DataFrame(
        {"A": [2, -3, 7, 10, -1, -4, 0, 1, 2], "B": [3, 6, 0, 1, 2, -4, 7, 7, 2]}
    )
    check_func(test_impl1, (df1,), sort_output=False)
    check_func(test_impl2, (df1,), sort_output=False)


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

    def impl6(df):
        return df.sort_values(by=["A", "B"], ascending=(True, False))

    with pytest.raises(
        BodoError, match="'ascending' parameter must be of type bool or list of bool"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'ascending' parameter must be of type bool or list of bool"
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        ValueError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl3)(df, True)
    with pytest.raises(
        ValueError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl4)(df)
    with pytest.raises(
        ValueError,
        match="ascending should be bool or a list of bool of the number of keys",
    ):
        bodo.jit(impl5)(df)
    with pytest.raises(
        BodoError, match="parameter must be of type bool or list of bool"
    ):
        bodo.jit(impl6)(df)


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
