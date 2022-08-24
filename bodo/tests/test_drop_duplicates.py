# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test drop_duplicate opration as called as df.drop_duplicates()
"""
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func, gen_random_list_string_array


@pytest.fixture(
    params=[
        pytest.param(
            np.array(
                [
                    [[[1, 2, 8], [3]], [[2, None]]],
                    [[[1, 2], [3]], [[2, None]]],
                    [[[3], [9], [1, None, 4]]],
                    [[[3], [], [1, None, 4]]],
                    [[[4, 5, 6, 11], []], [[1]], [[1, 2]]],
                    [[[4, 5, 6], []], [[1]], [[1, 2]]],
                    [[[4, 5, 6], []], [[1], [4, 3]]],
                ]
                * 2,
                dtype=object,
            ),
            id="nested_array_0_",
        ),
        pytest.param(
            np.array(
                [
                    [[["a", "bb", "c"], ["dddd"]], [["ee", None]]],
                    [[["fff", "g"], ["hh"]], [["iiiii", None]]],
                    [[["jj"], ["kkkk"], ["lll", None, "mm"]]],
                    [[["n"], [], ["o", None, "p"]]],
                    [[["qq", "rr", "sssssss", "ttt"], []], [["u"]], [["vvv", "ww"]]],
                    [[["x", "yy", "zzz"], []], [["aa"]], [["b", "c"]]],
                    [[["DD", "eee", "ff"], ["E1"]], [["gg"], ["hhhhhhh", "i"]]],
                ]
                * 2,
                dtype=object,
            ),
            id="nested_array_1_",
        ),
        # TODO None value fields in structs is not supported in typing.
        # Using -1 instead of None. Can change to None in the future
        pytest.param(
            np.array(
                [
                    [{"A": 1, "B": 2}, {"A": 10, "B": 20}],
                    [{"A": 11, "B": 21}, {"A": 10, "B": 201}, {"A": 10, "B": 20}],
                    [{"A": 3, "B": -11}],
                    [{"A": 31, "B": -1}, {"A": 3, "B": -1}],
                    [{"A": 5, "B": 6}, {"A": 50, "B": 60}],
                    [{"A": 53, "B": 61}, {"A": 50, "B": 60}, {"A": 501, "B": 604}],
                    [{"A": 52, "B": 61}, {"A": 501, "B": 604}, {"A": 500, "B": 600}],
                ]
                * 2,
                dtype=object,
            ),
            id="nested_array_2_",
        ),
        pytest.param(
            np.array(
                [
                    {"A": 3.0, "B": True},
                    {"A": 3.0, "B": True},
                    {"A": 5.0, "B": False},
                    {"A": 5.0, "B": True},
                    {"A": 30.0, "B": False},
                    {"A": 30.0, "B": False},
                    {"A": 3.0, "B": True},
                ]
                * 2
            ),
            id="nested_array_3_",
        ),
        pytest.param(
            np.array(
                [
                    {"A": [31, 5], "B": [-11, 9]},
                    {"A": [3, 50, 17], "B": [-1]},
                    {"A": [51, 13], "B": [6, 9, 12]},
                    {"A": [5], "B": [121, 15]},
                    {"A": [31, 41], "B": [51, 52, 53]},
                    {"A": [33], "B": [40, 50, 51, 52, 53]},
                    {"A": [30, 40, 19], "B": [1, 43, 50]},
                ]
                * 2
            ),
            id="nested_array_4_",
        ),
        pytest.param(
            np.array(
                [
                    {"A": [Decimal("1.0"), Decimal("2.2")], "B": [Decimal("4.14")]},
                    {
                        "A": [Decimal("0"), Decimal("3.2"), Decimal("4")],
                        "B": [Decimal("-1")],
                    },
                    {
                        "A": [Decimal("5")],
                        "B": [Decimal("644"), Decimal("9.1"), Decimal("154")],
                    },
                    {
                        "A": [Decimal("10.0"), Decimal("13.4")],
                        "B": [Decimal("3.14159")],
                    },
                    {"A": [Decimal("15.0"), None], "B": [Decimal("2.05")]},
                    {
                        "A": [Decimal("30"), Decimal("5.2")],
                        "B": [Decimal("0"), Decimal("2")],
                    },
                    {"A": [Decimal("-1"), Decimal("-2"), None], "B": [Decimal("60")]},
                ]
                * 2
            ),
            id="nested_array_5_",
        ),
        pytest.param(
            np.array(
                [
                    {"A": {"A1": 10, "A2": 2}, "B": {"B1": -11, "B2": 4}},
                    {"A": {"A1": 110, "A2": 2}, "B": {"B1": -11, "B2": 41}},
                    {"A": {"A1": -51, "A2": -9}, "B": {"B1": -15, "B2": 13}},
                    {"A": {"A1": -5, "A2": -9}, "B": {"B1": -15, "B2": 13}},
                    {"A": {"A1": -120, "A2": 2}, "B": {"B1": 14, "B2": 2}},
                    {"A": {"A1": -12, "A2": 2}, "B": {"B1": 14, "B2": 2}},
                    {"A": {"A1": -112, "A2": 12}, "B": {"B1": 144, "B2": 21}},
                ]
                * 2
            ),
            id="nested_array_6_",
        ),
        pytest.param(
            pd.Series(
                [
                    (123, 1000, np.int8(1), np.float32(6.5)),
                    (75, 2000, np.int8(12), np.float32(1.5)),
                    (12, 3000, np.int8(120), np.float32(33.3)),
                    (788, 4000, np.int8(22), np.float32(6.2)),
                    (62, 5000, np.int8(67), np.float32(7.1)),
                    (1, 6000, np.int8(100), np.float32(50.56)),
                    (33, 7000, np.int8(250), np.float32(67.5)),
                ]
                * 2
            ),
            id="nested_array_7_",
        ),
    ]
)
def nested_arrays_value(request):
    return request.param


@pytest.mark.smoke
@pytest.mark.parametrize(
    "test_df",
    [
        # all string
        pd.DataFrame(
            {
                "A": ["A", "B", "A", "B", "C", "D", "E", "A", "G"],
                "B": ["F", "E", "F", "S", "C", "F", "H", "L", "E"],
            }
        ),
        # mix string and numbers and index
        pd.DataFrame(
            {"A": [1, 3, 4, 1, 2, 3, 5], "B": ["F", "E", "A", "F", "S", "C", "QQ"]},
            index=[3, 1, 0, 2, 4, 6, 5],
        ),
        # string index
        pd.DataFrame(
            {"A": [1, 3, 1, 2, 3, 7, 11], "B": ["F", "E", "F", "S", "C", "AB", "d2"]},
            index=["A", "C", "D", "E", "AA", "B", "K"],
        ),
        # all numbers
        pd.DataFrame(
            {"A": [1, 3, 1, 2, 3], "B": [1.2, 3.1, 1.2, 2.5, 3.3]},
            index=[3, 1, 2, 4, 6],
        ),
        # nullable binary values
        pd.DataFrame(
            {
                "A": [
                    bytes(5),
                    b"hello",
                    b"hello",
                    b"qwerty",
                    b"HELLO",
                    b"h",
                    b"hbjkl",
                    b"asdgaasdg",
                    b"asdjgalsd",
                ],
                "B": [
                    b"a",
                    b"c",
                    b"a",
                    b"ab",
                    b"abcd",
                    b"abc",
                    None,
                    b"masdg",
                    b"asdjgalsd",
                ],
                "C": [
                    b"a",
                    None,
                    b"hi",
                    None,
                    b"abc",
                    b"d",
                    b"asdeefeky",
                    b"mlrgh",
                    b"adafslg",
                ],
            },
        ),
    ],
)
def test_df_drop_duplicates(test_df, memory_leak_check):
    def impl(df):
        return df.drop_duplicates()

    check_func(impl, (test_df,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_1col(memory_leak_check):
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4]})
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_2col(memory_leak_check):
    """
    Test drop_duplicates(): with 2 columns of integers
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5]})
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_2col_int_string(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one string
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame(
        {
            "A": [3, 3, 5, 1, 3, 3, 4, 7, 2],
            "B": ["bar", "baz", "A", "B", "bar", "baz", "bar", "AB", "E1"],
        }
    )
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
@pytest.mark.parametrize("n, len_siz", [(100, 10), (30, 3)])
def test_drop_duplicates_2col_random_nullable_int(n, len_siz, memory_leak_check):
    """
    Test drop_duplicates(): 2 columns drop duplicates with nullable_int_bool array
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    def get_random_column(n, len_siz):
        elist = []
        for _ in range(n):
            prob = random.randint(1, len_siz)
            if prob == 1:
                elist.append(None)
            else:
                elist.append(prob)
        return pd.array(elist, dtype="UInt16")

    def get_random_dataframe(n, len_siz):
        elist1 = get_random_column(n, len_siz)
        elist2 = get_random_column(n, len_siz)
        return pd.DataFrame({"A": elist1, "B": elist2})

    random.seed(5)
    df1 = get_random_dataframe(n, len_siz)
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_list_string_array_type_specific(memory_leak_check):
    """Test of list_string_array_type in a specific case"""

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": ["AB", "A,B,C", "AB,CD", "D,E", "D,F", "A,B,C", "D,E"]})
    df1.A = df1.A.str.split(",")
    check_func(test_impl, (df1,), sort_output=True, convert_columns_to_pandas=True)


@pytest.mark.slow
def test_list_string_array_type_random(memory_leak_check):
    """Test of list_string_array_type in parallel with a random list
    Also put a test on two columns for the combine_hash functionality
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    random.seed(5)

    # We remove the None from the first entry because it leads to a comparison
    # error nan vs None in the string comparison.
    n = 100
    df1 = pd.DataFrame({"A": gen_random_list_string_array(3, n)})
    df2 = pd.DataFrame(
        {
            "A": gen_random_list_string_array(3, n),
            "B": gen_random_list_string_array(3, n),
        }
    )
    check_func(test_impl, (df1,), sort_output=True, convert_columns_to_pandas=True)
    check_func(test_impl, (df2,), sort_output=True, convert_columns_to_pandas=True)


@pytest.mark.slow
def test_drop_duplicates_2col_int_numpynan_bool(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one nullable_int_bool array
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    def get_array(n):
        e_list_a = np.array([0] * n)
        e_list_b = []
        choice = [True, False, np.nan]
        for i in range(n):
            e_list_a[i] = i % 40
            e_list_b.append(choice[i % 3])
        df1 = pd.DataFrame({"A": e_list_a, "B": e_list_b})
        return df1

    check_func(test_impl, (get_array(150),), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_1col_nullable_int(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one nullable_int_bool array
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    def get_array(n):
        e_list = []
        for i in range(n):
            e_val = i % 40
            if e_val == 39:
                e_val = np.nan
            e_list.append(e_val)
        df1 = pd.DataFrame({"A": e_list})
        return df1

    check_func(test_impl, (get_array(150),), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_2col_int_np_float(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], float)})
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_2col_int_np_int(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], int)})
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicates_2col_int_np_int_index(memory_leak_check):
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats and an array in indices
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame(
        {"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], int)},
        index=[0, 1, 2, 3, 4],
    )
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.slow
def test_drop_duplicate_large_size(memory_leak_check):
    """
    Test drop_duplicates(): large size entries
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    def get_df(n):
        e_list_a = np.array([0] * n, dtype=np.int64)
        e_list_b = np.array([0] * n, dtype=np.int64)
        for i in range(n):
            idx = i % 100
            i_a = idx % 10
            i_b = (idx - i_a) / 10
            e_list_a[i] = i_a
            e_list_b[i] = i_b
        df1 = pd.DataFrame({"A": e_list_a, "B": e_list_b})
        return df1

    check_func(test_impl, (get_df(396),), sort_output=True)
    check_func(test_impl, (get_df(11111),), sort_output=True)


def test_drop_duplicate_nested_arrays(nested_arrays_value):
    def f(df):
        df2 = df.drop_duplicates()
        return df2

    df = pd.DataFrame({"A": [10, 5, 3000, 11, 0, 9, 12] * 2, "B": nested_arrays_value})
    check_func(f, (df,), sort_output=True, convert_columns_to_pandas=True)


#
# Tests below should be uncommented when functionality is implemented.
#


def test_dd_subset(memory_leak_check):
    """
    Test drop_duplicates subset
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset=["B"])
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5]})
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)


def test_dd_subset_many_labels(memory_leak_check):
    """
    Test drop_duplicates subset with a list of multiple labels.
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset=["A", "C"])
        return df2

    df1 = pd.DataFrame(
        {"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5], "C": [1, 2, 3, 4, 4]}
    )
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)


def test_dd_subset_label(memory_leak_check):
    """
    Test drop_duplicates subset with a single label
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset="B")
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5]})
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)


def test_dd_subset_int_label(memory_leak_check):
    """
    Test drop_duplicates subset with an int label
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset=2)
        return df2

    df1 = pd.DataFrame({1: [3, 3, 3, 1, 4], 2: [1, 2, 2, 5, 5]})
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)


# TODO: Add memory_leak_check. There appears to be a leak for struct arrays.
def test_dd_map_array_non_drop():
    """
    Test drop_duplicates works with a MapArray column
    that isn't part of the subset.
    """

    def test_impl(df1):
        df1["B"] = df1["A"].apply(lambda val: {str(j): val for j in range(val)})
        df2 = df1.drop_duplicates(subset="A")
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4]})
    # We can only run the sequential test because Pandas can't sort a dict column
    check_func(test_impl, (df1,), reset_index=True, dist_test=False)


# TODO: Add memory_leak_check. There appears to be a leak for struct arrays.
def test_dd_struct_array_non_drop():
    """
    Test drop_duplicates works with a StructArray column
    that isn't part of the subset.
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset="A")
        return df2

    df1 = pd.DataFrame(
        {
            "A": [3, 3, 3, 1, 4],
            "B": np.array(
                [
                    {"A": 1.0, "B": True},
                    {"A": 3.0, "B": False},
                    {"A": 5.0, "B": True},
                    {"A": 10.0, "B": True},
                    {"A": 30.0, "B": False},
                ]
            ),
        }
    )
    # We can only run the sequential test because Pandas can't sort a dict column
    check_func(test_impl, (df1,), reset_index=True, dist_test=False)


@pytest.mark.skip("keep argument not supported")
def test_dd_subset_last(memory_leak_check):
    """
    Test drop_duplicates subset with keep='last'
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset=["A"], keep="last")
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 5, 9, 5, 5]})
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)


@pytest.mark.skip("keep argument not supported")
def test_dd_subset_false(memory_leak_check):
    """
    Test drop_duplicates subset with keep=False
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates(subset=["A"], keep=False)
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 5, 9, 5, 5]})
    check_func(test_impl, (df1,), reset_index=True, sort_output=True)
