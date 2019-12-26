# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test drop_duplicate opration as called as df.drop_duplicates()
"""
import pandas as pd
import numpy as np
import bodo
import random
import pytest
from bodo.tests.utils import check_func


def test_drop_duplicates_1col():
    """
    Test drop_duplicates(): with just one column
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4]})
    check_func(test_impl, (df1,), sort_output=True)


def test_drop_duplicates_2col():
    """
    Test drop_duplicates(): with 2 columns of integers
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5]})
    check_func(test_impl, (df1,), sort_output=True)


def test_drop_duplicates_2col_int_string():
    """
    Test drop_duplicates(): 2 columns one integer, one string
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 3, 4], "B": ["bar", "baz", "bar", "baz", "bar"]})
    check_func(test_impl, (df1,), sort_output=True)


@pytest.mark.parametrize("n, len_siz", [(100,10), (30,3)])
def test_drop_duplicates_2col_random_nullable_int(n,len_siz):
    """
    Test drop_duplicates(): 2 columns drop duplicates with nullable_int_bool array
    """
    
    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2
    def get_random_column(n,len_siz):
        elist = []
        for _ in range(n):
            prob = random.randint(1,len_siz)
            if prob==1:
                elist.append(None)
            else:
                elist.append(prob)
        return pd.array(elist, dtype='UInt16')
    def get_random_dataframe(n,len_siz):
        elist1 = get_random_column(n,len_siz)
        elist2 = get_random_column(n,len_siz)
        return pd.DataFrame({'A':elist1,'B':elist2})

    random.seed(5)
    df1 = get_random_dataframe(n,len_siz)
    check_func(test_impl, (df1,), sort_output=True)





def test_drop_duplicates_2col_int_numpynan_bool():
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


def test_drop_duplicates_1col_nullable_int():
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


def test_drop_duplicates_2col_int_np_float():
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame(
        {"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], np.float)}
    )
    check_func(test_impl, (df1,), sort_output=True)


def test_drop_duplicates_2col_int_np_int():
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], np.int)})
    check_func(test_impl, (df1,), sort_output=True)


def test_drop_duplicates_2col_int_np_int_index():
    """
    Test drop_duplicates(): 2 columns one integer, one numpy array of floats and an array in indices
    """

    def test_impl(df1):
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame(
        {"A": [3, 3, 3, 3, 4], "B": np.array([1, 2, 1, 2, 17], np.int)},
        index=[0, 1, 2, 3, 4],
    )
    check_func(test_impl, (df1,), sort_output=True)


def test_drop_duplicatee_large_size():
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


#
# Tests below should be uncommented when functionality is implemented.
#


# def test_dd_subset():
#    """
#    Test merge(): sequentially merge on more than one integer key columns
#    """
#    def test_impl(df1):
#        df2 = df1.drop_duplicates(subset=["A"])
#        return df2
#    bodo_func = bodo.jit(test_impl)
#    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 2, 2, 5, 5]})
#    pd.testing.assert_frame_equal(
#        bodo_func(df1).sort_values("A").reset_index(drop=True),
#        test_impl(df1).sort_values("A").reset_index(drop=True),
#    )


# def test_dd_subset_last():
#    """
#    Test merge(): sequentially merge on more than one integer key columns
#    """
#
#    def test_impl(df1):
#        df2 = df1.drop_duplicates(subset=["A"], keep="last")
#        return df2
#    bodo_func = bodo.jit(test_impl)
#    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 5, 9, 5, 5]})
#    pd.testing.assert_frame_equal(
#        bodo_func(df1).sort_values("A").reset_index(drop=True),
#        test_impl(df1).sort_values("A").reset_index(drop=True),
#    )


# def test_dd_subset_false():
#    """
#    Test merge(): sequentially merge on more than one integer key columns
#    """
#    def test_impl(df1):
#        df2 = df1.drop_duplicates(subset=["A"], keep=False)
#        return df2
#    bodo_func = bodo.jit(test_impl)
#    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4], "B": [1, 5, 9, 5, 5]})
#    pd.testing.assert_frame_equal(
#        bodo_func(df1).sort_values("A").reset_index(drop=True),
#        test_impl(df1).sort_values("A").reset_index(drop=True),
#    )
