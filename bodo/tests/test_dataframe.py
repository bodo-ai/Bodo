# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Unittests for DataFrames
"""
import datetime
import random
import sys
import unittest
from decimal import Decimal

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    AnalysisTestPipeline,
    _get_dist_arg,
    _test_equal,
    check_func,
    count_array_OneDs,
    count_array_REPs,
    count_parfor_OneDs,
    count_parfor_REPs,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_array_struct_list_int,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    get_start_end,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError, BodoWarning


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
                2: [1.1, np.nan, 4.2, 3.1, -1.3],
                "C": [True, False, False, np.nan, True],
            },
            ["A", "BA", "", "DD", "C"],
        ),
        # uint8, float32 dtypes
        pytest.param(
            pd.DataFrame(
                {
                    3: np.array([1, 8, 4, 0, 3], dtype=np.uint8),
                    1: np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
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
                {"A": [1, 8, 4, 1, -2] * 3, "B": ["A", "B", "CG", "ACDE", "C"] * 3},
                range(0, 5 * 3, 1),
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: parallel range index with start != 0 and stop != 1
        # int index
        pd.DataFrame(
            {"A": [1, 8, 4, 1, -3] * 2, "B": ["A", "B", "CG", "ACDE", "C"] * 2},
            [-2, 1, 3, 5, 9, -3, -5, 0, 4, 7],
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
            pd.DataFrame({"A": [1, 8, 4, 11, -3], 2: [1.1, np.nan, 4.2, 3.1, -1.1]}),
            marks=pytest.mark.slow,
        ),
        # uint8, float32 dtypes
        pd.DataFrame(
            {
                55: np.array([1, 8, 4, 0, 2], dtype=np.uint8),
                -3: np.array([1.1, np.nan, 4.2, 3.1, -1.1], dtype=np.float32),
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
        pd.DataFrame({"product": ["a", "b", "c", "d", "e", "f"]}),
        pd.DataFrame(
            {"product": ["a", "b", "c", "d", "e", "f"], "keys": [1, 2, 3, 4, 5, 6]}
        ),
    ]
)
def column_name_df_value(request):
    return request.param


@pytest.fixture(
    params=[
        pd.DataFrame({"a": [1, 2] * 20, "b": [True, False] * 20, "c": [1.0, 2.0] * 20}),
    ]
)
def select_dtypes_df(request):
    return request.param


def test_df_select_dtypes_str_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes("bool")

    def test_impl2(df):
        return df.select_dtypes("float64")

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


def test_df_select_dtypes_str_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(exclude="bool")

    def test_impl2(df):
        return df.select_dtypes(exclude="float64")

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.skip(reason="Numba issue with np.number")
def test_df_select_dtypes_str_include_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes("number", "float64")

    check_func(test_impl, (df,))


def test_df_select_dtypes_type_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(np.bool)

    def test_impl2(df):
        return df.select_dtypes(np.float64)

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


def test_df_select_dtypes_type_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(exclude=np.bool)

    def test_impl2(df):
        return df.select_dtypes(exclude=np.float64)

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.skip(reason="Numba issue with np.number")
def test_df_select_dtypes_type_include_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(np.number, np.float64)

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Numba issue with one element lists")
def test_df_select_dtypes_list_one_elem_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(["bool"])

    def test_impl2(df):
        return df.select_dtypes([np.bool_])

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.slow
def test_df_select_dtypes_list_multi_elem_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(["float64", "bool"])

    def test_impl2(df):
        return df.select_dtypes([np.float64, np.bool_])

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.skip(reason="Numba issue with np.number")
def test_df_select_dtypes_list_number_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes([np.number, "bool"])

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Numba issue with one element lists")
def test_df_select_dtypes_list_one_elem_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(exclude=["bool"])

    def test_impl2(df):
        return df.select_dtypes(exclude=[np.float64])

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.slow
def test_df_select_dtypes_list_multi_elem_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(exclude=["float64", "bool"])

    def test_impl2(df):
        return df.select_dtypes(exclude=[np.float64, "bool"])

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.skip(reason="Numba issue with np.number")
def test_df_select_dtypes_list_number_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(exclude=[np.number, "bool"])

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_type_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(np.datetime64)

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_type_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(exclude=np.datetime64)

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_str_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes("datetime64")

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_str_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(exclude="datetime64")

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_list_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes([np.datetime64, "datetime64"])

    check_func(test_impl, (df,))


@pytest.mark.skip(reason="Index issue when creating Empty Dataframes #1596")
def test_df_select_dtypes_missing_list_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl(df):
        return df.select_dtypes(exclude=[np.datetime64, "datetime64"])

    check_func(test_impl, (df,))


@pytest.mark.smoke
def test_assign(memory_leak_check, is_slow_run):
    """Assign statements"""

    def test_impl1(df):
        return df.assign(B=42)

    def test_impl2(df):
        return df.assign(B=2 * df["A"])

    def test_impl3(df):
        return df.assign(B=2 * df["A"], C=42)

    def test_impl4(df):
        return df.assign(B=df["A"] + "XYZ")

    def test_impl5(df):
        return df.assign(A=df["B"], B=df["B"])

    df_int = pd.DataFrame({"A": [1, 2, 3] * 2})
    df_str = pd.DataFrame({"A": ["a", "b", "c", "d", "e", "f", "g"]})
    df_twocol = pd.DataFrame({"A": [1, 2, 3] * 2, "B": [4, 5, 6] * 2})
    check_func(test_impl1, (df_int,))
    if not is_slow_run:
        return
    check_func(test_impl2, (df_int,))
    check_func(test_impl3, (df_int,))
    check_func(test_impl4, (df_str,))
    check_func(test_impl5, (df_twocol,))


@pytest.mark.slow
def test_unbox_df1(df_value, memory_leak_check):
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
        return df_arg.iloc[:, 0]

    check_func(impl3, (df_value,))


@pytest.mark.slow
def test_unbox_df2(column_name_df_value, memory_leak_check):
    """unbox column with name overlaps with pandas function"""

    def impl1(df_arg):
        return df_arg["product"]

    check_func(impl1, (column_name_df_value,))


@pytest.mark.slow
def test_box_repeated_names(memory_leak_check):
    """test dataframe boxing where column names repeat"""

    def impl1(df):
        return pd.concat([df, df], axis=1)

    df = pd.DataFrame({"A": [3, 2, 1, -4, 7]})
    check_func(impl1, (df,))


@pytest.mark.slow
def test_unbox_df3(memory_leak_check):
    # unbox dataframe datetime and unsigned int indices
    def impl(df):
        return df

    df1 = pd.DataFrame(
        {"A": [3, 5, 1, -1, 4]},
        pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
    )
    df2 = pd.DataFrame(
        {"A": [3, 5, 1, -1, 4]},
        np.array([1, 8, 4, 0, 2], dtype=np.uint8),
    )
    check_func(impl, (df1,))
    check_func(impl, (df2,))


@pytest.mark.slow
def test_unbox_df_multi(memory_leak_check):
    """
    box/unbox dataframe with MultiIndex columns structure (sometimes created in groupby,
    ...)
    """
    # TODO: add a MultiIndex dataframe to all tests
    def impl(df):
        return df

    df = pd.DataFrame(
        data=np.arange(36).reshape(6, 6),
        columns=pd.MultiIndex.from_product((["A", "B"], ["CC", "DD", "EE"])),
    )
    check_func(impl, (df,))


@pytest.mark.slow
def test_empty_df_unbox(memory_leak_check):
    """test boxing/unboxing of an empty df"""

    def impl(df):
        return df

    df = pd.DataFrame()
    check_func(impl, (df,))


@pytest.mark.slow
def test_empty_df_create(memory_leak_check):
    """test creation of an empty df"""

    def impl1():
        return pd.DataFrame()

    def impl2():
        return pd.DataFrame(columns=["A"])

    def impl3():
        return pd.DataFrame(columns=["A"], dtype=np.float32)

    check_func(impl1, ())
    # check_typing_issues=False since the input is intentionally empty
    check_func(impl2, (), check_typing_issues=False)
    check_func(impl3, ())


@pytest.mark.smoke
def test_empty_df_set_column(memory_leak_check):
    """test column setitem of an empty df"""

    def impl1(n):
        df = pd.DataFrame()
        df["A"] = np.arange(n) * 2
        return df

    def impl2(n):
        df = pd.DataFrame()
        df["A"] = pd.Series(np.arange(n) * 2, index=np.ones(n))
        return df

    check_func(impl1, (11,))
    check_func(impl2, (11,))


def test_empty_df_drop_column(memory_leak_check):
    """test dropping the only column of a dataframe so it becomes empty"""

    def impl1(n):
        df = pd.DataFrame({"A": np.arange(n) * 2})
        df.drop(columns=["A"])
        return df

    check_func(impl1, (11,))


@pytest.mark.slow
def test_df_from_np_array_int(memory_leak_check):
    """
    Create a dataframe from numpy 2D-array of type int
    """

    def impl():
        arr = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
        np_arr = np.array(arr)
        return pd.DataFrame({"A": np_arr[:, 0], "B": np_arr[:, 1], "C": np_arr[:, 2]})

    check_func(impl, (), is_out_distributed=False)


@pytest.mark.slow
def test_create_df_force_const(memory_leak_check):
    """
    Test forcing dataframe column name to be constant in pd.DataFrame()
    """

    def impl(c_name, n):
        return pd.DataFrame({"A": np.ones(n), c_name: np.arange(n)})

    check_func(impl, ("BB", 11))


@pytest.mark.slow
def test_df_from_np_array_bool(memory_leak_check):
    """
    Create a dataframe from numpy 2D-array of type bool
    """

    def impl():
        arr = [[True, False, False, False, True], [False, False, True, True, False]]
        np_arr = np.array(arr)
        return pd.DataFrame({"A": np_arr[:, 0], "B": np_arr[:, 1], "C": np_arr[:, 2]})

    check_func(impl, (), is_out_distributed=False)


def test_create_df_scalar(memory_leak_check):
    """
    Test scalar to array conversion in pd.DataFrame()
    """

    def impl(n):
        return pd.DataFrame({"A": 2, "B": np.arange(n)})

    check_func(impl, (11,))


def test_df_multi_get_level(memory_leak_check):
    """
    getitem with string to get a level of dataframe with MultiIndex columns structure
    """

    def impl1(df):
        return df["B"]

    def impl2(df):
        return df.A

    def impl3(df):
        return df.A.CC

    df = pd.DataFrame(
        data=np.arange(36).reshape(6, 6),
        columns=pd.MultiIndex.from_product((["A", "B"], ["CC", "DD", "EE"])),
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.parametrize(
    "data", [pd.DataFrame({"A": range(10)}), np.arange(10), pd.Series(np.arange(10))]
)
def test_rebalance_simple(data, memory_leak_check):
    def impl(data):
        return bodo.rebalance(data)

    check_func(impl, (data,), py_output=data)

    if bodo.get_size() == 2:
        if bodo.get_rank() == 0:
            data_chunk = data[:9]
        else:
            data_chunk = data[9:]
        res = bodo.jit(distributed=["data"], all_returns_distributed=True)(impl)(
            data_chunk
        )
        assert len(res) == 5
        res = bodo.gatherv(res)
        if bodo.get_rank() == 0:
            if isinstance(data, pd.DataFrame):
                pd.testing.assert_frame_equal(data, res)
            else:
                np.testing.assert_array_equal(data, res)


@pytest.mark.parametrize("seed", [None, 151397])
@pytest.mark.parametrize(
    "data", [pd.DataFrame({"A": range(100)}), np.arange(100), pd.Series(np.arange(100))]
)
def test_random_shuffle(seed, data, memory_leak_check):
    def impl(data):
        return bodo.random_shuffle(data, seed=seed)

    try:
        check_func(impl, (data,), py_output=data, sort_output=False)
    except AssertionError:
        # this is correct, shuffled output should not match original data
        pass
    else:
        raise AssertionError

    check_func(impl, (data,), py_output=data, sort_output=True)

    if bodo.get_size() == 2:
        if bodo.get_rank() == 0:
            data_chunk = data[:70]
        else:
            data_chunk = data[70:]
        res = bodo.jit(distributed=["data"], all_returns_distributed=True)(impl)(
            data_chunk
        )
        # assert that data has been balanced across ranks
        assert len(res) == 50

        res = bodo.gatherv(res)
        if bodo.get_rank() == 0:
            try:
                _test_equal(res, data, sort_output=False)
            except AssertionError:
                # this is correct, shuffled output should not match original data
                pass
            else:
                raise AssertionError

            _test_equal(res, data, sort_output=True)


@pytest.mark.parametrize(
    "data", [pd.DataFrame({"A": range(10)}), np.arange(10), pd.Series(np.arange(10))]
)
def test_rebalance_group(data, memory_leak_check):
    """Test the bodo.rebalance(data, dests=[...]) functionality which gets
    data from all the ranks and distributes to only a given subset of ranks"""

    def impl0(data):  # this test is only for coverage purposes
        return bodo.rebalance(data, dests=[0])

    def impl1(data):
        return bodo.rebalance(data, dests=[0, 2])

    check_func(impl0, (data,), py_output=data)

    if bodo.get_size() == 3:  # run this only with 3 processes
        # give a different chunk size for each of the 3 processes
        if bodo.get_rank() == 0:
            data_chunk = data[0:2]
        elif bodo.get_rank() == 1:
            data_chunk = data[2:7]
        else:
            data_chunk = data[7:]
        # rebalance and send to process 0 and 2
        res = bodo.jit(distributed=["data"], all_returns_distributed=True)(impl1)(
            data_chunk
        )
        if bodo.get_rank() in {0, 2}:
            assert len(res) == 5
        else:
            assert len(res) == 0
        res = bodo.gatherv(res)
        if bodo.get_rank() == 0:
            if isinstance(data, pd.DataFrame):
                pd.testing.assert_frame_equal(data, res)
            else:
                np.testing.assert_array_equal(data, res)


def test_rebalance():
    """The bodo.rebalance function. It takes a dataframe which is unbalanced and
    returns a balanced one"""
    random.seed(5)
    # We create an unbalanced dataframe on input.
    rank = bodo.get_rank()
    n = 10 * (1 + rank)
    # The data from other nodes. It ends at prev_siz
    prev_siz = 10 * rank + 5 * rank * (rank - 1)
    # We need a nontrivial index for the run to be correct.
    elist = [4 + prev_siz + x for x in range(n)]
    flist = [prev_siz + x for x in range(n)]
    df_in = pd.DataFrame({"A": elist}, index=flist)
    df_in_merge = bodo.gatherv(df_in)
    # Direct calling the function
    df_out = bodo.libs.distributed_api.rebalance(df_in)
    df_out_merge = bodo.gatherv(df_out)
    pd.testing.assert_frame_equal(df_in_merge, df_out_merge)
    # The distributed case
    def f(df):
        return bodo.rebalance(df)

    bodo_dist = bodo.jit(all_args_distributed_block=True, all_returns_distributed=True)(
        f
    )
    df_out_dist = bodo_dist(df_in)
    df_out_dist_merge = bodo.gatherv(df_out_dist)
    df_len_dist = pd.DataFrame({"A": [len(df_out_dist)]})
    df_len_dist_merge = bodo.gatherv(df_len_dist)
    if bodo.get_rank() == 0:
        delta_size = df_len_dist_merge["A"].max() - df_len_dist_merge["A"].min()
        assert delta_size <= 1
    pd.testing.assert_frame_equal(df_in_merge, df_out_dist_merge)
    # The replicated case
    bodo_rep = bodo.jit(
        all_args_distributed_block=False, all_returns_distributed=False
    )(f)
    df_out_rep = bodo_rep(df_in_merge)
    pd.testing.assert_frame_equal(df_in_merge, df_out_rep)


def test_df_replace(memory_leak_check):
    # Implementation for single value and single value
    def impl1(df):
        return df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    # Implementation for list and single value
    def impl2(df):
        return df.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({"A": [1.0, 2.4, -np.inf], "B": [np.inf, np.nan, 5.2]})
    check_func(impl1, (df,))
    check_func(impl2, (df,))


@pytest.mark.slow
def test_box_df(memory_leak_check):
    """box dataframe contains column with name overlaps with pandas function"""

    def impl():
        df = pd.DataFrame({"product": ["a", "b", "c"], "keys": [1, 2, 3]})
        return df

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl(), check_dtype=False)


@pytest.mark.slow
def test_df_dtor(df_value, memory_leak_check):
    """make sure df destructor is working and there is no memory leak when columns are
    unboxed.
    """

    def impl(df):
        # len() forces unbox for a column to get its length
        return len(df)

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_index(df_value, memory_leak_check):
    def impl(df):
        return df.index

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_index_range_index(memory_leak_check):
    """test RangeIndex created inside the function"""

    def impl():
        df = pd.DataFrame({"A": [2, 3, 1]})
        return df.index

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


@pytest.mark.slow
def test_df_columns(df_value, memory_leak_check):
    def impl(df):
        return df.columns

    check_func(impl, (df_value,), is_out_distributed=False)


@pytest.mark.slow
def test_df_columns_nested(memory_leak_check):
    """make sure nested df column names can be returned properly"""

    def impl(df):
        df1 = df.groupby(["A"], as_index=False)
        df2 = df1.agg({"B": ["sum", "count"], "C": ["sum", "count"]})
        return df2.columns

    df = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1], "C": [2, 3, 1, 5]}
    )
    check_func(impl, (df,), is_out_distributed=False)


@pytest.mark.slow
def test_df_values(numeric_df_value, memory_leak_check):
    def impl(df):
        return df.values

    check_func(impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_values_nullable_int(memory_leak_check):
    def impl(df):
        return df.values

    # avoiding nullable integer column for Pandas test since the output becomes object
    # array with pd.NA object and comparison is not possible. Bodo may convert some int
    # columns to nullable somtimes when Pandas converts to float, so this matches actual
    # use cases.
    df = pd.DataFrame({"A": pd.array([3, 1, None, 4]), "B": [1.2, 3.0, -1.1, 2.0]})
    df2 = pd.DataFrame({"A": [3, 1, None, 4], "B": [1.2, 3.0, -1.1, 2.0]})
    bodo_out = bodo.jit(impl)(df)
    py_out = impl(df2)
    np.testing.assert_allclose(bodo_out, py_out)


@pytest.mark.slow
def test_df_to_numpy(numeric_df_value, memory_leak_check):
    def impl(df):
        return df.to_numpy()

    check_func(impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_ndim(df_value, memory_leak_check):
    def impl(df):
        return df.ndim

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_size(df_value, memory_leak_check):
    def impl(df):
        return df.size

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_shape(df_value, memory_leak_check):
    def impl(df):
        return df.shape

    check_func(impl, (df_value,))


# TODO: empty df: pd.DataFrame()
@pytest.mark.slow
@pytest.mark.parametrize("df", [pd.DataFrame({"A": [1, 3]}), pd.DataFrame({"A": []})])
def test_df_empty(df, memory_leak_check):
    def impl(df):
        return df.empty

    bodo_func = bodo.jit(impl)
    assert bodo_func(df) == impl(df)


def test_df_astype_num(numeric_df_value, memory_leak_check):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.astype(np.float32)

    check_func(impl, (numeric_df_value,))


def test_df_astype_str(numeric_df_value, memory_leak_check):
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


@pytest.mark.slow
def test_df_copy_deep(df_value, memory_leak_check):
    def impl(df):
        return df.copy()

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_copy_shallow(df_value, memory_leak_check):
    def impl(df):
        return df.copy(deep=False)

    check_func(impl, (df_value,))


def test_df_rename(memory_leak_check):
    def impl(df):
        return df.rename(columns={"B": "bb", "C": "cc"})

    def impl2(df):
        df.rename(columns={"B": "bb", "C": "cc"}, inplace=True)
        return df

    # raise error if const dict value is updated inplace
    def impl3(df):
        d = {"B": "bb", "C": "cc"}
        d.pop("C")
        return df.rename(columns=d)

    def impl4(df):
        d = {"B": "bb", "C": "cc"}
        d["C"] = "dd"
        return df.rename(columns=d)

    def impl5(df, a, b):
        df.rename(columns={"B": "bb", "C": "cc"}, inplace=(a > b))
        return df

    def impl6(df):
        p = True
        df.rename(columns={"B": "bb", "C": "cc"}, errors=1)
        return df

    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": [True, False, False, True, True],
        }
    )
    check_func(impl, (df,))
    check_func(impl2, (df,))
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable 'd' is updated inplace using 'pop'",
    ):
        bodo.jit(impl3)(df)
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable 'd' is updated inplace using 'setitem'",
    ):
        bodo.jit(impl4)(df)
    with pytest.raises(
        BodoError,
        match="'inplace' keyword only supports boolean constant assignment",
    ):
        bodo.jit(impl5)(df, 2, 3)
    with pytest.raises(
        BodoError,
        match="'error' keyword only supports default parameter values 'None' and 'ignore'",
    ):
        bodo.jit(impl6)(df)


@pytest.mark.smoke
def test_df_isna(df_value, memory_leak_check):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.isna()

    check_func(impl, (df_value,))


@pytest.mark.smoke
def test_df_notna(df_value, memory_leak_check):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.notna()

    check_func(impl, (df_value,))


def test_df_head(df_value, memory_leak_check):
    def impl(df):
        return df.head(3)

    check_func(impl, (df_value,), is_out_distributed=False)


def test_df_tail(df_value, memory_leak_check):
    def impl(df):
        return df.tail(3)

    check_func(impl, (df_value,), is_out_distributed=False)


@pytest.mark.parametrize(
    "other", [pd.DataFrame({"A": np.arange(5), "C": np.arange(5) ** 2}), [2, 3, 4, 5]]
)
def test_df_isin(other, memory_leak_check):
    # TODO: more tests, other data types
    # TODO: Series and dictionary values cases
    def impl(df, other):
        return df.isin(other)

    df = pd.DataFrame({"A": np.arange(5), "B": np.arange(5) ** 2})
    check_func(impl, (df, other))


def test_df_abs1(memory_leak_check):
    def impl(df):
        return df.abs()

    df = pd.DataFrame({"A": [1, 8, 4, 1, -2]}, range(0, 5, 1))
    check_func(impl, (df,))


@pytest.mark.slow
def test_df_abs2(numeric_df_value, memory_leak_check):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.abs()

    check_func(impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_corr(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # XXX pandas excludes bool columns with NAs, which we can't do dynamically
    for c in df_value.columns:
        if is_bool_object_series(df_value[c]) and df_value[c].hasnans:
            return

    def impl(df):
        return df.corr()

    check_func(impl, (df_value,), is_out_distributed=False)


def test_df_corr_parallel(memory_leak_check):
    def impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        return df.corr()

    bodo_func = bodo.jit(impl)
    n = 11
    pd.testing.assert_frame_equal(bodo_func(n), impl(n))
    assert count_array_OneDs() >= 3
    assert count_parfor_OneDs() >= 1


def test_df_cov(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # XXX pandas excludes bool columns with NAs, which we can't do dynamically
    for c in df_value.columns:
        if is_bool_object_series(df_value[c]) and df_value[c].hasnans:
            return

    def impl(df):
        return df.cov()

    check_func(impl, (df_value,), is_out_distributed=False)


def test_df_count(df_value, memory_leak_check):
    def impl(df):
        return df.count()

    check_func(impl, (df_value,), is_out_distributed=False)


def test_df_prod(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.prod()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (df_value,), is_out_distributed=False, check_dtype=False)


def test_df_sum(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.sum()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


def test_df_min(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.min()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


def test_df_max(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.max()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": np.arange(11, dtype=np.float64), "B": np.ones(11) + 4}),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3, 4, 5, 5, 5], "B": [1, 2, 3, 3, 4, 5, 10]}),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1.0, 2.0, 3.0, 4.0, None, 5.0, 6.0, None],
                    "B": [1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, None],
                }
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, 4, np.nan, 5, 6, None],
                    "B": [1, 2, None, 3, 4, 5, 6, np.nan],
                }
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3.0, 4, None, 5, 6, None],
                    "B": [1, 2, None, 3, 4, 5.0, 6, None],
                }
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 4, 3, 4, None, 5, 6, None, np.nan],
                    "B": [1, 2, None, 3, 4, 5, 6, None, 0],
                    "C": [1, 2, 5, 4, None, 5, 6, None, 1],
                }
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_df_reduce_axis1(df, memory_leak_check, is_slow_run):
    """test dataframe reductions across columns (axis=1)"""
    # TODO: support and test other reduce functions
    # TODO: Test with nullable ints

    def impl_max(df):
        return df.max(axis=1)

    def impl_min(df):
        return df.min(axis=1)

    def impl_sum(df):
        return df.sum(axis=1)

    def impl_prod(df):
        return df.prod(axis=1)

    def impl_mean(df):
        return df.mean(axis=1)

    def impl_median(df):
        return df.median(axis=1)

    def impl_var(df):
        return df.var(axis=1)

    def impl_std(df):
        return df.std(axis=1)

    check_func(impl_max, (df,))
    if not is_slow_run:
        return
    check_func(impl_min, (df,))
    check_func(impl_sum, (df,))
    check_func(impl_prod, (df,))
    check_func(impl_mean, (df,))
    check_func(impl_median, (df,))
    check_func(impl_var, (df,))
    check_func(impl_std, (df,))


@pytest.mark.smoke
def test_df_mean(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.mean()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


def test_df_var(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.var()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


def test_df_std(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.std()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


def test_df_median1(memory_leak_check):
    # remove this after NAs are properly handled
    def impl(df):
        return df.median()

    df = pd.DataFrame({"A": [1, 8, 4, 11, -3]})
    check_func(impl, (df,), is_out_distributed=False)


def test_df_median2(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    # skip NAs
    # TODO: handle NAs
    if numeric_df_value._get_numeric_data().isna().sum().sum():
        return

    def impl(df):
        return df.median()

    check_func(impl, (numeric_df_value,), is_out_distributed=False)


def test_df_quantile(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # pandas returns object Series for some reason when input has IntegerArray
    if isinstance(df_value.iloc[:, 0].dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def impl(df):
        return df.quantile(0.3)

    check_func(impl, (df_value,), is_out_distributed=False, check_names=False)


def test_df_pct_change(numeric_df_value, memory_leak_check):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def test_impl(df):
        return df.pct_change(2)

    check_func(test_impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_describe(numeric_df_value, memory_leak_check):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def test_impl(df):
        return df.describe()

    check_func(test_impl, (numeric_df_value,), is_out_distributed=False)


def test_df_cumprod(numeric_df_value, memory_leak_check):
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


def test_df_cumsum1(memory_leak_check):
    # remove this after NAs are properly handled
    def impl(df):
        return df.cumsum()

    df = pd.DataFrame({"A": [1, 8, 4, 11, -3]})
    check_func(impl, (df,))


def test_df_cumsum2(numeric_df_value, memory_leak_check):
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


# TODO: add memory_leak_check
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
    check_func(impl, (df_value,), is_out_distributed=False)


def _is_supported_argminmax_typ(d):
    # distributed argmax types, see distributed_api.py
    supported_typs = [np.int32, np.float32, np.float64]
    if not sys.platform.startswith("win"):
        # long is 4 byte on Windows
        supported_typs.append(np.int64)
        supported_typs.append(np.dtype("datetime64[ns]"))
    return d in supported_typs


def test_df_idxmax_datetime(memory_leak_check):
    def impl(df):
        return df.idxmax()

    df = pd.DataFrame(
        {"A": [3, 5, 1, -1, 2]},
        pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
    )
    check_func(impl, (df,), is_out_distributed=False)


def test_df_idxmax(numeric_df_value, memory_leak_check):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmax()

    check_func(impl, (numeric_df_value,), is_out_distributed=False)


def test_df_idxmin(numeric_df_value, memory_leak_check):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmin()

    check_func(impl, (numeric_df_value,), is_out_distributed=False)


def test_df_take(df_value, memory_leak_check):
    def impl(df):
        return df.take([1, 3])

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(df_value), impl(df_value), check_dtype=False
    )


# TODO: add memory_leak_check
def test_df_sort_index(df_value):
    # skip NAs
    # TODO: handle NA order
    if df_value.isna().sum().sum():
        return

    def impl(df):
        return df.sort_index()

    # TODO: use larger input to avoid empty object array in output
    check_func(impl, (df_value,), check_typing_issues=False)


def test_df_shift(numeric_df_value, memory_leak_check):
    # not supported for dt64
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.shift(2)

    check_func(impl, (numeric_df_value,))


def test_df_set_index(df_value, memory_leak_check):
    # singe column dfs become zero column which are not supported, TODO: fix
    if len(df_value.columns) < 2:
        return

    # TODO: fix nullable int
    if isinstance(df_value.iloc[:, 0].dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # TODO(ehsan): test non-str columns using 'df_value.columns[0]' instead of 'A" when
    # Numba can convert freevars to literals
    if "A" not in df_value.columns:
        return

    def impl(df):
        return df.set_index("A")

    check_func(impl, (df_value,))


def test_df_reset_index1(df_value, memory_leak_check):
    """Test DataFrame.reset_index(drop=False) on various dataframe/index combinations"""

    def impl(df):
        return df.reset_index()

    check_func(impl, (df_value,))


@pytest.mark.parametrize(
    "test_index",
    [
        # named numeric index
        pd.Int64Index([3, 1, 2, 4, 6], name="AA"),
        pytest.param(
            pd.UInt64Index([3, 1, 2, 4, 6], name="AA"), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.Float64Index([3.1, 1.2, 2.3, 4.4, 6.6], name="AA"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.RangeIndex(0, 5, name="AA"), marks=pytest.mark.slow
        ),  # TODO: Range(1, 6) when RangeIndex is fixed
        # named string index
        pytest.param(
            pd.Index(["A", "C", "D", "E", "AA"], name="ABC"), marks=pytest.mark.slow
        ),
        # named date/time index
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=5, name="ABC"),
            marks=pytest.mark.slow,
        ),
        # TODO: test PeriodIndex when PeriodArray is supported
        # pd.period_range(start='2017-01-01', end='2017-05-01', freq='M', name="ACD"),
        pytest.param(
            pd.timedelta_range(start="1D", end="5D", name="ABC"), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.MultiIndex.from_arrays(
                [
                    ["ABCD", "V", "CAD", "", "AA"],
                    [1.3, 4.1, 3.1, -1.1, -3.2],
                    pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
                ]
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.MultiIndex.from_arrays(
                [
                    ["ABCD", "V", "CAD", "", "AA"],
                    [1.3, 4.1, 3.1, -1.1, -3.2],
                    pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
                ],
                names=["AA", "ABC", "ABCD"],
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_df_reset_index2(test_index, memory_leak_check):
    """Test DataFrame.reset_index(drop=False) with MultiIndex and named indexes"""

    def impl(df):
        return df.reset_index()

    test_df = pd.DataFrame({"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]})
    test_df.index = test_index
    check_func(impl, (test_df,))


# TODO: add memory_leak_check when groupby leaks are resolved (#1472)
def test_df_reset_index3():
    """Test DataFrame.reset_index(drop=False) after groupby() which is a common pattern"""

    def impl1(df):
        return df.groupby("A").sum().reset_index()

    def impl2(df):
        return df.groupby(["A", "B"]).sum().reset_index()

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1],
            "B": [-8, 2, 3, 1, 5, 6, 7],
            "C": [3, 5, 6, 5, 4, 4, 3],
        }
    )
    check_func(impl1, (df,), sort_output=True, reset_index=True)
    check_func(impl2, (df,), sort_output=True, reset_index=True)


def test_df_reset_index4(memory_leak_check):
    """Test DataFrame.reset_index(drop=False, inplace=True)"""

    def impl(df):
        df.reset_index(drop=False, inplace=True)
        return df

    test_df = pd.DataFrame(
        {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]},
        [3.1, 1.2, 2.3, 4.4, 6.6],
    )
    check_func(impl, (test_df,), copy_input=True)


# TODO: add memory_leak_check
def test_df_duplicated():
    def impl(df):
        return df.duplicated()

    df = pd.DataFrame({"A": ["A", "B", "A", "B", "C"], "B": ["F", "E", "F", "S", "C"]})
    check_func(impl, (df,), sort_output=True)
    df = pd.DataFrame(
        {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]}, index=[3, 1, 2, 4, 6]
    )
    check_func(impl, (df,), sort_output=True)


##################### binary ops ###############################


@pytest.mark.smoke
def test_dataframe_binary_add(memory_leak_check):
    def test_impl(df, other):
        return df + other

    df = pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=[3, 5, 0, 7, 2])
    # df/df
    check_func(test_impl, (df, df))
    # df/scalar
    check_func(test_impl, (df, 2))
    check_func(test_impl, (2, df))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_binary_ops)
def test_dataframe_binary_op(op, memory_leak_check):
    # TODO: test parallelism
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
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


def test_dataframe_binary_iadd(memory_leak_check):
    def test_impl(df, other):
        df += other
        return df

    df = pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=[3, 5, 0, 7, 2])
    check_func(test_impl, (df, df), copy_input=True)
    check_func(test_impl, (df, 2), copy_input=True)


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_inplace_binary_ops)
def test_dataframe_inplace_binary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
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
def test_dataframe_unary_op(op, memory_leak_check):
    # TODO: fix operator.pos
    import operator

    if op == operator.pos:
        return

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
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
        pytest.param([2, 3, 5], marks=pytest.mark.slow),
        pytest.param([2.1, 3.2, np.nan, 5.4], marks=pytest.mark.slow),
        pytest.param(["A", "C", "AB"], marks=pytest.mark.slow),
        # int array, no NA sentinel value
        pytest.param(np.array([2, 3, 5, -1, -4, 9]), marks=pytest.mark.slow),
        # float array with np.nan
        pytest.param(np.array([2.9, np.nan, 1.4, -1.1, -4.2]), marks=pytest.mark.slow),
        pd.Series([2.1, 5.3, np.nan, -1.0, -3.7], [3, 5, 6, -2, 4], name="C"),
        pytest.param(
            pd.Int64Index([10, 12, 14, 17, 19], name="A"), marks=pytest.mark.slow
        ),
        pytest.param(pd.RangeIndex(5), marks=pytest.mark.slow),
        # dataframe
        pd.DataFrame(
            {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
            [1.1, -2.1, 7.1, 0.1, 3.1],
        ),
        # scalars
        3,
        pytest.param(1.3, marks=pytest.mark.slow),
        np.nan,
        "ABC",
        None,
        np.datetime64("NaT"),
        pytest.param(np.timedelta64("NaT"), marks=pytest.mark.slow),
    ]
)
def na_test_obj(request):
    return request.param


def test_pd_isna(na_test_obj, memory_leak_check):
    obj = na_test_obj

    def impl(obj):
        return pd.isna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


@pytest.mark.slow
def test_pd_notna(na_test_obj, memory_leak_check):
    obj = na_test_obj

    def impl(obj):
        return pd.notna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


def test_pd_isna_getitem(memory_leak_check):
    """test support for NA check for array values, e.g. pd.isna(A[i]) pattern matching
    in SeriesPass
    """

    def impl1(df):
        s = 0
        for i in bodo.prange(len(df)):
            l = 0
            if pd.isna(df.iloc[i, 0]):
                l = 10
            else:
                l = len(df.iloc[i, 0])
            s += l
        return s

    def impl2(S, i):
        return pd.notna(S.iloc[i])

    def impl3(A, i):
        return pd.isnull(A[i])

    df = pd.DataFrame(
        {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
        [1.1, -2.1, 7.1, 0.1, 3.1],
    )
    check_func(impl1, (df,))
    S = pd.Series([2.1, 5.3, np.nan, -1.0, -3.7], [3, 5, 6, -2, 4], name="C")
    assert bodo.jit(impl2)(S, 0) == impl2(S, 0)
    assert bodo.jit(impl2)(S, 2) == impl2(S, 2)
    A = np.array([1.3, 2.2, np.nan, 3.1, np.nan, -1.1])
    assert bodo.jit(impl3)(A, 0) == impl3(A, 0)
    assert bodo.jit(impl3)(A, 2) == impl3(A, 2)


def test_setitem_na(memory_leak_check):
    """test support for setting NA value to array location, e.g. A[i] = None"""

    def impl(S, i):
        S.iloc[i] = None
        return S

    S = pd.Series(["AA", np.nan, "", "D", "GG"], name="C")
    # TODO: support distributed setitem with scalar
    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(bodo_func(S.copy(), 0), impl(S.copy(), 0))
    pd.testing.assert_series_equal(bodo_func(S.copy(), 1), impl(S.copy(), 1))
    pd.testing.assert_series_equal(bodo_func(S.copy(), 2), impl(S.copy(), 2))


def test_set_column_scalar_str(memory_leak_check):
    """set df column with a string scalar"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = "AA"
        return df

    # test unicode characters
    def test_impl2(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = "😀🐍,⚡ 오늘 ، هذاú,úũ"
        return df

    n = 11
    check_func(test_impl, (n,))
    check_func(test_impl2, (n,))


def test_set_column_scalar_num(memory_leak_check):
    """set df column with a numeric scalar"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = 3
        return df

    n = 11
    check_func(test_impl, (n,))


def test_set_column_scalar_timestamp(memory_leak_check):
    """set df column with a timestamp scalar"""

    def test_impl(n, t):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df["C"] = t
        return df

    n = 11
    t = pd.Timestamp("1994-11-23T10:11:35")
    check_func(test_impl, (n, t))


def test_set_column_cond1(memory_leak_check):
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


def test_set_column_cond2(memory_leak_check):
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


def test_set_column_cond3(memory_leak_check):
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


def test_set_column_setattr(memory_leak_check):
    """set df column using setattr instead of setitem"""

    # same type as existing column
    def impl1(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df.B = 2
        return df

    # change column type
    def impl2(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        df.B = "AA"
        return df

    n = 11
    check_func(impl1, (n,))
    check_func(impl2, (n,))


def test_set_column_reflect_error(memory_leak_check):
    """set column of dataframe argument that is passed from another JIT function, so it
    doesn't have a parent dataframe object (even though it is an argument).
    See set_df_column_with_reflect()
    """

    @bodo.jit
    def f(data):
        data["B"] = data["A"].str.len()
        return data

    def impl():
        df = pd.DataFrame({"A": ["BB", "CCB", "DDD", "A"]})
        df = f(df)
        return df

    pd.testing.assert_frame_equal(bodo.jit(impl)(), impl())


def test_df_filter(memory_leak_check):
    def test_impl(df, cond):
        df2 = df[cond]
        return df2

    def test_impl2(df, cond):
        # using .values to test nullable boolean array
        df2 = df[cond.values]
        return df2

    df = pd.DataFrame(
        {
            "A": [2, 1, 1, 1, 2, 2, 1] * 2,
            "B": ["A", "B", np.nan, "ACDE", "C", np.nan, "AA"] * 2,
            "C": [2, 3, -1, 1, np.nan, 3.1, -1] * 2,
        }
    )
    cond = df.A > 1
    check_func(test_impl, (df, cond))
    check_func(test_impl2, (df, cond))


def test_create_series_input1(memory_leak_check):
    def test_impl(S):
        df = pd.DataFrame({"A": S})
        return df

    bodo_func = bodo.jit(test_impl)
    S = pd.Series([2, 4], [3, -1])
    pd.testing.assert_frame_equal(bodo_func(S), test_impl(S))


def test_df_apply_getitem(memory_leak_check):
    """test getitem access of row value passed in df.apply()"""

    def test_impl(df):
        return df.apply(lambda r: r["B"] if r["A"] == "AA" else 0, axis=1)

    df = pd.DataFrame(
        {"A": ["AA", "B", "CC", "C", "AA"], "B": [3, 1, 2, 5, 9]}, index=[3, 1, 4, 6, 0]
    )
    check_func(test_impl, (df,))


def test_df_apply_int_getitem_unsorted_columns(memory_leak_check):
    """
    test int getitem access of row passed in df.apply() where column names are not in
    sorted order (issue #2019)
    """

    def impl(df):
        return df.apply(lambda x: (x[0], x[2], x[1]), axis=1)

    df = pd.DataFrame(
        {"A": np.arange(10), "C": np.arange(10, 20), "B": np.arange(20, 30)}
    )
    check_func(impl, (df,))


def test_df_apply_bool(memory_leak_check):
    # check bool output of UDF for BooleanArray use
    def test_impl(df):
        return df.apply(lambda r: r.A == 2, axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df,))


def test_df_apply_str(memory_leak_check):
    """make sure string output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: r.A if r.A == "AA" else "BB", axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_list_str(memory_leak_check):
    """make sure list(str) output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: [r.A] if r.A == "AA" else ["BB", r.A], axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_array_item(memory_leak_check):
    """make sure array(item) output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: [len(r.A)] if r.A == "AA" else [3, len(r.A)], axis=1)

    df = pd.DataFrame({"A": ["AA", "B", "CC", "C", "AA"]}, index=[3, 1, 4, 6, 0])
    check_func(test_impl, (df,))


def test_df_apply_date(memory_leak_check):
    """make sure datetime.date output can be handled in apply() properly"""

    def test_impl(df):
        return df.apply(lambda r: r.A.date(), axis=1)

    df = pd.DataFrame(
        {"A": pd.date_range(start="2018-04-24", end="2019-04-29", periods=5)}
    )
    check_func(test_impl, (df,))


def test_df_apply_timestamp(memory_leak_check):
    """make sure Timestamp (converted to datetime64) output can be handled in apply()
    properly
    """

    def test_impl(df):
        return df.apply(lambda r: r.A + datetime.timedelta(days=1), axis=1)

    df = pd.DataFrame(
        {"A": pd.date_range(start="2018-04-24", end="2019-04-29", periods=5)}
    )
    check_func(test_impl, (df,))


def test_df_apply_general_colnames(memory_leak_check):
    """make sure all column names (e.g. not string, not identifier-compatible string) can be handled in apply() properly"""

    def impl1(df):
        return df.apply(lambda r: r["C C"], axis=1)

    def impl2(df):
        return df.apply(lambda r: r[2], axis=1)

    def impl3(df):
        return df.apply(lambda r: r.A, axis=1)

    df = pd.DataFrame(
        {
            "A": ["AA", "B", "CC", "C", "AA"],
            2: [3, 1, 4, 2, 6],
            "C C": [1.1, 2.2, 3.3, 4.4, 5.5],
        },
        index=[3, 1, 4, 6, 0],
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.slow
def test_df_apply_decimal(memory_leak_check):
    """make sure Decimal output can be handled in apply() properly"""
    # just returning input value since we don't support any Decimal creation yet
    # TODO: support Decimal(str) constructor
    # TODO: fix using freevar constants in UDFs
    def test_impl(df):
        return df.apply(lambda r: r.A, axis=1)

    df = pd.DataFrame(
        {
            "A": [
                Decimal("1.6"),
                Decimal("-0.222"),
                Decimal("1111.316"),
                Decimal("1234.00046"),
                Decimal("5.1"),
                Decimal("-11131.0056"),
                Decimal("0.0"),
            ]
        }
    )
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_args(memory_leak_check):
    """test passing extra args to apply UDF"""

    def test_impl(df, b):
        return df.apply(lambda r, a: r.A == a, axis=1, args=(b,))

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df, 3))


def test_df_apply_kws(memory_leak_check):
    """test passing extra keyword args to apply UDF"""

    # only kw args
    def impl1(df, b):
        return df.apply(lambda r, c=1, a=2: r.A == a + c, a=b, axis=1)

    # both positional and kw args
    def impl2(df, b, d):
        return df.apply(lambda r, c=1, a=2: r.A == a + c, a=b, axis=1, args=(d,))

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(impl1, (df, 3))
    check_func(impl2, (df, 3, 2))


def g(r):
    return 2 * r.A


def test_df_apply_func_case1(memory_leak_check):
    """make sure a global function can be used in df.apply"""

    def test_impl(df):
        return df.apply(g, axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(test_impl, (df,))


@bodo.jit
def g2(r):
    return 2 * r[0]


@pytest.mark.slow
def test_df_apply_func_case2(memory_leak_check):
    """make sure a UDF calling another function doesn't fail (#964)"""

    def test_impl(df):
        return df.apply(lambda x: g2(x), axis=1)

    n = 121
    df = pd.DataFrame({"A": np.arange(n)})
    # NOTE: not using check_func since regular Pandas calling g2 can cause hangs due to
    # barriers generated by Bodo
    res = bodo.jit(
        test_impl, all_args_distributed_block=True, all_returns_distributed=True
    )(_get_dist_arg(df, False))
    res = bodo.allgatherv(res)
    py_res = df.apply(lambda r: 2 * r[0], axis=1)
    pd.testing.assert_series_equal(res, py_res)


def test_df_apply_error_check():
    """make sure a proper error is raised when UDF is not supported (not compilable)"""

    def test_impl(df):
        # some UDF that cannot be supported, lambda calling a non-jit function
        return df.apply(lambda r: g(r), axis=1)

    df = pd.DataFrame({"A": np.arange(11)})
    with pytest.raises(
        BodoError, match="DataFrame.apply.*: user-defined function not supported"
    ):
        bodo.jit(test_impl)(df)


@pytest.mark.slow
def test_df_apply_df_output(memory_leak_check):
    """test DataFrame.apply() with dataframe output 1 column"""

    def impl1(df):
        return df.apply(lambda a: pd.Series([a[0], "AA"]), axis=1)

    def impl2(df):
        def g(a):
            # TODO: support assert in UDFs properly
            # assert a > 0.0
            if a[0] > 3:
                return pd.Series([a[0], 2 * a[0]], ["A", "B"])
            return pd.Series([a[0], 3 * a[0]], ["A", "B"])

        return df.apply(g, axis=1)

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    check_func(impl1, (df,))
    check_func(impl2, (df,))


def test_df_apply_df_output_multicolumn(memory_leak_check):
    """test DataFrame.apply() with dataframe output with multiple columns"""

    def test_impl(df):
        return df.apply(lambda a: pd.Series([a[0], a[1]]), axis=1)

    df = pd.DataFrame({"A": np.arange(20), "B": ["hi", "there"] * 10})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_df_apply_df_output_multistring(memory_leak_check):
    def test_impl(df):
        def f(row):
            s1 = ""
            s2 = ""
            s3 = ""
            s4 = ""
            if row[1] == 0:
                s3 = str(row[0]) + ","
                s4 = str(row[2]) + ","
            elif row[1] == 1:
                s1 = str(row[0]) + ","
                s2 = str(row[2]) + ","
            return pd.Series([s1, s2, s3, s4], index=["s1", "s2", "s3", "s4"])

        return df.apply(f, axis=1)

    df = pd.DataFrame({"A": np.arange(40), "B": [0, 1] * 20, "C": np.arange(40)})
    check_func(test_impl, (df,))


def test_df_drop_inplace_branch(memory_leak_check):
    def test_impl(cond):
        if cond:
            df = pd.DataFrame({"A": [2, 3, 4], "B": [1, 2, 6]})
        else:
            df = pd.DataFrame({"A": [5, 6, 7], "B": [1, 0, -6]})
        df.drop("B", axis=1, inplace=True)
        return df

    check_func(test_impl, (True,), False)


# TODO: add memory_leak_check when join memory leaks are fixed
@pytest.mark.slow
def test_df_filter_rm_index():
    """
    Make sure dataframe index is removed correctly and parallelism warning is thrown
    when a dataframe is filtered after a join.
    """

    def impl(df1, df2):
        df3 = df1.merge(df2, on="A")
        return df3[df3.A > 3]

    df1 = pd.DataFrame({"A": [2, 3, 4], "B": [1, 2, 6]})
    df2 = pd.DataFrame({"A": [3, 4, 1]})
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(df1, df2)
    else:
        bodo.jit(impl)(df1, df2)


def test_concat_df_columns(memory_leak_check):
    """Test dataframe concatenation with axis=1 (add new columns)"""

    def test_impl(df, df2):
        return pd.concat([df, df2], axis=1)

    df = pd.DataFrame({"A": [1, 2, 3, 9, 11]})
    df2 = pd.DataFrame({"B": [4.0, 5.0, 4.1, 6.2, 2.1], "C": [7, 1, 3, -4, -1]})
    check_func(test_impl, (df, df2))


def test_concat_typing_transform(memory_leak_check):
    """Test list to tuple trasnform in typing pass, when other typing related changes
    are also required.
    """

    def test_impl(df, df2):
        df3 = pd.concat([df, df2], axis=1)
        df3["D"] = 3
        return df3

    df = pd.DataFrame({"A": [1, 2, 3, 9, 11]})
    df2 = pd.DataFrame({"B": [4.0, 5.0, 4.1, 6.2, 2.1], "C": [7, 1, 3, -4, -1]})
    check_func(test_impl, (df, df2))


def test_concat_int_float(memory_leak_check):
    """Test dataframe concatenation when integer and float are put together"""

    def test_impl(df, df2):
        return df.append(df2, ignore_index=True)

    def test_impl_concat(df, df2):
        return pd.concat((df, df2), ignore_index=True)

    df = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [4.0, 5.0]})
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)
    check_func(test_impl_concat, (df, df2), sort_output=True, reset_index=True)


def test_concat_nulls(memory_leak_check):
    """Test dataframe concatenation when full NA arrays need to be appended"""

    def test_impl(df, df2):
        return df.append(df2, ignore_index=True)

    def test_impl_concat(df, df2):
        return pd.concat((df, df2), ignore_index=True)

    n = 5
    df = pd.DataFrame(
        {
            "A": ["ABC", None, "AA", "B", None, "AA"],
            "D": pd.date_range(start="2017-01-12", periods=6),
        }
    )
    df2 = pd.DataFrame(
        {
            "B": np.arange(n),
            "C": np.ones(n),
            "E": pd.timedelta_range(start=3, periods=n),
        }
    )
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)
    check_func(test_impl_concat, (df, df2), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "df",
    [
        # RangeIndex and numeric types
        pd.DataFrame(
            {
                "B": np.arange(11),
                "C": np.ones(11),
                "E": pd.timedelta_range(start=3, periods=11),
            },
        ),
        # variable item size data and index
        pd.DataFrame(
            {
                "A": ["ABC", None, "AA", "B", None, "AA", "CC", "G"],
            },
            index=["AA", "C", "BB", "A", "D", "L", "K", "P"],
        ),
    ],
)
def test_append_empty_df(df):
    """Test appending to an empty dataframe in a loop (common pattern)"""
    # TODO: fix casting refcount in Numba since Numba increfs value after cast

    def test_impl(df2):
        df = pd.DataFrame()
        for _ in range(3):
            df = df.append(df2)
        return df

    check_func(test_impl, (df,), sort_output=True, reset_index=True, check_dtype=False)


def test_init_dataframe_array_analysis():
    """make sure shape equivalence for init_dataframe() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
        return df

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(10)
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("n")


def test_get_dataframe_data_array_analysis():
    """make sure shape equivalence for get_dataframe_data() is applied correctly"""
    import numba.tests.test_array_analysis

    def impl(df):
        B = df.A.values
        return B

    test_func = numba.njit(pipeline_class=AnalysisTestPipeline, parallel=True)(impl)
    test_func(pd.DataFrame({"A": np.ones(10), "B": np.arange(10)}))
    array_analysis = test_func.overloads[test_func.signatures[0]].metadata[
        "preserved_array_analysis"
    ]
    eq_set = array_analysis.equiv_sets[0]
    assert eq_set._get_ind("df#0") == eq_set._get_ind("B#0")


def test_df_const_set_rm_index(memory_leak_check):
    """Make sure dataframe related variables like the index are removed correctly and
    parallelism warning is thrown when a column is being set using a constant.
    Test for a bug that was keeping RangeIndex around as a 1D so warning wasn't thrown.
    """

    def impl(A):
        df = pd.DataFrame({"A": A})
        df["B"] = 1
        return df.A.values

    A = np.arange(10)
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(A)
    else:
        bodo.jit(impl)(A)


def test_df_dropna(memory_leak_check):
    """Test df.dropna() with various data types and arguments"""

    def impl1(df):
        return df.dropna(subset=["A", "B"])

    def impl2(df):
        return df.dropna(thresh=2)

    def impl3(df):
        return df.dropna(how="all")

    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 1.0] * 3,
            "B": [4, 5, 6, np.nan] * 3,
            "C": [np.nan, "AA", np.nan, "ABC"] * 3,
            "D": [[1, 2], None, [1], []] * 3,
        }
    )
    # TODO: fix 1D_Var RangeIndex
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))


def test_df_dropna_inplace_check():
    """make sure inplace=True is not used in df.dropna()"""

    def test_impl(df):
        df.dropna(inplace=True)

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
    with pytest.raises(BodoError, match="inplace=True is not supported"):
        bodo.jit(test_impl)(df)


def test_df_drop_inplace_instability_check():
    """make sure df.drop(inplace=True) doesn't cause type instability"""

    def test_impl(a):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
        if len(a) > 3:
            df.drop("B", 1, inplace=True)
        return df

    with pytest.raises(BodoError, match="inplace change of dataframe schema"):
        bodo.jit(test_impl)([2, 3])


# TODO: fix casting refcount in Numba since Numba increfs value after cast
def test_df_range_index_unify():
    """Test dataframe type unification when RangeIndex should be converted to
    IntegerIndex
    """

    def test_impl(df, df2):
        if len(df2) > 0:
            df = df2
        return df

    df = pd.DataFrame({"A": [1, -2, 3, 5, 11, 4, -1]})
    df2 = pd.DataFrame(
        {"A": [0, 4, 11, 2, -2, -3, 9]}, index=[11, -20, 31, 52, 1, 41, -11]
    )
    check_func(test_impl, (df, df2), sort_output=True, reset_index=True)


################################## indexing  #################################


@pytest.mark.smoke
def test_column_list_getitem1(memory_leak_check):
    """Test df[["A", "B"]] getitem case"""

    def test_impl(df):
        return df[["A", "C", "B"]]

    df = pd.DataFrame(
        {
            "A": [1.1, 2.3, np.nan, 1.7, 3.6] * 2,
            "A2": [3, 1, 2, 3, 5] * 2,
            "B": [True, False, None, False, True] * 2,
            "C": ["AA", "C", None, "ABC", ""] * 2,
        },
        index=[3, 1, 2, 4, 0] * 2,
    )
    check_func(test_impl, (df,))


def test_column_list_getitem_infer(memory_leak_check):
    """Test df[["A", "B"]] getitem case when column names list has to be inferred in
    partial typing.
    """

    def test_impl(df):
        return df[["A"] + ["C", "B"]]

    df = pd.DataFrame(
        {
            "A": [1.1, 2.3, np.nan, 1.7, 3.6] * 2,
            "A2": [3, 1, 2, 3, 5] * 2,
            "B": [True, False, None, False, True] * 2,
            "C": ["AA", "C", None, "ABC", ""] * 2,
        },
        index=[3, 1, 2, 4, 0] * 2,
    )
    check_func(test_impl, (df,))


def test_iloc_bool_arr(memory_leak_check):
    """test df.iloc[bool_arr]"""

    def test_impl(df):
        return df.iloc[(df.A > 3).values]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_iloc_slice(memory_leak_check):
    def test_impl(df, n):
        return df.iloc[1:n]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    bodo_func = bodo.jit(test_impl)
    # TODO: proper distributed support for slicing
    pd.testing.assert_frame_equal(bodo_func(df, n), test_impl(df, n))


def test_iloc_slice_col_ind(memory_leak_check):
    """test df.iloc[slice, col_ind]"""

    def test_impl(df):
        return df.iloc[:, 1].values

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_iloc_slice_col_slice(memory_leak_check):
    """test df.iloc[slice, slice] which selects a set of columns"""

    def test_impl1(df):
        return df.iloc[:, 1:]

    def test_impl2(df):
        return df.iloc[:, 1:3]

    def test_impl3(df):
        return df.iloc[:, :-1]

    def test_impl4(df):
        return df.iloc[1:, 1:]

    def test_impl5(df):
        return df.iloc[:, :]

    def test_impl6(df, n):
        return df.iloc[:, 1:n]

    def test_impl7(df):
        return df.iloc[:, 0:3:2]

    n = 11
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.arange(n) ** 2 + 1.0,
            "C": np.arange(n) + 2.0,
            "D": np.arange(n) + 3,
        }
    )
    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))
    check_func(test_impl3, (df,))
    check_func(test_impl4, (df,))
    check_func(test_impl5, (df,))
    # error checking for when slice is not constant
    with pytest.raises(BodoError, match=r"df.iloc\[slice1,slice2\] should be constant"):
        bodo.jit(test_impl6)(df, 3)
    check_func(test_impl7, (df,))


def test_iloc_int_col_ind(memory_leak_check):
    """test df.iloc[int, col_ind]"""

    def test_impl(df):
        return df.iloc[3, 1]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_loc_bool_arr(memory_leak_check):
    """test df.loc[bool_arr]"""

    def test_impl(df):
        return df.loc[(df.A > 3).values]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_loc_col_name(memory_leak_check):
    """test df.loc[slice, col_ind]"""

    def test_impl(df):
        return df.loc[(df.A > 3).values, "B"].values

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df,))


def test_loc_range_index(memory_leak_check):
    """test df.loc[int, col_ind] for RangeIndex"""

    def test_impl(df, i):
        return df.loc[i, "B"]

    n = 11
    i = 4
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
    check_func(test_impl, (df, i))


def test_loc_range_index_prange(memory_leak_check):
    """test df.loc[int, col_ind] for RangeIndex in a parallel loop"""

    def impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        s = 0
        for i in bodo.prange(len(df)):
            s += df.loc[i, "B"]
        return s

    n = 11
    check_func(impl, (n,))


def test_loc_col_select(memory_leak_check):
    """test df.iloc[slice, col_ind] where col_ind is a list of column names or bools"""

    def impl1(df):
        return df.loc[:, ["A", "C"]]

    def impl2(df):
        return df.loc[:, [True, False, True]]

    def impl3(df):
        return df.loc[:, df.columns != "B"]

    def impl4(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        df.columns = ["AB", "CD", "EF"]
        return df.loc[:, ["AB", "EF"]]

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    check_func(impl3, (df,))
    check_func(impl4, (n,))


def test_loc_setitem(memory_leak_check):
    """test df.iloc[idx, col_ind]  setitem where col_ind is a list of column names or bools"""

    # set existing column with full slice
    def impl1(df):
        df.loc[:, "B"] = 11
        return df

    # set new columns with full slice
    def impl2(df):
        df.loc[:, ["D", "E"]] = 11
        return df

    # set values with bool index
    def impl3(df):
        df.loc[df.A > 4, "B"] = 11
        return df

    # boolean column selection
    def impl4(df):
        df.loc[:, [True, False, True]] = 11
        return df

    # dynamic column selection
    def impl5(df):
        df.loc[:, df.columns != "B"] = 11
        return df

    # schema change
    def impl6(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        df.columns = ["AB", "CD", "EF"]
        df.loc[:, ["AB", "EF"]] = 11
        return df

    # boolean column selection
    def impl7(df):
        df.loc[df.A > 4, [True, False, True]] = 11
        return df

    # dynamic column selection
    def impl8(df):
        df.loc[df.A > 4, df.columns != "B"] = 11
        return df

    # schema change
    def impl9(n):
        df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n), "C": np.ones(n)})
        cond = df.A > 4
        df.columns = ["AB", "CD", "EF"]
        df.loc[cond, ["AB", "EF"]] = 11
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)
    check_func(impl3, (df,), copy_input=True)
    check_func(impl4, (df,), copy_input=True)
    check_func(impl5, (df,), copy_input=True)
    check_func(impl6, (n,))
    check_func(impl7, (df,), copy_input=True)
    check_func(impl8, (df,), copy_input=True)
    check_func(impl9, (n,))


@pytest.mark.smoke
def test_iat_setitem():
    """test df.iat[] setitem (single value)"""

    def impl(df, n):
        df.iat[n - 1, 1] = n ** 2
        return df

    n = 11
    df = pd.DataFrame({"B": np.ones(n), "A": np.arange(n) + n})
    check_func(impl, (df, n), copy_input=True)


def test_df_schema_change(memory_leak_check):
    """
    Dataframe operations like setting new columns change the schema, so other
    operations need to handle type change during typing pass.

    df.drop() checks for drop columns to be in the schema, so it has to let typing pass
    change the type. This example is from the forecast code.
    """

    def test_impl(df):
        df["C"] = 3
        return df.drop(["C"], 1)

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))


def test_set_df_column_names(memory_leak_check):
    """test setting dataframe column names using df.columns"""

    def impl1(df):
        df.columns = ["a", "b"]
        return df

    # invalid length
    def impl2(df):
        df.columns = ["a", "b", "c"]
        return df

    # type instability due to control flow
    def impl3(df, flag):
        if flag:
            df.columns = ["a", "b"]
        return df

    # non-constant column names
    def impl4(df, a):
        df.columns = a[1:]
        return df

    # test setattr on df with nested names (#2126)
    def impl5(df):
        df1 = df.groupby(["A"], as_index=False)
        df2 = df1.agg({"B": ["sum", "count"], "C": ["sum", "count"]})
        df2.columns = ["A", "testCol1", "count(B)", "testCol2", "count(C)"]
        return df2

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    check_func(impl1, (df,), copy_input=True)
    with pytest.raises(
        BodoError,
        match="DataFrame.columns: number of new column names does not match number of existing columns",
    ):
        bodo.jit(impl2)(df)
    with pytest.raises(
        BodoError,
        match="DataFrame.columns: setting dataframe column names inside conditionals and loops not supported yet",
    ):
        bodo.jit(impl3)(df, False)
    with pytest.raises(
        BodoError, match="DataFrame.columns: new column names should be a constant list"
    ):
        bodo.jit(impl4)(df, ["a", "b", "c"])
    df = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1], "C": [2, 3, 1, 5]}
    )
    check_func(impl5, (df,))


def test_set_df_index(memory_leak_check):
    """test setting dataframe index using df.index"""

    def impl1(df):
        df.index = ["AA", "BB", "CC", "DD"]
        return df

    def impl2(df):
        df.index = pd.Int64Index([3, 1, 2, 0])
        return df

    # type instability due to control flow
    def impl3(df, flag):
        if flag:
            df.index = ["AA", "BB", "CC", "DD"]
        return df

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    check_func(impl1, (df,), copy_input=True, dist_test=False)
    check_func(impl2, (df,), copy_input=True, dist_test=False)
    with pytest.raises(
        BodoError,
        match="DataFrame.index: setting dataframe index inside conditionals and loops not supported yet",
    ):
        bodo.jit(impl3)(df, True)


def test_df_multi_schema_change(memory_leak_check):
    """Test multiple df schema changes while also calling other Bodo functions.
    Makes sure global state variables in typing pass are saved properly and are not
    disrupted by calling another Bodo function (which calls the compiler recursively)
    """

    @bodo.jit
    def g(df):
        return df.assign(D=np.ones(len(df)))

    # inspired by user code (PO reconciliation project)
    def test_impl(df):
        df["C"] = 3
        df_cols = list(df.columns)
        df = g(df)
        df_cols = df_cols + ["D"]
        return df[df_cols]

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))


def test_df_drop_column_check(memory_leak_check):
    def test_impl(df):
        return df.drop(columns=["C"])

    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [4, 5, 6, 7]})
    with pytest.raises(BodoError, match="not in DataFrame columns"):
        bodo.jit(test_impl)(df)


def test_df_fillna_str_inplace(memory_leak_check):
    """Make sure inplace fillna for string columns is reflected in output"""

    def test_impl(df):
        df.B.fillna("ABC", inplace=True)
        return df

    df_str = pd.DataFrame(
        {"A": [2, 1, 1, 1, 2, 2, 1], "B": ["ab", "b", np.nan, "c", "bdd", "c", "a"]}
    )
    check_func(test_impl, (df_str,), copy_input=True)


def test_df_alias(memory_leak_check):
    """Test alias analysis for df data arrays. Without proper alias info, the fillna
    changes in data array will be optimized away incorrectly.
    This example is from the forecast code.
    """

    def test_impl():
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0], "B": [1.2, np.nan, 1.1, 3.1]})
        df.B.fillna(1, inplace=True)
        return df

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(), test_impl())


def test_df_type_unify_error():
    """makes sure type unification error is thrown when two dataframe schemas for the
    same variable are not compatible.
    """

    def test_impl(a):
        if len(a) > 3:  # some computation that cannot be inferred statically
            df = pd.DataFrame({"A": [1, 2, 3]})
        else:
            df = pd.DataFrame({"A": ["a", "b", "c"]})
        return df

    with pytest.raises(numba.TypingError, match="Cannot unify dataframe"):
        bodo.jit(test_impl)([3, 4])


# TODO: fix memory leak and add memory_leak_check
@pytest.mark.slow
def test_dataframe_constant_lowering():
    df = pd.DataFrame({"A": [2, 1], "B": [1.2, 3.3]})

    def impl():
        return df

    pd.testing.assert_frame_equal(bodo.jit(impl)(), df)


def test_dataframe_columns_const_passing(memory_leak_check):
    """Test passing df.columns as a constant to another call"""

    def impl(df):
        return df.dropna(subset=df.columns, how="any")

    df = pd.DataFrame({"A": [2, 1], "B": [1.2, 3.3]})
    check_func(impl, (df,))


def test_dataframe_sample_number(memory_leak_check):
    """Checking the random routine is especially difficult to do.
    We can mostly only check incidental information about the code"""

    def f(df):
        return df.sample(n=4, replace=False).size

    bodo_f = bodo.jit(all_args_distributed_block=True, all_returns_distributed=False)(f)
    n = 10
    df = pd.DataFrame({"A": [x for x in range(n)]})
    py_output = f(df)
    df_loc = _get_dist_arg(df)
    assert bodo_f(df_loc) == py_output


@pytest.mark.slow
def test_dataframe_sample_uniform(memory_leak_check):
    """Checking the random routine, this time with uniform input"""

    def f1(df):
        return df.sample(n=4, replace=False)

    def f2(df):
        return df.sample(frac=0.5, replace=False)

    n = 10
    df = pd.DataFrame({"A": [1 for _ in range(n)]})
    check_func(f1, (df,), reset_index=True, is_out_distributed=False)
    check_func(f2, (df,), reset_index=True, is_out_distributed=False)


@pytest.mark.slow
def test_dataframe_sample_sorted(memory_leak_check):
    """Checking the random routine. Since we use sorted and the number of entries is equal to
    the number of sampled rows, after sorting the output becomes deterministic, that is independent
    of the random number generated"""

    def f(df, n):
        return df.sample(n=n, replace=False)

    n = 10
    df = pd.DataFrame({"A": [x for x in range(n)]})
    check_func(f, (df, n), reset_index=True, sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_dataframe_sample_index(memory_leak_check):
    """Checking that the index passed coherently to the A entry."""

    def f(df):
        return df.sample(5)

    df = pd.DataFrame({"A": list(range(20))})
    bodo_f = bodo.jit(all_args_distributed_block=False, all_returns_distributed=False)(
        f
    )
    df_ret = bodo_f(df)
    S = df_ret.index == df_ret["A"]
    assert S.all()


# TODO: fix leak and add memory_leak_check
@pytest.mark.slow
def test_dataframe_sample_nested_datastructures():
    """The sample function relies on allgather operations that deserve to be tested"""

    def check_gather_operation(df):
        siz = df.size

        def f(df, m):
            return df.sample(n=m, replace=False).size

        py_output = f(df, siz)
        start, end = get_start_end(len(df))
        df_loc = df.iloc[start:end]
        bodo_f = bodo.jit(
            all_args_distributed_block=True, all_returns_distributed=False
        )(f)
        df_ret = bodo_f(df_loc, siz)
        assert df_ret == py_output

    n = 10
    random.seed(1)
    df1 = pd.DataFrame({"B": gen_random_arrow_array_struct_int(10, n)})
    df2 = pd.DataFrame({"B": gen_random_arrow_array_struct_list_int(10, n)})
    df3 = pd.DataFrame({"B": gen_random_arrow_list_list_int(1, 0.1, n)})
    df4 = pd.DataFrame({"B": gen_random_arrow_struct_struct(10, n)})
    check_gather_operation(df1)
    check_gather_operation(df2)
    check_gather_operation(df3)
    check_gather_operation(df4)


@pytest.mark.slow
def test_dataframe_columns_name():
    """A little known feature of pandas dataframe is that one can attribute
    a name to the columns. As far as I know this shows up only in pivot_table
    and crosstab functionalities.
    -
    This feature is only partially supported in BODO. It is supported in
    boxing/unboxing, but it is not in gatherv which makes this test fail
    in distributed mode. When columns name are supported, remove the
    dist_test=False
    -
    A complete support of this feature in Bodo looks like a lot of work
    for only esthetic purposes."""

    def f(df):
        return df

    data = {"Name": ["Tom", "Jack", "nick", "juli"], "marks": [99, 98, 95, 90]}
    df = pd.DataFrame(data, index=["rank1", "rank2", "rank3", "rank4"])
    df.columns.name = "D"
    df.index.name = "A"
    check_func(f, (df,), dist_test=False)


def test_unroll_loop(memory_leak_check, is_slow_run):
    """Test unrolling constant loops when necessary for typing in getitem/setitem of
    dataframe columns.
    """

    def impl1(df):
        s = 0
        for c in df.columns:
            s += df[c].sum()
        return s

    def impl2(df):
        s = 0
        for c in ["A", "B"]:
            if c != "A":
                s += df[c].sum()
        return s

    def impl3(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
        for c in df.columns:
            df[c + "2"] = (df[c] - df[c].mean()) / df[c].std()
        return df

    # loop with multiple exits shouldn't be transformed
    def impl4(df):
        s = 0
        i = 0
        c_list = ["A", "B"]
        while True:
            c = c_list[i]
            if c not in ["B"]:
                break
            s += df[c].sum()
            i += 1
            if i == len(c_list):
                break
        return s

    def impl5(n):
        df = pd.DataFrame({"A1": np.arange(n), "A2": np.arange(n) ** 2})
        s = 0
        for i in range(2):
            s += df["A" + str(i + 1)].sum()
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    if is_slow_run:
        check_func(impl3, (n,))
    with pytest.raises(BodoError, match="getitem using"):
        bodo.jit(impl4)(df)
    check_func(impl5, (n,))


@pytest.mark.slow
def test_unsupported_df_method():
    """ Raise Bodo error for unsupported df methods"""

    def test_impl():
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
        return df.agg(["sum", "min"])

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_impl)()


############################# old tests ###############################


@bodo.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


@pytest.mark.slow
class TestDataFrame(unittest.TestCase):
    def test_create1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
            return df.A

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_kws1(self):
        def test_impl(n):
            df = pd.DataFrame(data={"A": np.ones(n), "B": np.random.ranf(n)})
            return df.A

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_dtype1(self):
        def test_impl(n):
            df = pd.DataFrame(
                data={"A": np.ones(n), "B": np.random.ranf(n)}, dtype=np.int8
            )
            return df.A

        np.random.seed(5)
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
        df1 = bodo_func(n)
        df2 = test_impl(n)
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

        np.random.seed(5)
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

        np.random.seed(5)
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
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_box3(self):
        def test_impl(df):
            df2 = df[df.A != "dd"]
            return df2

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({"A": ["aa", "bb", "dd", "cc"]}, [3, 1, 2, -1])
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df), check_dtype=False)

    def test_box_dist_return(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n), "B": np.arange(n)})
            return df

        bodo_func = bodo.jit(distributed_block={"df"})(test_impl)
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

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape1(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.ones(n, np.int64), "B": np.random.ranf(n)})
            return df.shape

        np.random.seed(5)
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

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

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

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({"A": np.ones(n), "B": np.random.ranf(n)})
        bodo_func(df, arr)
        self.assertIn("C", df)
        np.testing.assert_almost_equal(df.C.values, arr)

    def test_set_column_reflect1_2(self):
        # same as previous test but with integer column names
        def test_impl(df, arr):
            df[2] = arr
            return df[2].sum()

        np.random.seed(5)
        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({1: np.ones(n), 3: np.random.ranf(n)})
        bodo_func(df, arr)
        self.assertIn(2, df)
        np.testing.assert_almost_equal(df[2].values, arr)

    def test_set_column_reflect2(self):
        def test_impl(df, arr):
            df["C"] = arr
            return df.C.sum()

        np.random.seed(5)
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
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df), check_dtype=False)

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
        out = test_impl(df)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna2(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    @unittest.skip("pending remove of old list(str) array")
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
        out = test_impl(df)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out, check_dtype=False)

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
