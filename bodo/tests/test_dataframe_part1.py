# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Unittests for DataFrames
"""
import datetime
import operator
import random
import sys

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.tests.dataframe_common import *  # noqa
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    _get_dist_arg,
    _test_equal,
    check_func,
    count_array_OneDs,
    count_parfor_OneDs,
    gen_random_string_binary_array,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError, BodoWarning
from bodo.utils.utils import is_call_assign


@pytest.fixture(params=[1, -3, 20, 0, 1000])
def shift_amnt(request):
    return request.param


@pytest.mark.slow
def test_init_empty_df():
    def test_impl():
        return pd.DataFrame()

    check_func(
        test_impl,
        (),
    )


@pytest.mark.slow
def test_df_to_string():
    def test_impl(df):
        return df.to_string()

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [3.44, 4.5, 4.4, 6.77],
            "C": ["AABBCC", "DDEE FF", "EE", "EFG"],
            "D": ["AABBCC", "DDEE FF", "EE", "EFG"],
        }
    )
    check_func(test_impl, (df,), only_seq=True)


@pytest.mark.weekly
def test_df_select_dtypes_str_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes("bool")

    def test_impl2(df):
        return df.select_dtypes("float64")

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.weekly
def test_df_select_dtypes_include():
    """Make sure non-nullable array types are selected properly in df.select_dtypes()."""

    def test_impl_bool():
        df = pd.DataFrame(
            {"a": [1, 2] * 20, "b": [True, False] * 20, "c": [1.0, 2.0] * 20}
        )
        return df.select_dtypes(["bool"])

    def test_impl_int():
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2] * 20, dtype="Int64"),
                "b": [True, False] * 20,
                "c": [1.0, 2.0] * 20,
                "d": [1, 2] * 20,
            }
        )
        return df.select_dtypes(["int"])

    check_func(test_impl_bool, (), only_seq=True)
    check_func(test_impl_int, (), only_seq=True)


@pytest.mark.slow
def test_df_filter_cols(memory_leak_check):
    df = pd.DataFrame(
        np.array(([1, 2, 3], [4, 5, 6])),
        index=["mouse", "rabbit"],
        columns=["one", "two", "three"],
    )

    def test_regex(df):
        return df.filter(regex="e$")

    def test_like(df):
        return df.filter(like="ne")

    def test_items(df):
        return df.filter(items=["one", "three"])

    def test_items_with_axis_str(df):
        return df.filter(items=["one", "three"], axis="columns")

    def test_items_with_axis_int(df):
        return df.filter(items=["one", "three"], axis=1)

    check_func(test_regex, (df,))
    check_func(test_like, (df,))
    check_func(test_items, (df,))
    check_func(test_items_with_axis_str, (df,))
    check_func(test_items_with_axis_int, (df,))


@pytest.mark.weekly
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


@pytest.mark.weekly
def test_df_select_dtypes_type_include(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(bool)

    def test_impl2(df):
        return df.select_dtypes(np.float64)

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.weekly
def test_df_select_dtypes_type_exclude(select_dtypes_df):
    df = select_dtypes_df

    def test_impl1(df):
        return df.select_dtypes(exclude=bool)

    def test_impl2(df):
        return df.select_dtypes(exclude=np.float64)

    check_func(test_impl1, (df,))
    check_func(test_impl2, (df,))


@pytest.mark.slow
def test_dataframe_rename_dropped_col():
    """Tests that DataFrame.rename removes any unused columns
    even if renamed."""

    def test_impl():
        df = pd.DataFrame({"A": [1, 2] * 20, "B": ["a, werqwer", "erwr"] * 20})
        df = df.rename({"A": "C", "B": "D"}, axis=1)
        return df["D"]

    check_func(test_impl, (), dist_test=False)
    _check_IR_no_get_dataframe_data(test_impl, ())


def _check_IR_no_get_dataframe_data(test_impl, args):
    """ensures there is get_dataframe_data after optimizations"""
    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    # make sure there is no const call in IR
    for block in fir.blocks.values():
        for stmt in block.body:
            assert not (
                is_call_assign(stmt)
                and (
                    guard(find_callname, fir, stmt.value)
                    in (("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext"),)
                )
            )


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


@pytest.mark.weekly
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


@pytest.mark.weekly
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
def test_assign_lambda(memory_leak_check):
    """tests that df.assign is supported with lambda functions"""
    df = pd.DataFrame(
        {"A": [1, 2, 3] * 2, "B": ["a", "b", "c", "d", "e", "f"], "Z": [3, 2, 1] * 2}
    )

    def test_impl(df):
        return df.assign(C=lambda x: x["A"])

    def test_impl2(df):
        return df.assign(
            Q=df["A"] > df["Z"],
            B=lambda x: 2 * x["A"],
            C=42,
            D=lambda x: x["B"],
            A=lambda x: x["B"],
        )

    # check that assign has no side effects on the original dataframe
    def test_impl3(df):
        df1 = df.assign(
            A=lambda x: x["A"] > x["Z"],
        )

        df2 = df1.assign(
            A=df["A"] < df1["Z"],
        )
        return (df, df1, df2)

    # check everything
    def test_impl4(df):
        df1 = df.assign(
            A1=10,
            A2=df["A"] - 10,
            A3=df["A"] - 20,
            Q=df["A"] > df["Z"],
            K1=lambda x: "__" + x["B"] + "__",
            B=lambda x: 2 * x["A"],
            K2=lambda x: x["B"] * 3,
            C=42,
            D=lambda x: x["B"] + 10,
            L1=lambda x: x["A"],
            A=df["A"] - 30,
            L2=lambda x: x["A"],
        )
        df2 = df1.assign(
            A=lambda x: x["A"] * 2,
            Q2=df["A"] + df1["A"],
        )
        return (df, df2)

    check_func(test_impl, (df,))
    check_func(test_impl2, (df,))
    check_func(test_impl3, (df,))
    # test_impl3 unbox/boxing of df changes the data type to ArrowStringArray which
    # needs reversed to avoid failures in regular Pandas
    df["B"] = np.array(df["B"].values, object)
    check_func(test_impl4, (df,))


@pytest.mark.slow
def test_df_insert(memory_leak_check, is_slow_run):
    """Test df.insert()"""

    # new column
    def impl1(df):
        df.insert(0, "B", 42)
        return df

    # duplicate column name
    def impl2(df):
        df.insert(1, "B", 2 * df["A"], True)
        return df

    # invalid column index
    def impl3(df):
        df.insert("C", "B", 2)
        return df

    # invalid column name
    def impl4(df):
        df.insert(0, [df], 2)
        return df

    # invalid allow_duplicates
    def impl5(df):
        df.insert(0, "B", 2, "C")
        return df

    # duplicate error
    def impl6(df):
        df.insert(0, "B", 2)
        return df

    with pytest.warns(
        BodoWarning, match="input dataframe is passed as argument to JIT function"
    ):
        df = pd.DataFrame({"A": [1, 2, 3] * 2})
        check_func(impl1, (df,), copy_input=True)
        df = pd.DataFrame({"A": [1, 2, 3] * 2, "B": ["AA", "BBB", "CCCC"] * 2})
        check_func(impl2, (df,), copy_input=True)
    with pytest.raises(BodoError, match="should be a constant integer"):
        bodo.jit(impl3)(df)
    with pytest.raises(BodoError, match="should be a constant"):
        bodo.jit(impl4)(df)
    with pytest.raises(BodoError, match="should be a constant boolean"):
        bodo.jit(impl5)(df)
    with pytest.raises(BodoError, match="cannot insert"):
        bodo.jit(impl6)(df)


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


@pytest.mark.slow
def test_df_create_non_existent_columns(memory_leak_check):
    """test selecting non-existent columns"""

    def impl3():
        return pd.DataFrame(
            {"A": [1, 2, 3] * 5, "B": [4, 5, 6] * 5},
            columns=["A", "C"],
            dtype=np.float32,
        )

    # check_typing_issues=False since the input is intentionally empty
    check_func(impl3, (), is_out_distributed=False)


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


@pytest.mark.slow
def test_empty_df_drop_column(memory_leak_check):
    """test dropping the only column of a dataframe so it becomes empty"""

    def impl1(n):
        df = pd.DataFrame({"A": np.arange(n) * 2})
        return df.drop(columns=["A"])

    err_msg = "DataFrame.drop.*: Dropping all columns not supported."

    # TODO: [BE-285] Fix empty df issue
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(11)


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

    def impl1(c_name, n):
        return pd.DataFrame({"A": np.ones(n), c_name: np.arange(n)})

    def impl2(a, n):
        return pd.DataFrame({"A": np.ones(n), "B{}".format(a): np.arange(n)})

    check_func(impl1, ("BB", 11))
    check_func(impl2, (3, 11))


@pytest.mark.slow
def test_force_const_empty_list(memory_leak_check):
    """
    Make sure forcing an empty list value to be constant does not hang in typing
    [BE-1353]
    """

    def impl(df, a, i):
        column_list = a[:i]
        return df.groupby(column_list).sum()

    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(numba.core.errors.TypingError):
        bodo.jit(impl)(df, ["A", "B"], 0)


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


@pytest.mark.slow
def test_create_df_scalar(memory_leak_check):
    """
    Test scalar to array conversion in pd.DataFrame()
    """

    def impl(n):
        return pd.DataFrame({"A": 2, "B": np.arange(n)})

    check_func(impl, (11,))


@pytest.mark.slow
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
    "data",
    [
        pd.DataFrame({"A": range(10)}),
        np.arange(10),
        pd.Series(np.arange(10)),
        pd.array([1, 4, 1, 5, 11, 1, 3, 1, -1, 6]),
        np.array(
            [b"32234", b"342432", b"g43b2", b"4t242t", b" weew"] * 2, dtype=object
        ),
    ],
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
                pd.testing.assert_frame_equal(data, res, check_column_type=False)
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


@pytest.mark.slow
@pytest.mark.parametrize("nitems, niters", [(13, 10000)])
def test_random_shuffle_uniform(nitems, niters, memory_leak_check):
    """Test that bodo.random_shuffle's output follows a uniform distribution"""

    def impl(data):
        return bodo.random_shuffle(data)

    dist_impl = bodo.jit(distributed=["data"])(impl)

    data = np.arange(nitems)
    # Entry (i, j) indicates the frequency of element i moving to index j
    output_freqs = np.zeros((nitems, nitems))

    for _ in range(niters):
        x = _get_dist_arg(data)
        x = dist_impl(x)
        x = bodo.allgatherv(x)
        for i in range(nitems):
            output_freqs[i, x[i]] += 1
    expected_freq = niters / nitems
    assert np.all(3 / 4 * expected_freq < output_freqs)
    assert np.all(output_freqs < 4 / 3 * expected_freq)


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
                pd.testing.assert_frame_equal(data, res, check_column_type=False)
            else:
                np.testing.assert_array_equal(data, res)


@pytest.mark.slow
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
    pd.testing.assert_frame_equal(df_in_merge, df_out_merge, check_column_type=False)
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
    pd.testing.assert_frame_equal(
        df_in_merge, df_out_dist_merge, check_column_type=False
    )
    # The replicated case
    bodo_rep = bodo.jit(
        all_args_distributed_block=False,
        all_returns_distributed=False,
        returns_maybe_distributed=False,
        args_maybe_distributed=False,
    )(f)
    df_out_rep = bodo_rep(df_in_merge)
    pd.testing.assert_frame_equal(df_in_merge, df_out_rep, check_column_type=False)


@pytest.mark.slow
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
    pd.testing.assert_frame_equal(
        bodo_func(), impl(), check_dtype=False, check_column_type=False
    )


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
def test_df_index_init_with_range(memory_leak_check):
    def impl():
        return pd.DataFrame({"A": range(100)}).index

    check_func(impl, ())


@pytest.mark.slow
def test_df_index_range_index(memory_leak_check):
    """test RangeIndex created inside the function"""

    def impl():
        df = pd.DataFrame({"A": [2, 3, 1]})
        return df.index

    # distributed=False necessary due to [BE-1112]
    bodo_func = bodo.jit(distributed=False)(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


@pytest.mark.slow
def test_df_columns(df_value, memory_leak_check):
    def impl(df):
        return df.columns

    check_func(impl, (df_value,), is_out_distributed=False)


# TODO: Remove test when more empty DataFrame functionality is supported
# Include empty DataFrame in dataframe_common.py
def test_empty_df_columns(memory_leak_check):
    def impl(df):
        return df.columns

    df = pd.DataFrame()
    check_func(impl, (df,), is_out_distributed=False)


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


@pytest.mark.slow
def test_df_dtypes(df_value):
    def impl(df):
        return df.dtypes

    # Bodo avoids object arrays for string/bool dtypes so we use the equivalent
    # python output dtypes for testing
    py_output = df_value.dtypes
    df_type = bodo.typeof(df_value)
    for i in range(len(df_value.columns)):
        if py_output.iloc[i] == object:
            if df_type.data[i] == bodo.boolean_array:
                py_output.iloc[i] = pd.BooleanDtype()
            if df_type.data[i] in (bodo.string_array_type, bodo.dict_str_arr_type):
                py_output.iloc[i] = pd.StringDtype()
        # Bodo reads all bool arrays as nullable
        if py_output.iloc[i] == np.bool_:
            py_output.iloc[i] = pd.BooleanDtype()
        # Bodo boxes string arrays of categories as ArrowStringArray
        if (
            isinstance(df_type.data[i], bodo.CategoricalArrayType)
            and df_type.data[i].dtype.elem_type == bodo.string_type
        ):
            py_output.iloc[i] = pd.CategoricalDtype(
                py_output.iloc[i].categories.astype("string[pyarrow]")
            )

    check_func(impl, (df_value,), is_out_distributed=False, py_output=py_output)


def test_df_astype_dtypes(memory_leak_check):
    """test converting dataframe types to dtypes of another dataframe (BE-535)"""

    def impl(df1, df2):
        return df1.astype(df2.dtypes)

    df1 = pd.DataFrame(
        {
            "A": [2, 1, 1] * 2,
            "B": ["a", "b", "c"] * 2,
            "C": pd.array([1, 2, 3] * 2, "Int64"),
            "D": [True, False, True] * 2,
            "E": pd.date_range("2017-01-03", periods=6),
        }
    )
    df2 = df1.copy()
    df2["A"] = [2.0, 1.3, 1.6] * 2
    check_func(impl, (df1, df2))


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


def test_df_astype_dict(memory_leak_check):
    """test passing a dictionary of new data types to df.astype()"""

    def impl1(df):
        return df.astype({"A": str, "B": np.float32})

    def impl2(df, dtype):
        return df.astype(dtype)

    df = pd.DataFrame(
        {
            "A": [1, 3, 4, 2, 3],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": ["AA", "BB", "C", "D", "A2"],
        }
    )
    check_func(impl1, (df,))
    # TODO(ehsan): support passing non-str dtypes as argument
    check_func(impl2, (df, {"A": "str", "B": "float32"}))


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


@pytest.mark.slow
def test_df_rename_all_types(df_value, memory_leak_check):
    """
    Tests that df rename works with all dataframes
    in the fixture.
    """

    def test_impl(df):
        df.rename(columns={"A": "first", "B": "second"})

    check_func(test_impl, (df_value,))


@pytest.mark.slow
def test_df_rename_mapper_all_types(df_value, memory_leak_check):
    """
    Tests that df rename works with all dataframes
    in the fixture using mapper.
    """

    def test_impl(df):
        df.rename({"A": "first", "B": "second"}, axis=1)

    check_func(test_impl, (df_value,))


@pytest.mark.slow
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

    # test forcing input to constant dict
    def impl7(df, c):
        df.rename(columns=c)
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
        match="DataFrame.rename.*: errors parameter only supports default value ignore",
    ):
        bodo.jit(impl6)(df)
    check_func(impl7, (df, {"B": "bb", "C": "cc"}))


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


@pytest.mark.smoke
def test_df_notnull(df_value, memory_leak_check):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.notnull()

    check_func(impl, (df_value,))


def test_df_head(df_value, memory_leak_check):
    def impl(df):
        return df.head(3)

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_head_no_args(memory_leak_check):
    """
    Test that df.head() works with the default args.
    """

    def impl(df):
        return df.head()

    df = pd.DataFrame({"A": np.arange(5), "C": np.arange(5) ** 2})
    check_func(impl, (df,))


def test_df_tail(df_value, memory_leak_check):
    def impl(df):
        return df.tail(3)

    check_func(impl, (df_value,))


@pytest.mark.slow
def test_df_tail_no_args(memory_leak_check):
    """
    Test that df.tail() works with the default args.
    """

    def impl(df):
        return df.tail()

    df = pd.DataFrame({"A": np.arange(5), "C": np.arange(5) ** 2})
    check_func(impl, (df,))


@pytest.mark.slow
def test_df_tail_zero(memory_leak_check):
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [-5, -4, -3, -2, -1]})
    f = lambda df: df.tail(0)
    check_func(f, [df])


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


def test_df_abs3(memory_leak_check):
    """test dataframe.abs with timedelta values"""

    def impl(df):
        return df.abs()

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "A": pd.Series(
                np.random.randint(0, np.iinfo(np.int64).max, size=12)
            ).astype("timedelta64[ns]")
        }
    )
    check_func(impl, (df,))


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


@pytest.mark.slow
def test_df_corr_float64(memory_leak_check):
    """
    Test df.corr with no astype is required.
    """

    def impl(df):
        return df.corr()

    df = pd.DataFrame({"A": np.linspace(1, 50, dtype=np.float64)})
    check_func(impl, (df,), is_out_distributed=False)


@pytest.mark.slow
def test_df_corr_parallel(memory_leak_check):
    def impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        return df.corr()

    bodo_func = bodo.jit(impl)
    n = 11
    pd.testing.assert_frame_equal(
        bodo_func(n), impl(n), check_column_type=False, check_index_type=False
    )
    assert count_array_OneDs() >= 3
    assert count_parfor_OneDs() >= 1


@pytest.mark.slow
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


@pytest.mark.slow
def test_df_count(df_value, memory_leak_check):
    def impl(df):
        return df.count()

    check_func(impl, (df_value,), is_out_distributed=False)


@pytest.mark.slow
def test_df_prod(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.prod()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (df_value,), is_out_distributed=False, check_dtype=False)


@pytest.mark.slow
def test_df_sum(numeric_df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(numeric_df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.sum()

    # TODO: match Pandas 1.1.1 output dtype
    check_func(impl, (numeric_df_value,), is_out_distributed=False, check_dtype=False)


@pytest.mark.slow
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
        pytest.param(
            pd.DataFrame(
                {
                    "A": ["a", "b", "c", "d", None, None, None, "e", "f"],
                    "B": [1, 4, 3, 4, None, 5, 6, None, np.nan],
                    "C": [1, 2, None, 3, 4, 5, 6, None, 0],
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


@pytest.mark.slow
def test_empty_df_var_std(memory_leak_check):
    """Test var/std operation on empty dataframe"""

    def impl(df):
        return df.var()

    def impl2(df):
        return df.std()

    df = pd.DataFrame({"A": []})
    check_func(impl, (df,), is_out_distributed=False)
    check_func(impl2, (df,), is_out_distributed=False)


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


@pytest.mark.slow
def test_df_quantile(df_value, memory_leak_check):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    # pandas returns object Series for some reason when input has IntegerArray
    if isinstance(df_value.iloc[:, 0].dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # boolean input fails in Pandas with Numpy 1.20
    # TODO(ehsan): remove check when fixed
    if np.dtype("bool") in df_value.dtypes.values:
        return

    def impl(df):
        return df.quantile(0.3)

    check_func(impl, (df_value,), is_out_distributed=False, check_names=False)


@pytest.mark.slow
def test_df_pct_change(numeric_df_value, memory_leak_check):
    # not supported for dt64 yet, TODO: support and test
    if any(d == np.dtype("datetime64[ns]") for d in numeric_df_value.dtypes):
        return

    def test_impl(df):
        return df.pct_change(2)

    check_func(test_impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_describe(numeric_df_value, memory_leak_check):
    def test_impl(df):
        return df.describe(datetime_is_numeric=True)

    check_func(test_impl, (numeric_df_value,), is_out_distributed=False)


@pytest.mark.slow
def test_df_describe_mixed_dt(memory_leak_check):
    """Test mix of datetime and numeric data in describe(). Pandas avoids std for all
    datetime case, but adds std to the end when mixed.
    """

    def test_impl(df):
        return df.describe(datetime_is_numeric=True)

    # all datetime
    df = pd.DataFrame(
        {
            "A": pd.date_range("2017-01-03", periods=6),
            "B": pd.date_range("2019-01-03", periods=6),
        }
    )
    check_func(test_impl, (df,), is_out_distributed=False)
    # datetime mixed with numeric
    df = pd.DataFrame({"A": pd.date_range("2017-01-03", periods=6), "B": np.arange(6)})
    check_func(test_impl, (df,), is_out_distributed=False)


@pytest.mark.slow
def test_df_describe_mixed_types(memory_leak_check):
    """Test describe with numeric and not numeric columns. Default is to drop non-numeric ones"""

    def impl(df):
        return df.describe()

    df = pd.DataFrame(
        {
            "A": ["aa", "bb", "cc", "dd", "ee"],
            "B": [1, 2, 3, 4, 5],
            "C": pd.Categorical([1, 2, 5, 3, 3], ordered=True),
            "D": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )

    check_func(impl, (df,), is_out_distributed=False)


@pytest.mark.slow
def test_df_stack_trace(memory_leak_check):
    """Test that stack trace is suppressed in user mode and available in developer mode"""

    def test_impl(df):
        return df.pct_change()

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6],
            "B": pd.Series(pd.date_range(start="1/1/2018", end="1/4/2018", periods=6)),
        }
    )

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE

    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0
    with pytest.raises(
        numba.TypingError, match="Compilation error for DataFrame.pct_change"
    ):
        bodo.jit(test_impl)(df)

    # Test as a developer
    numba.core.config.DEVELOPER_MODE = 1
    with pytest.raises(numba.TypingError, match="- Resolution failure "):
        bodo.jit(test_impl)(df)

    # Reset back to original setting
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest.mark.slow
def test_df_stack_trace_numba(memory_leak_check):
    """Test that stack trace from numba is suppressed in user mode and available in developer mode"""

    def test_impl():
        ans = "xx" + 1
        return ans

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE

    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0
    with pytest.raises(numba.TypingError, match="Compilation error"):
        bodo.jit(test_impl)()

    # Test as a developer
    numba.core.config.DEVELOPER_MODE = 1
    with pytest.raises(
        numba.TypingError, match="No implementation of function Function"
    ):
        bodo.jit(test_impl)()

    # Reset back to original setting
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest.mark.slow
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


@pytest.mark.slow
def test_df_cumsum1(memory_leak_check):
    # remove this after NAs are properly handled
    def impl(df):
        return df.cumsum()

    df = pd.DataFrame({"A": [1, 8, 4, 11, -3]})
    check_func(impl, (df,))


@pytest.mark.slow
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


@pytest.mark.slow
def test_df_nunique(df_value, memory_leak_check):
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


@pytest.mark.slow
def test_df_idxmax_datetime(memory_leak_check):
    def impl(df):
        return df.idxmax()

    df = pd.DataFrame(
        {"A": [3, 5, 1, -1, 2]},
        pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
    )
    check_func(impl, (df,), is_out_distributed=False)


@pytest.mark.slow
def test_df_idxmax_all_types_axis0(df_value, memory_leak_check):
    """
    Test df.idxmax on all df types with axis=0
    """

    def test_impl(df):
        return df.idxmax()

    err_msg = "DataFrame.idxmax.* only supported for numeric column types. Column type: .* not supported."

    skip = False
    gen_output = False
    # If any of the columns isn't directly supported in Pandas, we need to compute
    # the py_output to match what Pandas should support. If a column isn't supported
    # in either Pandas or Bodo, we need to check the error message.
    for i, dtype in enumerate(df_value.dtypes):
        # This isn't supported in Pandas with BooleanArrays
        if dtype == object and is_bool_object_series(df_value[df_value.columns[i]]):
            gen_output = True

        # This isn't supported in Pandas
        if isinstance(dtype, pd.CategoricalDtype):
            gen_output = True

        # This isn't supported in Pandas
        if isinstance(dtype, pd.core.arrays.integer._IntegerDtype):
            gen_output = True

        # not supported for Strings or Bytes, This isn't supported in Pandas
        if not isinstance(dtype, pd.CategoricalDtype) and (
            isinstance(df_value[df_value.columns[i]].iat[0], str)
            or isinstance(df_value[df_value.columns[i]].iat[0], bytes)
        ):
            skip = True
            break

    if skip:
        # Check for an appropriate error message
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(df_value)
    else:
        if gen_output:
            # Generate the py_output
            outputs = []
            for colname in df_value.columns:
                column = df_value[colname]
                if column.dtype == object and is_bool_object_series(column):
                    series_val = test_impl(column.dropna().astype(np.bool_))
                elif column.dtype == object and isinstance(
                    column.iat[0], datetime.date
                ):
                    series_val = test_impl(
                        column.dropna().astype(np.dtype("datetime64[ns]"))
                    )
                elif isinstance(column.dtype, pd.CategoricalDtype):
                    column = column.astype(
                        pd.CategoricalDtype(column.dtype.categories, ordered=True)
                    )
                    df_value[colname] = column
                    na_dropped = column.dropna()
                    series_val = na_dropped.index[na_dropped.values.codes.argmax()]
                elif isinstance(column.dtype, pd.core.arrays.integer._IntegerDtype):
                    series_val = test_impl(
                        column.dropna().astype(column.dtype.numpy_dtype)
                    )
                else:
                    series_val = test_impl(column)
                outputs.append(series_val)
            py_output = pd.Series(outputs, index=df_value.columns)
        else:
            py_output = None
        check_func(
            test_impl, (df_value,), is_out_distributed=False, py_output=py_output
        )


@pytest.mark.slow
def test_df_idxmax_all_types_axis1(df_value, memory_leak_check):
    """
    Test df.idxmax on all df types with axis=1
    """
    # TODO: Support axis=1 [BE-281]
    def test_impl(df):
        return df.idxmax(axis=1)

    err_msg = "DataFrame.idxmax.*: axis parameter only supports default value 0"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(test_impl)(df_value)


@pytest.mark.slow
def test_df_idxmin_all_types_axis0(df_value, memory_leak_check):
    """
    Test df.idxmin on all df types with axis=0
    """

    def test_impl(df):
        return df.idxmin()

    err_msg = "DataFrame.idxmin.* only supported for numeric column types. Column type: .* not supported."

    skip = False
    gen_output = False
    # If any of the columns isn't directly supported in Pandas, we need to compute
    # the py_output to match what Pandas should support. If a column isn't supported
    # in either Pandas or Bodo, we need to check the error message.
    for i, dtype in enumerate(df_value.dtypes):
        # This isn't supported in Pandas with BooleanArrays
        if dtype == object and is_bool_object_series(df_value[df_value.columns[i]]):
            gen_output = True

        # This isn't supported in Pandas
        if isinstance(dtype, pd.CategoricalDtype):
            gen_output = True

        # This isn't supported in Pandas
        if isinstance(dtype, pd.core.arrays.integer._IntegerDtype):
            gen_output = True

        # not supported for Strings or Bytes, This isn't supported in Pandas
        if not isinstance(dtype, pd.CategoricalDtype) and (
            isinstance(df_value[df_value.columns[i]].iat[0], str)
            or isinstance(df_value[df_value.columns[i]].iat[0], bytes)
        ):
            skip = True
            break

    if skip:
        # Check for an appropriate error message
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(df_value)
    else:
        if gen_output:
            # Generate the py_output
            outputs = []
            for colname in df_value.columns:
                column = df_value[colname]
                if column.dtype == object and is_bool_object_series(column):
                    series_val = test_impl(column.dropna().astype(np.bool_))
                elif column.dtype == object and isinstance(
                    column.iat[0], datetime.date
                ):
                    series_val = test_impl(
                        column.dropna().astype(np.dtype("datetime64[ns]"))
                    )
                elif isinstance(column.dtype, pd.CategoricalDtype):
                    column = column.astype(
                        pd.CategoricalDtype(column.dtype.categories, ordered=True)
                    )
                    df_value[colname] = column
                    na_dropped = column.dropna()
                    series_val = na_dropped.index[na_dropped.values.codes.argmin()]
                elif isinstance(column.dtype, pd.core.arrays.integer._IntegerDtype):
                    series_val = test_impl(
                        column.dropna().astype(column.dtype.numpy_dtype)
                    )
                else:
                    series_val = test_impl(column)
                outputs.append(series_val)
            py_output = pd.Series(outputs, index=df_value.columns)
        else:
            py_output = None
        check_func(
            test_impl, (df_value,), is_out_distributed=False, py_output=py_output
        )


@pytest.mark.slow
def test_df_idxmin_all_types_axis1(df_value, memory_leak_check):
    """
    Test df.idxmin on all df types with axis=1
    """
    # TODO: Support axis=1 [BE-281]
    def test_impl(df):
        return df.idxmin(axis=1)

    err_msg = "DataFrame.idxmin.*: axis parameter only supports default value 0"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(test_impl)(df_value)


@pytest.mark.slow
def test_df_idxmax(numeric_df_value, memory_leak_check):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmax()

    check_func(impl, (numeric_df_value,), is_out_distributed=False)


@pytest.mark.slow
def test_df_idxmin(numeric_df_value, memory_leak_check):
    if any(not _is_supported_argminmax_typ(d) for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.idxmin()

    check_func(impl, (numeric_df_value,), is_out_distributed=False)


@pytest.mark.slow
def test_df_take(df_value, memory_leak_check):
    def impl(df):
        return df.take([1, 3])

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(df_value),
        impl(df_value),
        check_dtype=False,
        check_column_type=False,
        check_categorical=False,
        check_index_type=False,
    )


# TODO: add memory_leak_check
@pytest.mark.slow
def test_df_sort_index(df_value):
    # skip NAs
    # TODO: handle NA order
    if df_value.isna().sum().sum():
        return

    def impl(df):
        return df.sort_index()

    # TODO: use larger input to avoid empty object array in output
    check_func(impl, (df_value,), check_typing_issues=False)


def test_df_shift_numeric(numeric_df_value, shift_amnt, memory_leak_check):
    def impl(df, n):
        return df.shift(n)

    check_func(
        impl,
        (
            numeric_df_value,
            shift_amnt,
        ),
        check_dtype=False,
    )


def test_df_shift_numeric_with_fill(numeric_df_value, shift_amnt, memory_leak_check):
    def impl(df, fill_val, n):
        return df.shift(n, fill_value=fill_val)

    check_func(impl, (numeric_df_value, numeric_df_value.iloc[0, 0], shift_amnt))


def test_df_shift_string_df_with_fill(shift_amnt, memory_leak_check):
    df = pd.DataFrame(
        {
            "A": gen_random_string_binary_array(12),
            "B": gen_nonascii_list(12),
            "C": gen_random_string_binary_array(12),
        }
    )

    def impl(df, n):
        return df.shift(n, fill_value="foo")

    check_func(impl, (df, shift_amnt))


def test_df_shift_binary_df_with_fill(shift_amnt, memory_leak_check):
    df = pd.DataFrame(
        {
            "A": gen_random_string_binary_array(12, is_binary=True),
            "B": gen_random_string_binary_array(12, is_binary=True),
            "C": gen_random_string_binary_array(12, is_binary=True),
        }
    )

    def impl(df, n):
        return df.shift(n, fill_value=b"foo")

    check_func(impl, (df, shift_amnt))


@pytest.mark.slow
def test_df_shift_unsupported(df_value, memory_leak_check):
    """
    Test for the Dataframe.shift inputs that are expected to be unsupported.
    """
    # Dataframe.shift supports ints, floats, dt64, nullable
    # int/bool/decimal/date and strings
    is_unsupported = False
    bodo_type = bodo.typeof(df_value)
    for column_type in bodo_type.data:
        if not bodo.hiframes.rolling.is_supported_shift_array_type(column_type):
            is_unsupported = True
            break

    def impl(df):
        return df.shift(2)

    if not is_unsupported:
        check_func(impl, (df_value,))
        return

    with pytest.raises(BodoError, match=r"Dataframe.shift\(\) column input type"):
        bodo.jit(impl)(df_value)


@pytest.mark.slow
def test_df_shift_error_periods(memory_leak_check):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    def test_impl(df, periods):
        return df.shift(periods)

    with pytest.raises(BodoError, match="'periods' input must be an integer"):
        bodo.jit(test_impl)(df, 1.0)


@pytest.mark.slow
def test_df_diff(numeric_df_value, memory_leak_check):
    """test DataFrame.diff()"""
    # Pandas as of 1.2.2 is buggy for uint8 and produces wrong results
    if any(t == np.uint8 for t in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.diff()

    check_func(impl, (numeric_df_value,))


@pytest.mark.slow
def test_df_set_index(df_value, memory_leak_check):
    """
    Test DataFrame.set_index on all of our df types.
    """

    def impl(df):
        return df.set_index("A")

    # TODO: [BE-284] fix nullable int. Produces the incorrect value when there are nulls.
    if isinstance(df_value.iloc[:, 0].dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # # TODO(ehsan): test non-str columns using 'df_value.columns[0]' instead of 'A" when
    # # Numba can convert freevars to literals

    if "A" not in df_value.columns:
        df = df_value.rename(columns={df_value.columns[0]: "A"})
    else:
        df = df_value.copy(deep=True)

    check_func(impl, (df,))


@pytest.mark.slow
def test_df_set_index_empty_dataframe(memory_leak_check):
    """
    Tests DataFrame.set_index that produces an empty DataFrame.
    """

    def test_impl(df):
        return df.set_index("A")

    np.random.seed(5)
    df = pd.DataFrame({"A": np.random.randn(10)})
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_empty_df_len(memory_leak_check):
    """
    Tests that the length of a dataframe returns the length of the index.
    """

    def test_impl():
        df = pd.DataFrame({"A": np.arange(10)})
        df = df.set_index("A")
        return len(df)

    check_func(test_impl, ())


def test_df_reset_index1(df_value, memory_leak_check):
    """Test DataFrame.reset_index(drop=False) on various dataframe/index combinations"""

    def impl(df):
        return df.reset_index()

    check_func(impl, (df_value,))


@pytest.mark.parametrize(
    "test_index",
    [
        # named numeric index
        pd.Index([3, 1, 2, 4, 6], dtype="Int64", name="AA"),
        pytest.param(
            pd.Index([3, 1, 2, 4, 6], dtype="UInt64", name="AA"), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.Index([3.1, 1.2, 2.3, 4.4, 6.6], dtype="float64", name="AA"),
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


@pytest.mark.slow
def test_df_reset_index_levels(memory_leak_check):
    """Test DataFrame.reset_index() with list or int in 'level' argument"""

    def impl1(df):
        return df.reset_index(level=[0, 1, 2], drop=True)

    def impl2(df):
        return df.reset_index(level=0, drop=True)

    df = pd.DataFrame(
        {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]},
        pd.MultiIndex.from_arrays(
            [
                ["ABCD", "V", "CAD", "", "AA"],
                [1.3, 4.1, 3.1, -1.1, -3.2],
                pd.date_range(start="2018-04-24", end="2018-04-27", periods=5),
            ],
        ),
    )
    check_func(impl1, (df,), only_seq=True)
    df = pd.DataFrame(
        {"A": [1, 3, 1, 2, 3], "B": ["F", "E", "F", "S", "C"]},
        [1.3, 4.1, 3.1, -1.1, -3.2],
    )
    check_func(impl2, (df,), only_seq=True)


def test_df_duplicated(memory_leak_check):
    def impl(df):
        return df.duplicated()

    df = pd.DataFrame({"A": ["A", "B", "A", "B", "C"], "B": ["F", "E", "F", "S", "C"]})
    check_func(impl, (df,), sort_output=True)
    df = pd.DataFrame(
        {
            "A": [1, 3, 1, 2, 3],
            "B": ["F", "E", "F", "S", "C"],
        },
        index=[3, 1, 2, 4, 6],
    )
    check_func(impl, (df,), sort_output=True)
    # empty dataframe corner case
    df = pd.DataFrame()
    check_func(impl, (df,))


def test_df_duplicated_binary_values():
    def impl(df):
        return df.duplicated()

    df = pd.DataFrame(
        {
            "A": [b"asdghas", b"bajskhd", b"", bytes(2), b"vjbh"],
            "B": [b"F", b"E", b"F", b"S", b"C"],
        }
    )
    check_func(impl, (df,), sort_output=True)


def test_drop_all_types(df_value, memory_leak_check):
    """
    Function that tests that drop works on our df types.
    """

    def test_impl1(df):
        return df.drop(labels="A", axis=1)

    def test_impl2(df):
        return df.drop(columns=["A"])

    if "A" not in df_value.columns:
        df = df_value.rename(columns={df_value.columns[0]: "A"})
    else:
        df = df_value.copy(deep=True)
    # Bodo Errors if it returns an empty DataFrame.
    if len(df_value.columns) == 1:
        df["B"] = df["A"]
    check_func(
        test_impl1,
        (df,),
    )
    check_func(
        test_impl2,
        (df,),
    )


def test_drop_duplicates_all_types(df_value, memory_leak_check):
    """
    Function that tests that drop_duplicates works on our df types.
    """

    def test_impl(df):
        return df.drop_duplicates()

    # Sort outputs for the distributed case where the output order doesn't
    # match.
    check_func(test_impl, (df_value,), sort_output=True, reset_index=True)


# TODO: [BE-266] Fix memory leak in duplicated
def test_duplicated_all_types(df_value):
    """
    Function that tests that duplicated works on our df types.
    """

    def test_impl(df):
        return df.duplicated()

    # TODO [BE-414]: Properly support NA

    # Index and order doesn't match in distributed case.
    check_func(test_impl, (df_value.dropna(),), sort_output=True, reset_index=True)


# TODO: [BE-266] Fix memory leak in duplicated
@pytest.mark.slow
def test_duplicated_cat_runtime():
    """
    Test that duplicated works on df types with Categories only
    known at runtime.
    """

    def test_impl(df):
        df["A"] = df["A"].astype("category")
        return df.duplicated()

    df = pd.DataFrame(
        {
            "A": pd.Series(["AA", "BB", "", "AA"] * 4),
        }
    )

    # Index and order doesn't match in distributed case.
    check_func(test_impl, (df,), copy_input=True, sort_output=True, reset_index=True)


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


@pytest.mark.slow
def test_dataframe_str_cmp(memory_leak_check):
    """test dataframe/string comparison for [BE-431]"""

    def test_impl(df, other):
        return df == other

    df = pd.DataFrame({"A": ["AA", "B", "A", "D", "A", "ABC"]})
    check_func(test_impl, (df, "A"))


def test_dataframe_binary_op_inconsistent_schemas(memory_leak_check):
    """Test dataframe binary operators for inputs with different schemas"""

    def impl1(df1, df2):
        return df1 - df2

    def impl2(df1, df2):
        df2 -= df1
        return df2

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"B": [3, 1, 2], "A": [1.2, 2.2, 3.3], "C": [1, 2, 3]})
    check_func(impl1, (df1, df2))
    check_func(impl2, (df1, df2), copy_input=True)


@pytest.mark.slow
@pytest.mark.parametrize("op", (operator.eq, operator.ne))
def test_dataframe_binary_comp_op_diff_types(op, memory_leak_check):
    """
    Checks for == and != between dataframes and scalars with types
    that may not match. Equality checks should always result in not
    equal if the types do not match.
    """
    # TODO: test parallelism
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(df, other):\n"
    func_text += "  return df {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    df1 = pd.DataFrame(
        {"A": [None, "str", "wall", "eat", "bear"], "B": [4, 6, 7, 1, 3]},
        index=[3, 5, 0, 7, 2],
    )
    df2 = pd.DataFrame(
        {
            "A": [None, "str", "wall", "eat", "bear"],
            "B": [4, 6, 7, 1, 3],
            "C": [None, "str", "wall", "eat", "bear"],
        },
        index=[3, 5, 0, 7, 2],
    )
    # df/scalar with 1 skipped column
    check_func(test_impl, (df1, 2))
    check_func(test_impl, (2, df1))
    # df/scalar with multiple skipped columns.
    # Adding this check ensures our generated arrays properly create
    # unique variable names.
    check_func(test_impl, (df2, 2))
    check_func(test_impl, (2, df2))


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


@pytest.mark.slow
def test_pd_isna(na_test_obj, memory_leak_check):
    obj = na_test_obj

    def impl(obj):
        return pd.isna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


def test_pd_isna_optional(memory_leak_check):
    """Test pd.isna() on Optional input"""

    def impl(a):
        b = None if a else 2
        return pd.isna(b)

    check_func(impl, (True,))
    check_func(impl, (False,))


@pytest.mark.slow
def test_pd_notna(na_test_obj, memory_leak_check):
    obj = na_test_obj

    def impl(obj):
        return pd.notna(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


@pytest.mark.slow
def test_pd_notnull(na_test_obj, memory_leak_check):
    obj = na_test_obj

    def impl(obj):
        return pd.notnull(obj)

    is_out_distributed = bodo.utils.utils.is_distributable_typ(bodo.typeof(obj))
    check_func(impl, (obj,), is_out_distributed)


@pytest.mark.slow
def test_scalar_dataframe(memory_leak_check):
    """
    Tests returning intializing a DataFrame of scalars
    using an index
    """

    def test_impl1():
        return pd.DataFrame(
            {"A": 12, "B": "cat", "C": True}, index=pd.RangeIndex(0, 20, 1)
        )

    def test_impl2():
        return pd.DataFrame(
            {"A": 12, "B": "cat", "C": True}, index=pd.RangeIndex(0, 1, 1)
        )

    check_func(test_impl1, ())
    check_func(test_impl2, ())


@pytest.mark.slow
def test_or_null(memory_leak_check):
    """
    Checks or null behavior inside DataFrames
    """

    def test_impl(df1, df2):
        return df1 | df2

    df1 = pd.DataFrame(
        {"A": pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")}
    )
    df2 = pd.DataFrame({"A": pd.Series([True, False, None] * 3, dtype="boolean")})

    check_func(test_impl, (df1, df2))


@pytest.mark.slow
def test_or_null_numpy(memory_leak_check):
    """
    Checks or null behavior inside DataFrames
    """

    def test_impl(df1, df2):
        return df1 | df2

    df1 = pd.DataFrame(
        {"A": pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")}
    )
    df2 = pd.DataFrame({"A": [True, False] * 3})

    check_func(test_impl, (df1, df2))


@pytest.mark.slow
def test_and_null(memory_leak_check):
    """
    Checks and null behavior inside DataFrames
    """

    def test_impl(df1, df2):
        return df1 & df2

    df1 = pd.DataFrame(
        {"A": pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")}
    )
    df2 = pd.DataFrame({"A": pd.Series([True, False, None] * 3, dtype="boolean")})

    check_func(test_impl, (df1, df2))


@pytest.mark.slow
def test_and_null_numpy(memory_leak_check):
    """
    Checks and null behavior inside DataFrames with Nullable
    and numpy booleans
    """

    def test_impl(df1, df2):
        return df1 & df2

    df1 = pd.DataFrame(
        {"A": pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")}
    )
    df2 = pd.DataFrame({"A": [True, False] * 3})

    check_func(test_impl, (df1, df2))


@pytest.mark.slow
def test_series_getitem_str_grpby_apply_and_lambda_assign():
    """
    Tests both series getitem with str in a groupby apply, and using a lambda function in df.assign.
    Example taken from H20 benchmark Q7.

    """
    data = [
        ["id016", "id016", "id0000042202", 15, 24, 5971, 5, 11, 37.211254],
        ["id016", "id016", "id0000042202", 15, 24, 5971, 5, 11, 37.211254],
        ["id016", "id016", "id0000042202", 15, 24, 5971, 5, 11, 37.211254],
        ["id016", "id016", "id0000042202", 15, 24, 5971, 5, 11, 37.211254],
        ["id016", "id016", "id0000042202", 15, 24, 5971, 5, 11, 37.211254],
    ]

    col_list = ["id1", "id2", "id3", "id4", "id5", "id6", "v1", "v2", "v3"]
    x = pd.DataFrame(data, columns=col_list)

    x["id1"] = x["id1"].astype("category")  # remove after datatable#1691
    x["id2"] = x["id2"].astype("category")
    x["id3"] = x["id3"].astype("category")
    x["id4"] = x["id4"].astype(
        "Int32"
    )  ## NA-aware types improved after h2oai/datatable#2761 resolved
    x["id5"] = x["id5"].astype("Int32")
    x["id6"] = x["id6"].astype("Int32")
    x["v1"] = x["v1"].astype("Int32")
    x["v2"] = x["v2"].astype("Int32")
    x["v3"] = x["v3"].astype("float64")

    def impl(df):
        return (
            df.groupby("id3", as_index=False, sort=False, observed=True, dropna=False)
            .agg({"v1": "max", "v2": "min"})
            .assign(range_v1_v2=lambda x: x["v1"] - x["v2"])[["id3", "range_v1_v2"]]
        )

    # Bodo returns a nullable type Int32, pandas returns an int64
    check_func(impl, (x,), sort_output=True, reset_index=True, check_dtype=False)
