import json
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
import bodo.io.snowflake
import bodo.tests.utils
from bodo.libs.streaming.groupby import (
    delete_groupby_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    get_query_profile_location,
    temp_env_override,
)


@pytest.mark.parametrize(
    "func_name",
    ["sum", "median", "mean", "nunique", "var", "std", "kurtosis", "skew", "first"],
)
@pytest.mark.parametrize("use_np_data", [True, False])
def test_groupby_basic(func_name, use_np_data, memory_leak_check):
    """
    Tests support for the basic streaming groupby functionality.
    """

    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType((func_name,))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((1,))

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    groups = [1, 2, 1, 1, 2, 0, 1, 2] * 100
    # First is nondeterministic on multiple ranks so set the data equal to the groups
    # so each group only has one unique value
    data = (
        [1, 3, 5, 11, 1, 3, 5, 3] * 100
        if func_name != "first" and func_name != "nunique"
        else groups
    )
    df = pd.DataFrame(
        {
            "A": groups,
            "B": np.array(data, dtype=np.int32) if use_np_data else data,
        }
    )

    py_func = pd.Series.kurt if func_name == "kurtosis" else func_name
    expected_df = df.groupby("A", as_index=False).agg(py_func)

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_groupby_drop_duplicates(memory_leak_check):
    """
    Tests support for the basic streaming groupby functionality for drop_duplicates
    operation (select distinct ...).
    """

    keys_inds = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(())
    f_in_offsets = bodo.utils.typing.MetaType((0,))
    f_in_cols = bodo.utils.typing.MetaType(())

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    df = pd.DataFrame(
        {
            "A": [1, 1, 2, 4, 4, 2],
            "B": [1, 1, 3, 6, 6, 3],
        }
    )
    expected_df = df.drop_duplicates()

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_groupby_key_reorder(memory_leak_check):
    """
    Tests groupby where the key isn't in the front.
    """

    keys_inds = bodo.utils.typing.MetaType((1,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(("sum",))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((0,))

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    df = pd.DataFrame(
        {
            "B": [1, 3, 5, 11, 1, 3, 5, 3],
            "A": [1, 2, 1, 1, 2, 0, 1, 2],
        }
    )
    expected_df = df.groupby("A", as_index=False).agg("sum")

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize("func_name", ["nunique", "min"])
def test_groupby_dict_str(func_name, memory_leak_check):
    """
    Test groupby with dictionary-encoded strings
    """
    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType((func_name,))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((1,))

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    df = pd.DataFrame(
        {
            "A": ["xyz", "xyz", "wxy", "wxy", "vwx", "vwx"],
            "B": ["abc", "abc", "bcd", "def", "def", "bcd"],
        }
    )
    expected_df = df.groupby("A", as_index=False).agg(func_name)
    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
def test_produce_output(memory_leak_check):
    """
    Test output len is 0 if produce_output parameter is False
    """
    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(("max",))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((1,))

    @bodo.jit
    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * batch_size), ((_temp1 + 1) * batch_size))
            )
            is_last1 = (_temp1 * batch_size) >= len(df)
            _temp1 = _temp1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T2, is_last1, True)

        out_dfs = []
        is_last2 = False
        _temp2 = 0
        output_when_not_request_input = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(
                groupby_state, _temp2 != 0
            )
            if _temp2 == 0 and len(out_table) != 0:
                output_when_not_request_input = True
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )

            out_dfs.append(df_final)
            _temp2 = _temp2 + 1

        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs), output_when_not_request_input

    # Ensure that the output is empty if produce_output is False
    assert not test_groupby(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))[1]


@pytest.mark.skipif(bodo.get_size() != 2, reason="requires 2 ranks")
def test_groupby_input_request(memory_leak_check):
    """Make sure input_request back pressure flag is set properly in groupby build"""

    keys_inds = bodo.utils.typing.MetaType((1,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 4000
    fnames = bodo.utils.typing.MetaType(("sum",))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    f_in_cols = bodo.utils.typing.MetaType((0,))

    @bodo.jit(distributed=["df"])
    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        saved_input_request1 = True
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            _iter_1 = _iter_1 + 1
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            is_last1, input_request1 = groupby_build_consume_batch(
                groupby_state, T3, is_last1, True
            )
            # Save input_request1 flag after the first iteration, which should be set to
            # False since shuffle send buffer is full (is_last1=True triggers shuffle)
            if _iter_1 == 1:
                saved_input_request1 = input_request1
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs), saved_input_request1

    df = pd.DataFrame(
        {
            "B": [1, 3, 5, 11, 1, 3, 5, 3],
            "A": [1, 2, 1, 1, 2, 0, 1, 2],
        }
    )
    # Lower shuffle size threshold to trigger shuffle in first iteration
    with temp_env_override({"BODO_SHUFFLE_THRESHOLD": "1"}):
        assert not test_groupby(df)[1]


def test_groupby_acc_path_fallback(memory_leak_check):
    """
    Test that functions like mean/count/sum/skew/kurtosis/etc.
    work correctly even when they go through the fallback
    ACC path (due to having one or more running values that
    are strings or because one of the functions is nunique/median).
    """
    keys_inds = bodo.utils.typing.MetaType((0,))
    out_cols = [
        "A",
        "B_max",
        "B_min",
        "C_sum",
        "C_count",
        "C_mean",
        "D_skew",
        "D_kurtosis",
        "E_var",
        "E_std",
        "F_boolxor_agg",
    ]
    out_col_meta = bodo.utils.typing.ColNamesMetaType(tuple(out_cols))
    in_kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3, 4, 5))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(
        (
            "max",
            "min",
            "sum",
            "count",
            "mean",
            "skew",
            "kurtosis",
            "var",
            "std",
            "boolxor_agg",
        )
    )
    f_in_cols = bodo.utils.typing.MetaType((1, 1, 2, 2, 2, 3, 3, 4, 4, 5))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    def impl(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            in_kept_cols,
            6,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
            _iter_1 = _iter_1 + 1

        is_last2 = False
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_groupby_state(groupby_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, out_col_meta
        )
        return out_df

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 100, dtype="Int64"),
            # The min/max on this column is what will force the accumulating path.
            "B": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 100
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 25
            ),
            "C": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 50,
                dtype=np.float32,
            ),
            "D": np.array(
                list(np.arange(23, 39)) * 50,
                dtype=np.uint64,
            ),
            "E": np.array(
                (list(np.arange(98, 106)) * 50) + (list(np.arange(32, 40)) * 50),
                dtype=np.float32,
            ),
            "F": np.array(
                [True] + [False] * 799,
                dtype=bool,
            ),
        }
    )

    expected_df = df.groupby("A", as_index=False).agg(
        {
            "B": ["max", "min"],
            "C": ["sum", "count", "mean"],
            "D": ["skew", pd.Series.kurt],
            "E": ["var", "std"],
        }
    )
    expected_df.reset_index(inplace=True, drop=True)
    expected_df.columns = out_cols[:-1]
    expected_df["F_boolxor_agg"] = False
    expected_df["F_boolxor_agg"] = expected_df["F_boolxor_agg"].where(
        expected_df["A"] != 1, True
    )

    check_func(
        impl,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize("func_names", [("sum", "var", "mean", "kurtosis", "skew")])
def test_groupby_multiple_funcs(func_names, memory_leak_check):
    """
    Tests support for multiple columns of different function types (all
    go through agg path).
    """

    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        )
    )
    kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3, 4, 5))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(func_names)
    f_in_offsets = bodo.utils.typing.MetaType((0, 1, 2, 3, 4, 5))
    f_in_cols = bodo.utils.typing.MetaType((1, 2, 3, 4, 5))

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 6
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    df = pd.DataFrame(
        {
            "A": [1, 2, 1, 1, 2, 0, 1, 2],
            "B": [1, 3, 5, 11, 1, 3, 5, 3],
            "C": [1, 3, 5, 11, 1, 3, 5, 3],
            "D": [1, 3, 5, 11, 1, 3, 5, 3],
            "E": [1, 3, 5, 11, 1, 3, 5, 3],
            "F": [1, 3, 5, 11, 1, 3, 5, 3],
        }
    )

    py_funcs = [
        pd.Series.kurt if func_name == "kurtosis" else func_name
        for func_name in func_names
    ]
    agg_args = {}
    for i, col in enumerate(df.columns[1:]):
        agg_args[col] = py_funcs[i]
    expected_df = df.groupby("A", as_index=False).agg(agg_args)

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 1, 0, 2, 1, 2, 2],
                    "B": pd.array(
                        [[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 2,
                        dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int64()))),
                    ),
                    "C": pd.array(
                        [[1, 2], [3], [4, 5, 6], [0]] * 2,
                        dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                    ),
                    "D": pd.array([1, 2, 3, 4] * 2),
                    "E": ["xyz", "xyz", "wxy", "wxy"] * 2,
                    "F": pd.array(
                        [
                            [],
                            None,
                            ["A"],
                            ["A", None, "B"],
                            ["A"],
                            ["X", None, "Y"],
                            None,
                            [],
                        ],
                        dtype=pd.ArrowDtype(pa.large_list(pa.string())),
                    ),
                    "G": pd.array(
                        [
                            [],
                            None,
                            ["A"],
                            ["A", None, "B"],
                            ["A"],
                            ["X", None, "Y"],
                            None,
                            [],
                        ],
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.dictionary(pa.int32(), pa.string()))
                        ),
                    ),
                }
            ),
            id="array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 1, 0, 2, 1, 2, 2],
                    "B": pd.array(
                        [
                            {
                                "Q": ["A"],
                                "R": ["A"],
                                "W": {"A": 1, "B": "A"},
                                "X": "AB",
                                "Y": [1.1, 2.2],
                                "Z": [[1], None, [3, None]],
                            },
                            {
                                "Q": None,
                                "R": None,
                                "W": {"A": 1, "B": "ABC"},
                                "X": "C",
                                "Y": [1.1],
                                "Z": [[11], None],
                            },
                            None,
                            {
                                "Q": ["AE", "IOU", None],
                                "R": ["A", "CDE", None],
                                "W": {"A": 1, "B": ""},
                                "X": "D",
                                "Y": [4.0, 6.0],
                                "Z": [[1], None],
                            },
                            {
                                "Q": ["Y"],
                                "R": ["CDE"],
                                "W": {"A": 1, "B": "AA"},
                                "X": "VFD",
                                "Y": [1.2],
                                "Z": [[], [3, 1]],
                            },
                            {
                                "Q": [],
                                "R": [],
                                "W": {"A": 1, "B": "DFG"},
                                "X": "LMMM",
                                "Y": [9.0, 1.2, 3.1],
                                "Z": [[10, 11], [11, 0, -3, -5]],
                            },
                            {
                                "Q": ["X", None, "Z"],
                                "R": ["CDE", None, "BC"],
                                "W": {"A": 1, "B": "DFG"},
                                "X": "LMMM",
                                "Y": [9.0, 1.2, 3.1],
                                "Z": [[10, 11], [11, 0, -3, -5]],
                            },
                            None,
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("Q", pa.large_list(pa.string())),
                                    pa.field(
                                        "R",
                                        pa.large_list(
                                            pa.dictionary(pa.int32(), pa.string())
                                        ),
                                    ),
                                    pa.field(
                                        "W",
                                        pa.struct(
                                            [
                                                pa.field("A", pa.int8()),
                                                pa.field("B", pa.string()),
                                            ]
                                        ),
                                    ),
                                    pa.field("X", pa.string()),
                                    pa.field("Y", pa.large_list(pa.float64())),
                                    pa.field(
                                        "Z", pa.large_list(pa.large_list(pa.int64()))
                                    ),
                                ]
                            )
                        ),
                    ),
                }
            ),
            id="struct",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 1, 0, 2, 1, 2, 2],
                    "B": pd.Series(
                        [{1: 1.4, 2: 3.1}, None, {}, {11: 3.4, 21: 3.1, 9: 8.1}] * 2,
                        dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
                    ),
                    "C": pd.Series(
                        [
                            {1: [], 2: None},
                            None,
                            {},
                            {11: ["A"], 21: ["B", None], 9: ["C"]},
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(
                            pa.map_(pa.int64(), pa.large_list(pa.string()))
                        ),
                    ),
                    "D": pd.Series(
                        [
                            {1: [], 2: None},
                            None,
                            {},
                            {11: ["A"], 21: ["B", None], 9: ["C"]},
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(
                            pa.map_(
                                pa.int64(),
                                pa.large_list(pa.dictionary(pa.int32(), pa.string())),
                            )
                        ),
                    ),
                }
            ),
            id="map",
            marks=pytest.mark.skip("fix map expected output handling in the test"),
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 1, 0, 2, 1, 2, 2],
                    "B": [(1, 1.1), (2, 2.2), None, (4, 4.4)] * 2,
                }
            ),
            id="tuple",
            marks=pytest.mark.skip(
                "[BSE-2076] TODO: Support tuple array in Arrow boxing/unboxing"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "fstr",
    [
        pytest.param("sum", id="check_error"),
        pytest.param("first", id="first"),
        pytest.param("count", id="count"),
        pytest.param("size", id="size"),
    ],
)
def test_groupby_nested_array_data(memory_leak_check, df, fstr):
    """
    Tests support for streaming groupby with nested array data.
    """
    from bodo.utils.typing import BodoError

    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType((fstr,) * (num_cols - 1))
    if fstr == "size":
        # 'size' doesn't need input columns.
        f_in_offsets = f_in_offsets = bodo.utils.typing.MetaType(tuple([0] * num_cols))
        f_in_cols = bodo.utils.typing.MetaType(tuple(range(0, 0)))
    else:
        f_in_offsets = bodo.utils.typing.MetaType(tuple(range(num_cols)))
        f_in_cols = bodo.utils.typing.MetaType(tuple(range(1, num_cols)))

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            kept_cols,
            num_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    if fstr == "count":
        expected_df = df.groupby("A", as_index=False, dropna=False, sort=True).count()
    elif fstr == "first":
        expected_df = df.groupby("A", as_index=False, dropna=False, sort=True).agg(
            dict.fromkeys(df.columns[1:], "first")
        )

        # We don't care if the value is actually the first element,
        # so we set df to have the same data value for each instance of a group key.
        cols = {"A": df["A"]}
        cols.update(
            {
                col: pd.Series(
                    # This works because df.groupby is sorted and
                    # the keys are 0,1,2 so we can use them as an index.
                    # If the sequence isn't consecutive, this wouldn't work.
                    df["A"].apply(lambda x: expected_df[col][x]),
                    dtype=df[col].dtype,
                )
                for col in expected_df.columns[1:]
            }
        )
        df = pd.DataFrame(cols)
    elif fstr == "size":
        expected_df = df.copy(deep=True)
        # Logically replace every column with an integer due to pyarrow
        # issues.
        for i in range(1, len(expected_df.columns)):
            expected_df[expected_df.columns[i]] = 1
        expected_df = expected_df.groupby("A", as_index=False, dropna=False).agg(
            {column: lambda x: len(x) for column in expected_df.columns[1:]}
        )
    elif fstr == "sum":
        expected_df = pd.DataFrame()

    if fstr == "sum":
        with pytest.raises(
            BodoError,
            match="Groupby does not support semi-structured arrays for aggregations other than first, count and size",
        ):
            bodo.jit(test_groupby)(df)
    else:
        check_func(
            test_groupby,
            (df,),
            py_output=expected_df,
            reset_index=True,
            convert_columns_to_pandas=True,
            sort_output=True,
        )


@pytest.mark.parametrize(
    "df, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [
                            {1: 1.4, 2: 3.1},
                            {1: 1.4, 2: 3.1},
                            None,
                            {},
                            {11: 3.4, 21: 3.1, 9: 8.1},
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
                    )
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [{1: 1.4, 2: 3.1}, None, {}, {11: 3.4, 21: 3.1, 9: 8.1}],
                        dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
                    )
                }
            ),
            id="map",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [
                            {"a": "xyz", "b": "abc"},
                            None,
                            {},
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [pa.field("a", pa.string()), pa.field("b", pa.string())]
                            )
                        ),
                    )
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [{"a": "xyz", "b": "abc"}, None, {}],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [pa.field("a", pa.string()), pa.field("b", pa.string())]
                            )
                        ),
                    )
                }
            ),
            id="struct",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [[{}, None, {1: 1, 2: 2}], [{3: 3}]] * 2,
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.map_(pa.int64(), pa.int64()))
                        ),
                    )
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [[{}, None, {1: 1, 2: 2}], [{3: 3}]],
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.map_(pa.int64(), pa.int64()))
                        ),
                    )
                }
            ),
            id="list_of_map",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [["A12", None, "A12", "ABC"], ["A12", "ABC", "C"]] * 2,
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.dictionary(pa.int32(), pa.string()))
                        ),
                    )
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [["A12", None, "A12", "ABC"], ["A12", "ABC", "C"]],
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.dictionary(pa.int32(), pa.string()))
                        ),
                    )
                }
            ),
            id="list_of_str",
        ),
    ],
)
def test_groupby_nested_array_key(df, expected_df, memory_leak_check):
    """
    Tests support for streaming groupby with nested array keys.
    """
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    keys_inds = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(())
    f_in_offsets = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    f_in_cols = bodo.utils.typing.MetaType(())

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            kept_cols,
            num_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


@pytest.mark.skip(reason="Gaps in Table Builder")
def test_groupby_timestamptz_key(memory_leak_check):
    """
    Tests support for streaming groupby with Timestamptz
    """
    tz_arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 300),
            None,
        ]
        * 5
    )
    df = pd.DataFrame(
        {
            "A": ["A", "B", "C", "D"] * 5,
            "B": tz_arr,
        }
    )
    expected_df = pd.DataFrame(
        {
            "B": np.array(
                [bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 400), None]
            ),
            "A": ["ACACACACAC", "BDBDBDBDBD"],
        }
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    keys_inds = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(())
    f_in_offsets = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    f_in_cols = bodo.utils.typing.MetaType(())

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            kept_cols,
            num_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_window_output_work_stealing(memory_leak_check, capfd, tmp_path):
    """
    Test that the window-output-redistribution works as expected.
    """
    from mpi4py import MPI

    from bodo.utils.typing import ColNamesMetaType, MetaType

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    df = pd.DataFrame(
        {
            "A": ([1] * 500_000) + ([15, 20] * 50_000),
            "B": np.arange(600000),
            "C": np.arange(300000, 900000),
            # Dict-encoded column
            "D": pd.array(
                ["pizza", "pasta", "burrito"] * 200000,
                dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
            ),
            # Regular string column
            "E": pd.array(
                [
                    "breaking-bad",
                    "better-call-saul",
                    "sopranos",
                    "the-wire",
                    "west-wing",
                ]
                * 120000,
            ),
        }
    )

    # This is essentially the query we're executing:
    # SELECT A, B, C, D, E, rank() OVER (PARTITION BY A ORDER BY B ) AS RANK FROM DF
    @bodo.jit(distributed=["df"])
    def ref_impl(df):
        out_rank = df.groupby(
            "A", as_index=False, dropna=False, _is_bodosql=True
        ).window((("rank",),), ("B",), (True,), ("last",))
        return out_rank

    # Get expected output using the non-streaming path
    local_df = _get_dist_arg(df)
    expected_out_rank = ref_impl(local_df)
    expected_out = local_df.copy()
    expected_out["RANK"] = expected_out_rank["AGG_OUTPUT_0"]
    expected_out = bodo.allgatherv(expected_out)

    global_1 = MetaType((0, 1, 2, 3, 4))
    global_2 = MetaType((0,))
    global_3 = MetaType((1,))
    global_4 = MetaType((True,))
    global_5 = MetaType(("rank",))
    global_6 = MetaType((0, 1, 2, 3, 4))
    global_7 = ColNamesMetaType(("A", "B", "C", "D", "E", "RANK"))
    global_8 = MetaType((0, 1, 2, 3, 4, 5))
    global_9 = MetaType(((),))

    @bodo.jit(distributed=["df"])
    def window_impl(df):
        bodo.libs.query_profile_collector.init()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), global_1, 5
        )
        T2 = T1
        __bodo_is_last_streaming_output_1 = False
        _iter_1 = 0
        _temp1 = bodo.hiframes.table.local_len(T2)
        state_1 = bodo.libs.streaming.window.init_window_state(
            0,
            global_2,
            global_3,
            global_4,
            global_4,
            global_5,
            global_9,
            global_6,
            True,
            5,
            global_9,
        )
        __bodo_is_last_streaming_output_2 = False
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not (__bodo_is_last_streaming_output_2):
            T3 = bodo.hiframes.table.table_local_filter(
                T2, slice((_iter_1 * 4096), ((_iter_1 + 1) * 4096))
            )
            __bodo_is_last_streaming_output_1 = (_iter_1 * 4096) >= _temp1
            (
                __bodo_is_last_streaming_output_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                state_1, T3, __bodo_is_last_streaming_output_1
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)
        __bodo_is_last_streaming_output_3 = False
        _produce_output_1 = True
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(-1)
        )
        while not (__bodo_is_last_streaming_output_3):
            (
                T4,
                __bodo_is_last_streaming_output_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                state_1, _produce_output_1
            )
            T5 = T4
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, T5
            )
        bodo.libs.streaming.window.delete_window_state(state_1)
        T6 = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        T7 = bodo.hiframes.table.table_subset(T6, global_8, False)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index_1, global_7)
        bodo.libs.query_profile_collector.finalize()
        return df2

    with temp_env_override(
        {
            # Sync often for testing purposes
            "BODO_STREAM_GROUPBY_OUTPUT_WORK_STEALING_SYNC_ITER": "2",
            # Start stealing immediately.
            "BODO_STREAM_GROUPBY_OUTPUT_WORK_STEALING_TIME_THRESHOLD_SECONDS": "0",
            "BODO_STREAM_GROUPBY_DEBUG_OUTPUT_WORK_STEALING": "1",
            # Set a low value so we can test the multiple shuffles case.
            "BODO_STREAM_GROUPBY_WORK_STEALING_MAX_SEND_RECV_BATCHES_PER_RANK": "2",
            # Start re-distribution as soon as any rank is done.
            "BODO_STREAM_GROUPBY_WORK_STEALING_PERCENT_RANKS_DONE_THRESHOLD": "0",
            # Enable tracing
            "BODO_TRACING_LEVEL": "1",
            "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0,
        }
    ):
        # Explicitly prohibit taking the sort-based implementation
        old_disable_value = bodo.bodo_disable_streaming_window_sort
        try:
            bodo.bodo_disable_streaming_window_sort = True
            output = window_impl(_get_dist_arg(df))
            global_output = bodo.allgatherv(output)
            stdout, stderr = capfd.readouterr()
        finally:
            bodo.bodo_disable_streaming_window_sort = old_disable_value

    ### Uncomment for debugging purposes ###
    # with capfd.disabled():
    #     for i in range(bodo.get_size()):
    #         if bodo.get_rank() == i:
    #             print(f"output:\n{output}")
    #             print(f"stdout:\n{stdout}")
    #             print(f"stderr:\n{stderr}")
    #             if i == 0:
    #                 print(f"expected_out:\n{expected_out}")
    #         bodo.barrier()
    ###

    if bodo.get_rank() == 0:
        expected_log_messages = [
            "[DEBUG][GroupbyOutputState] Started work-stealing timer since 1 of 2 ranks are done outputting.",
            "[DEBUG][GroupbyOutputState] Starting work stealing.",
            "[DEBUG][GroupbyOutputState] Done with work stealing.",
        ]
    else:
        expected_log_messages = [None] * 3

    # Verify that the expected log messages are present.
    for expected_log_message in expected_log_messages:
        assert_success = True
        if expected_log_message is not None:
            assert_success = expected_log_message in stderr
        assert_success = comm.allreduce(assert_success, op=MPI.LAND)
        assert assert_success

    # Check for certain log messages using regex (in case the exact numbers present in the logs change)
    if bodo.get_rank() == 0:
        expected_log_messages = [
            r"\[DEBUG\]\[GroupbyOutputState\] Performed \d+ shuffles to redistribute data.",
            r"\[DEBUG\]\[GroupbyOutputState\] Received \d+ rows from other ranks during redistribution.",
        ]
    else:
        expected_log_messages = [
            r"\[DEBUG\]\[GroupbyOutputState\] Sent \d+ rows to other ranks during redistribution."
        ] + ([None] * 1)

    # Verify that the expected log messages are present.
    for expected_log_message in expected_log_messages:
        assert_success = True
        if expected_log_message is not None:
            assert_success = bool(re.search(expected_log_message, stderr))
        assert_success = comm.allreduce(assert_success, op=MPI.LAND)
        assert assert_success

    # Verify that the output itself is correct.
    assert global_output.shape[0] == df.shape[0], (
        f"Final output size ({global_output.shape[0]}) is not as expected ({df.shape[0]})"
    )

    pd.testing.assert_frame_equal(
        global_output.sort_values(list(expected_out.columns)).reset_index(drop=True),
        expected_out.sort_values(list(expected_out.columns)).reset_index(drop=True),
        check_dtype=False,
        check_index_type=False,
        atol=0.1,
    )

    ## Verify that the profile is as expected
    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)
    # Ensure that the temp directory isn't deleted until all ranks have read their respective file
    bodo.barrier()
    operator_report = profile_json["operator_reports"]["0"]
    assert "stage_2" in operator_report
    output_metrics = operator_report["stage_2"]["metrics"]
    output_metrics_dict = {x["name"]: x["stat"] for x in output_metrics}

    ### Uncomment for debugging purposes ###
    # with capfd.disabled():
    #     print(f"output_metrics: {output_metrics_dict}")
    ###

    assert "work_stealing_enabled" in output_metrics_dict
    assert output_metrics_dict["work_stealing_enabled"] == 1
    if bodo.get_rank() == 0:
        assert "started_work_stealing_timer" in output_metrics_dict
        assert output_metrics_dict["started_work_stealing_timer"] == 1
        assert "n_ranks_done_at_timer_start" in output_metrics_dict
        assert output_metrics_dict["n_ranks_done_at_timer_start"] == 1
    assert "performed_work_redistribution" in output_metrics_dict
    assert output_metrics_dict["performed_work_redistribution"] == 1
    assert "max_batches_send_recv_per_rank" in output_metrics_dict
    assert output_metrics_dict["max_batches_send_recv_per_rank"] == 2
    if bodo.get_rank() == 0:
        assert "n_ranks_done_before_work_redistribution" in output_metrics_dict
        assert output_metrics_dict["n_ranks_done_before_work_redistribution"] == 1
    assert "num_shuffles" in output_metrics_dict
    assert output_metrics_dict["num_shuffles"] == 32
    assert "redistribute_work_total_time" in output_metrics_dict
    assert output_metrics_dict["redistribute_work_total_time"] > 0
    assert "determine_redistribution_time" in output_metrics_dict
    assert "determine_batched_send_counts_time" in output_metrics_dict
    assert "shuffle_dict_unification_time" in output_metrics_dict
    assert "shuffle_output_append_time" in output_metrics_dict
    assert "shuffle_time" in output_metrics_dict
    assert "num_recv_rows" in output_metrics_dict
    if rank == 0:
        assert output_metrics_dict["num_recv_rows"] == 262144
    else:
        assert output_metrics_dict["num_recv_rows"] == 0
    assert "num_sent_rows" in output_metrics_dict
    if rank == 0:
        assert output_metrics_dict["num_sent_rows"] == 0
    else:
        assert output_metrics_dict["num_sent_rows"] == 262144


def test_groupby_decimal_types(memory_leak_check):
    """
    E2E test to verify decimal types are properly passed between python/C++
    """
    from decimal import Decimal

    # groups has some struct that contains a decimal
    groups = pd.Series(
        [
            {
                "x": Decimal("100.0"),
                "y": Decimal("99.9"),
            },
            {
                "x": Decimal("99.9"),
                "y": Decimal("100.0"),
            },
        ]
        * 5,
        dtype=pd.ArrowDtype(
            pa.struct(
                [
                    pa.field(
                        "x",
                        pa.decimal128(32, 12),
                    ),
                    pa.field(
                        "y",
                        pa.decimal128(31, 9),
                    ),
                ]
            )
        ),
    )

    df = pd.DataFrame(
        {
            "A": groups,
        }
    )

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    keys_inds = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 5
    fnames = bodo.utils.typing.MetaType(())
    f_in_offsets = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    f_in_cols = bodo.utils.typing.MetaType(())

    def test_groupby(df):
        groupby_state = init_groupby_state(
            -1, keys_inds, fnames, f_in_offsets, f_in_cols
        )

        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), kept_cols, 2
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(groupby_state)
        return pd.concat(out_dfs)

    expected_df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    {
                        "x": Decimal("100.0"),
                        "y": Decimal("99.9"),
                    },
                    {
                        "x": Decimal("99.9"),
                        "y": Decimal("100.0"),
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "x",
                                pa.decimal128(32, 12),
                            ),
                            pa.field(
                                "y",
                                pa.decimal128(31, 9),
                            ),
                        ]
                    )
                ),
            )
        }
    )

    check_func(
        test_groupby,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        only_seq=True,
    )
