import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.libs.stream_groupby import (
    delete_groupby_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.tests.utils import check_func, pytest_mark_one_rank
from bodo.utils.typing import BodoError


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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
    f_in_offsets = bodo.utils.typing.MetaType((0))
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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
            is_last1 = groupby_build_consume_batch(groupby_state, T2, is_last1, True)

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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
    expected_df["F_boolxor_agg"][expected_df["A"] == 1] = True

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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
                    "B": np.array(
                        [[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 2, object
                    ),
                    "C": np.array([[1, 2], [3], [4, 5, 6], [0]] * 2, object),
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
                                "X": "AB",
                                "Y": [1.1, 2.2],
                                "Z": [[1], None, [3, None]],
                                "W": {"A": 1, "B": "A"},
                                "Q": ["A"],
                            },
                            {
                                "X": "C",
                                "Y": [1.1],
                                "Z": [[11], None],
                                "W": {"A": 1, "B": "ABC"},
                                "Q": None,
                            },
                            None,
                            {
                                "X": "D",
                                "Y": [4.0, 6.0],
                                "Z": [[1], None],
                                "W": {"A": 1, "B": ""},
                                "Q": ["AE", "IOU", None],
                            },
                            {
                                "X": "VFD",
                                "Y": [1.2],
                                "Z": [[], [3, 1]],
                                "W": {"A": 1, "B": "AA"},
                                "Q": ["Y"],
                            },
                            {
                                "X": "LMMM",
                                "Y": [9.0, 1.2, 3.1],
                                "Z": [[10, 11], [11, 0, -3, -5]],
                                "W": {"A": 1, "B": "DFG"},
                                "Q": [],
                            },
                            {
                                "X": "LMMM",
                                "Y": [9.0, 1.2, 3.1],
                                "Z": [[10, 11], [11, 0, -3, -5]],
                                "W": {"A": 1, "B": "DFG"},
                                "Q": ["X", None, "Z"],
                            },
                            None,
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("X", pa.string()),
                                    pa.field("Y", pa.large_list(pa.float64())),
                                    pa.field(
                                        "Z", pa.large_list(pa.large_list(pa.int64()))
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
                                    pa.field("Q", pa.large_list(pa.string())),
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
                }
            ),
            id="map",
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

    keys_inds = bodo.utils.typing.MetaType((0,))
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(tuple([fstr]) * (num_cols - 1))
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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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

    match fstr:
        case ("count"):
            expected_df = df.groupby(
                "A", as_index=False, dropna=False, sort=True
            ).count()
        case ("first"):
            expected_df = df.groupby("A", as_index=False, dropna=False, sort=True).agg(
                {column: "first" for column in df.columns[1:]}
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
        case ("size"):
            expected_df = df.groupby("A", as_index=False, dropna=False).agg(
                {column: lambda x: len(x) for column in df.columns[1:]}
            )
        case ("sum"):
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
    ],
)
def test_groupby_nested_array_key(df, expected_df, memory_leak_check):
    """
    Tests support for streaming groupby with nested array keys.
    """
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    num_cols = len(df.columns)
    keys_inds = bodo.utils.typing.MetaType((tuple(range(num_cols))))
    kept_cols = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    batch_size = 3
    fnames = bodo.utils.typing.MetaType(tuple())
    f_in_offsets = bodo.utils.typing.MetaType(tuple(range(num_cols)))
    f_in_cols = bodo.utils.typing.MetaType(tuple())

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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
