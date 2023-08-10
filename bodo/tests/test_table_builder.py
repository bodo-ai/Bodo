# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's Table Builder python interface
"""

import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.libs.table_builder
from bodo.tests.utils import check_func
from bodo.utils.typing import ColNamesMetaType, MetaType


def test_table_builder_empty(memory_leak_check):
    """Test that table_builder outputs an empty Table when the input Table has
    no rows"""
    global_1 = MetaType((0, 1))
    col_names = ColNamesMetaType(("A", "B"))

    def test():
        A1 = np.arange(3) / 10
        B1 = np.arange(3)
        df1 = pd.DataFrame({"A": A1, "B": B1})
        df2 = df1[:0]
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), global_1, 2
        )
        table_builder = bodo.libs.table_builder.init_table_builder_state()

        loop = True
        while loop:
            bodo.libs.table_builder.table_builder_append(table_builder, T1)
            loop = False
        T2 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        idx = bodo.hiframes.pd_index_ext.init_range_index(0, len(T2), 1, None)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), idx, col_names)
        return out_df

    expected_df = pd.DataFrame({"A": [], "B": []}).astype(
        {"A": np.float64, "B": np.int64}
    )
    check_func(test, (), py_output=expected_df, convert_to_nullable_float=False)


@pytest.mark.parametrize("df1_len", [3, 5, 10])
@pytest.mark.parametrize("df2_len", [3, 5, 10])
def test_table_builder(df1_len, df2_len, memory_leak_check):
    """Test that table_builder will correctly concatenate it's inputs"""
    global_1 = MetaType((0, 1))
    col_names = ColNamesMetaType(("A", "B"))

    def test(df1, df2):
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), global_1, 2
        )
        table_builder = bodo.libs.table_builder.init_table_builder_state()

        loop = True
        while loop:
            bodo.libs.table_builder.table_builder_append(table_builder, T1)
            bodo.libs.table_builder.table_builder_append(table_builder, T2)
            loop = False
        T3 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        idx = bodo.hiframes.pd_index_ext.init_range_index(0, len(T3), 1, None)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T3,), idx, col_names)

        return out_df

    data1_a = np.arange(df1_len)
    data1_b = data1_a / 10
    df1 = pd.DataFrame({"A": data1_a, "B": data1_b})
    data2_a = np.arange(df2_len)
    data2_b = data2_a / 10
    df2 = pd.DataFrame({"A": data2_a, "B": data2_b})

    expected_df = pd.concat([df1, df2], ignore_index=True)

    # We need sort_output=True and reset_index=True here because when we have
    # multiple hosts we will execute the appends in a different order than if we
    # do it sequentially (i.e. appends are not commutative w.r.t ordering)
    check_func(
        test,
        (df1, df2),
        py_output=expected_df,
        sort_output=True,
        reset_index=True,
        convert_to_nullable_float=False,
    )


def test_table_builder_with_strings(memory_leak_check):
    """Test that table_builder will correctly concatenate it's string inputs
    (dictionary encoded)"""
    global_1 = MetaType((0,))
    col_names = ColNamesMetaType(("A",))

    def test():
        df1 = pd.DataFrame({"A": ["a", "b", "a", "b", "c", "a", "b", "c", "d"]})
        df2 = pd.DataFrame({"A": ["a", "b", "c", "d", "e", "e", "e", "e", "e"]})
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 1
        )
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), global_1, 1
        )
        table_builder = bodo.libs.table_builder.init_table_builder_state()

        loop = True
        while loop:
            bodo.libs.table_builder.table_builder_append(table_builder, T1)
            bodo.libs.table_builder.table_builder_append(table_builder, T2)
            loop = False
        T3 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        idx = bodo.hiframes.pd_index_ext.init_range_index(0, len(T3), 1, None)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T3,), idx, col_names)
        return out_df

    t1 = ["a", "b", "a", "b", "c", "a", "b", "c", "d"]
    t2 = ["a", "b", "c", "d", "e", "e", "e", "e", "e"]
    expected_df = pd.DataFrame({"A": t1 + t2})
    check_func(
        test, (), py_output=expected_df, convert_to_nullable_float=False, only_seq=True
    )
