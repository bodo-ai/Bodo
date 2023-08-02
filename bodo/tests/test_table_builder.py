# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Test Bodo's Table Builder python interface
"""


import numpy as np
import pandas as pd

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


def test_table_builder(memory_leak_check):
    """Test that table_builder will correctly concatenate it's inputs"""
    global_1 = MetaType((0, 1))
    col_names = ColNamesMetaType(("A", "B"))

    def test():
        A1 = np.arange(3) / 10
        A2 = np.arange(3) / 10 + 1
        B1 = np.arange(3)
        B2 = np.arange(3) + 10
        df1 = pd.DataFrame({"A": A1, "B": B1})
        df2 = pd.DataFrame({"A": A2, "B": B2})
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

    expected_df = pd.DataFrame(
        {"A": [0, 0.1, 0.2, 1, 1.1, 1.2], "B": [0, 1, 2, 10, 11, 12]}
    )
    check_func(test, (), py_output=expected_df, convert_to_nullable_float=False)


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
