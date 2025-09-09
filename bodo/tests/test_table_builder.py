"""Test Bodo's Table Builder python interface"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func

if bodo.test_compiler:
    import bodo.libs.table_builder
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
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        bodo.libs.table_builder.table_builder_append(table_builder, T1)
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
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        bodo.libs.table_builder.table_builder_append(table_builder, T1)
        bodo.libs.table_builder.table_builder_append(table_builder, T2)
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
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        bodo.libs.table_builder.table_builder_append(table_builder, T1)
        bodo.libs.table_builder.table_builder_append(table_builder, T2)
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


def test_table_builder_types_across_scopes(memory_leak_check):
    """Test that table_builder will infer it's input types even if the append is
    in a different block from the init call"""
    global_1 = MetaType((0,))
    col_names = ColNamesMetaType(("A",))

    def test():
        df1 = pd.DataFrame({"A": [1, 2, 3, 4]})
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 1
        )
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while True:
            bodo.libs.table_builder.table_builder_append(table_builder, T1)
            break
        T3 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        idx = bodo.hiframes.pd_index_ext.init_range_index(0, len(T3), 1, None)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T3,), idx, col_names)
        return out_df

    expected_df = pd.DataFrame({"A": [1, 2, 3, 4]})
    check_func(
        test, (), py_output=expected_df, convert_to_nullable_float=False, only_seq=True
    )


def test_table_builder(memory_leak_check):
    """Simple test for table builder"""

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    col_names_meta = ColNamesMetaType(tuple(df.columns))

    def f(df):
        idx = df.index
        state = bodo.libs.table_builder.init_table_builder_state(-1)
        # Note that the for loop is necessary,
        # _replace_state_definition requires that the init_table_builder_state
        # is not in the same basic block as the append.
        for i in range(1):
            table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
            bodo.libs.table_builder.table_builder_append(state, table)
        out_table = bodo.libs.table_builder.table_builder_finalize(state)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), idx, col_names_meta
        )
        return out_df

    check_func(f, (df,), py_output=df, use_table_format=True)


def test_chunked_table_builder_simple(memory_leak_check):
    """Simple test for the chunked table builder"""

    df = pd.DataFrame({"A": [1, 2, 3] * 10, "B": ["A", "B", "C"] * 10})
    col_names_meta = ColNamesMetaType(tuple(df.columns))

    def f(df):
        idx = df.index
        state = bodo.libs.table_builder.init_table_builder_state(
            -1, use_chunked_builder=True
        )
        for i in range(1):
            table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
            bodo.libs.table_builder.table_builder_append(state, table)
        out_table, is_last = bodo.libs.table_builder.table_builder_pop_chunk(state)
        bodo.libs.table_builder.delete_table_builder_state(state)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), idx, col_names_meta
        )
        return out_df

    check_func(f, (df,), py_output=df, use_table_format=True)


def test_chunked_table_builder_multiple_chunks(memory_leak_check):
    """test for the chunked table builder with multiple appends/pops of chunks"""
    chunk_size = bodo.bodosql_streaming_batch_size
    num_iters = 5

    df = pd.DataFrame({"A": np.arange(chunk_size), "B": ["A"] * chunk_size})
    col_names_meta = ColNamesMetaType(tuple(df.columns))

    def f(df):
        idx = df.index
        state = bodo.libs.table_builder.init_table_builder_state(
            -1, use_chunked_builder=True
        )
        for i in range(num_iters):
            table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
            bodo.libs.table_builder.table_builder_append(state, table)

        out_list = []
        for i in range(num_iters):
            out_table, is_last = bodo.libs.table_builder.table_builder_pop_chunk(state)
            out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), idx, col_names_meta
            )
            out_list.append(out_df)

        bodo.libs.table_builder.delete_table_builder_state(state)
        return pd.concat(out_list)

    check_func(
        f,
        (df,),
        py_output=pd.concat([df] * num_iters),
        use_table_format=True,
        sort_output=True,
    )
