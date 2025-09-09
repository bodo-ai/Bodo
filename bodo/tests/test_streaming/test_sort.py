import bodo

if bodo.test_compiler:
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.libs.str_arr_ext import StringArrayType
    from bodo.transforms.distributed_analysis import Distribution
    from bodo.utils.typing import ColNamesMetaType, MetaType


def test_stream_sort_compiles():
    """Test that the stream sort operator compiles successfully"""
    global_2 = ColNamesMetaType(("A", "B", "C"))
    global_1 = MetaType((0, 1, 2))

    # This is an edited version of the code generated for `SELECT * FROM TABLE ORDER BY C`
    def impl(df1):
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            4001, bodo.libs.memory_budget.OperatorType.SORT, 0, 0, -1
        )
        bodo.libs.memory_budget.register_operator(
            5001, bodo.libs.memory_budget.OperatorType.ACCUMULATE_TABLE, 1, 1, 1680
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 3
        )
        __bodo_is_last_streaming_output_1 = False
        _iter_1 = 0
        _temp9 = bodo.hiframes.table.local_len(T1)
        state_1 = bodo.libs.streaming.sort.init_stream_sort_state(
            4001, -1, -1, ["C"], [True], ["last"], ("A", "B", "C")
        )
        __bodo_is_last_streaming_output_2 = False
        while not (__bodo_is_last_streaming_output_2):
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * 4096), ((_iter_1 + 1) * 4096))
            )
            __bodo_is_last_streaming_output_1 = (_iter_1 * 4096) >= _temp9
            __bodo_is_last_streaming_output_2 = (
                bodo.libs.streaming.sort.sort_build_consume_batch(
                    state_1, T2, __bodo_is_last_streaming_output_1
                )
            )
            _iter_1 = _iter_1 + 1
        __bodo_is_last_streaming_output_3 = False
        _iter_2 = 0
        _produce_output_1 = True
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(5001)
        )
        while not (__bodo_is_last_streaming_output_3):
            (
                T4,
                __bodo_is_last_streaming_output_3,
            ) = bodo.libs.streaming.sort.produce_output_batch(
                state_1, _produce_output_1
            )
            T5 = T4
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, T5
            )
            _iter_2 = _iter_2 + 1
        bodo.libs.streaming.sort.delete_stream_sort_state(state_1)
        T6 = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T6,), index_1, global_2)
        return df2

    df_type = DataFrameType(
        data=(StringArrayType(), StringArrayType(), StringArrayType()),
        columns=("A", "B", "C"),
        dist=Distribution.OneD,
    )
    # Trigger eager compilation by supplying the expected type, but don't run
    # the resulting compiled code
    bodo.jit(df_type(df_type))(impl)
