import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
from bodo.libs.streaming.groupby import (
    get_partition_state,
)
from bodo.tests.utils import (
    check_func,
    pytest_mark_one_rank,
    temp_env_override,
)
from bodo.utils.typing import ColNamesMetaType, MetaType


def gen_simple_window_over_nothing_tests():
    params = []
    int32_nullable_arr = pd.array(
        [None if (i // 2500) % 3 < 2 else (i**3) % 99999 for i in range(10000)],
        dtype=pd.ArrowDtype(pa.int32()),
    )
    decimal_arr = pd.array(
        [
            None if i % 7 == 0 else Decimal(str(i) * 6 + "." + str(i) * 2)
            for i in range(10000)
        ],
        dtype=pd.ArrowDtype(pa.decimal128(32, 8)),
    )
    int32_numpy_arr = pd.array(
        [(i**4) % 99999 for i in range(10000)], dtype=pd.ArrowDtype(pa.int32())
    )
    string_arr = pd.array(
        [None if (i // 2000) % 2 == 0 else str(i**2) for i in range(5000)],
        dtype=pd.ArrowDtype(pa.large_string()),
    )
    int32_arr_all_null = pd.array(
        [None for i in range(10000)],
        dtype=pd.ArrowDtype(pa.int32()),
    )
    string_arr_all_null = pd.array(
        [None for i in range(5000)], dtype=pd.ArrowDtype(pa.large_string())
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "min",
            pd.array([8] * len(int32_nullable_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="min-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "max",
            pd.array(
                [99972] * len(int32_nullable_arr), dtype=pd.ArrowDtype(pa.int64())
            ),
            id="max-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "count",
            pd.array([2500] * len(int32_nullable_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="count-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "sum",
            pd.array(
                [124646669] * len(int32_nullable_arr), dtype=pd.ArrowDtype(pa.int64())
            ),
            id="sum-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "min",
            pd.array(
                [Decimal("111111.11000000")] * len(decimal_arr),
                dtype=pd.ArrowDtype(pa.decimal128(32, 8)),
            ),
            id="min-decimal_nullable",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "max",
            pd.array(
                [Decimal("999999999999999999999999.99999999")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(32, 8)),
            ),
            id="max-decimal",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "sum",
            pd.array(
                [Decimal("4242867611357352697748771451.87305829")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="sum-decimal",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "min",
            pd.array([0] * len(int32_numpy_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="min-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "max",
            pd.array([99976] * len(int32_numpy_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="max-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "count",
            pd.array([10000] * len(int32_numpy_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="count-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "sum",
            pd.array(
                [494581416] * len(int32_numpy_arr), dtype=pd.ArrowDtype(pa.int64())
            ),
            id="sum-int32_numpy",
        )
    )
    params.append(
        pytest.param(
            string_arr,
            "min",
            pd.array(
                ["10004569"] * len(string_arr), dtype=pd.ArrowDtype(pa.large_string())
            ),
            id="min-string",
        )
    )
    params.append(
        pytest.param(
            string_arr,
            "max",
            pd.array(
                ["9998244"] * len(string_arr), dtype=pd.ArrowDtype(pa.large_string())
            ),
            id="max-string",
        )
    )
    params.append(
        pytest.param(
            string_arr,
            "count",
            pd.array([2000] * len(string_arr), dtype=pd.ArrowDtype(pa.int64())),
            id="count-string",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "min",
            pd.array([None] * len(int32_arr_all_null), dtype=pd.ArrowDtype(pa.int32())),
            id="min-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "max",
            pd.array([None] * len(int32_arr_all_null), dtype=pd.ArrowDtype(pa.int32())),
            id="max-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "count",
            pd.array([0] * len(int32_arr_all_null), dtype=pd.ArrowDtype(pa.int32())),
            id="count-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "sum",
            pd.array([None] * len(int32_arr_all_null), dtype=pd.ArrowDtype(pa.int32())),
            id="sum-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "min",
            pd.array(
                [None] * len(string_arr_all_null),
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            id="min-string_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "max",
            pd.array(
                [None] * len(string_arr_all_null),
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            id="max-string_all_null",
        )
    )
    params.append(
        pytest.param(
            string_arr_all_null,
            "count",
            pd.array([0] * len(string_arr_all_null), dtype=pd.ArrowDtype(pa.int32())),
            id="count-string_all_null",
        )
    )
    params.append(
        pytest.param(
            decimal_arr,
            "mean",
            pd.array(
                [Decimal("495025972623655664187232.697686741138")] * len(decimal_arr),
                pd.ArrowDtype(pa.decimal128(38, 12)),
            ),
            id="avg-decimal",
        )
    )
    params.append(
        pytest.param(
            int32_nullable_arr,
            "mean",
            pd.array(
                [49858.6676] * len(int32_nullable_arr), pd.ArrowDtype(pa.float64())
            ),
            id="avg-int32_nullable",
        )
    )
    params.append(
        pytest.param(
            int32_arr_all_null,
            "mean",
            pd.array(
                [None] * len(int32_arr_all_null),
                pd.ArrowDtype(pa.float64()),
            ),
            id="avg-int32_all_null",
        )
    )
    params.append(
        pytest.param(
            int32_numpy_arr,
            "mean",
            pd.array([49458.1416] * len(int32_numpy_arr), pd.ArrowDtype(pa.float64())),
            id="avg-int32_numpy",
        )
    )
    return params


@pytest.mark.parametrize(
    "data, func_name, answer", gen_simple_window_over_nothing_tests()
)
def test_simple_window_over_nothing(data, func_name, answer, memory_leak_check):
    """
    Tests the streaming window code for `F(X) OVER ()`
    """
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(("idx", "win"))
    output_col_order = bodo.utils.typing.MetaType((0, 1))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType((0,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((1,),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            2,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            empty_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            2,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        out_table = bodo.hiframes.table.table_subset(
            window_output, output_col_order, False
        )

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    in_df = pd.DataFrame(
        {
            "idx": pd.array(
                [str(i) for i in range(len(data))],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "data": data,
        }
    )
    out_df = pd.DataFrame(
        {
            "idx": pd.array(
                [str(i) for i in range(len(data))],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "win": answer,
        }
    )

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "kept_input_indices",
    [
        pytest.param([], id="keep_nothing"),
        pytest.param([0], id="keep_idx"),
        pytest.param([1], id="keep_input"),
        pytest.param([0, 1], id="keep_both"),
    ],
)
def test_max_over_nothing_different_kept_inputs(kept_input_indices, memory_leak_check):
    """
    Tests keeping or tossing the input/pass-through columns to a window function.
    """
    output_col_names = ["idx", "data", "win"]
    output_col_names = [
        name
        for idx, name in enumerate(output_col_names)
        if idx in kept_input_indices or name == "win"
    ]

    kept_cols = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(output_col_names))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType(tuple(kept_input_indices))
    func_names = bodo.utils.typing.MetaType(("max",))
    func_input_indices = bodo.utils.typing.MetaType(((1,),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            3,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            empty_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            2,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    in_df = pd.DataFrame(
        {
            "idx": pd.array(
                [str(i) for i in range(1000)], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "data": pd.array(
                [
                    None if i % 7 == 0 else int(str(i).replace("9", "1"))
                    for i in range(1000)
                ],
                dtype=pd.ArrowDtype(pa.int32()),
            ),
        }
    )
    out_df = pd.DataFrame(
        {
            "idx": pd.array(
                [str(i) for i in range(1000)], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "data": pd.array(
                [
                    None if i % 7 == 0 else int(str(i).replace("9", "1"))
                    for i in range(1000)
                ],
                dtype=pd.ArrowDtype(pa.int32()),
            ),
            "win": pd.array(
                [888 for _ in range(1000)], dtype=pd.ArrowDtype(pa.int32())
            ),
        }
    ).loc[:, output_col_names]

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest_mark_one_rank
def test_multi_function_repartition(capfd):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild.
    """
    np.random.seed(543)  # Fix seed for deterministic output on all ranks.

    # We need to keep the data small so that the transient state is relatively
    # larger and we can induce re-partitioning during Finalize.
    n_rows = 32000
    df = pd.DataFrame(
        {
            "P": pd.array(
                [min(i % 4, i % 5) for i in range(n_rows)],
                dtype=pd.ArrowDtype(pa.int32()),
            ),
            "O": pd.array(
                [np.tan(i) for i in range(n_rows)], dtype=pd.ArrowDtype(pa.float64())
            ),
            "S": pd.array(
                [str(i)[2:] for i in range(n_rows)],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
        }
    )

    def window_calculations(part):
        ordered = part.sort_values(by="O", ascending=True)
        row_number = range(1, len(part) + 1)
        bucket_sizes = len(part) // 100
        ntile = [1 + i // bucket_sizes for i in range(len(part))]
        lag = ordered["S"].shift(3, fill_value="fizzbuzz")
        result = pd.DataFrame(
            {
                "O": ordered["O"].values,
                "S": ordered["S"].values,
                "RN": row_number,
                "NT": ntile,
                "LG": lag.values,
            },
            index=ordered.index,
        )
        return result.sort_index()

    expected_out = df.groupby(["P"], as_index=False, dropna=False).apply(
        window_calculations
    )
    expected_output_size = len(expected_out)

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(1, 0), (2, 2), (2, 3)]
    # Verify that we split a partition during FinalizeBuild.
    expected_log_messages = [
        "[DEBUG] GroupbyState::FinalizeBuild: Checking histogram buckets to disable partitioning",
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 0.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Checking histogram buckets to disable partitioning",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Checking histogram buckets to disable partitioning",
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 1.",
        "[DEBUG] Splitting partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Checking histogram buckets to disable partitioning",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Checking histogram buckets to disable partitioning",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 3.",
    ]

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_GROUPBY_ENABLE_PARTITIONING": "1",
            "BODO_STREAM_GROUPBY_MAX_PARTITION_DEPTH": None,
        }
    ):
        # Setup globals for the window function run
        batch_size = 4000
        kept_cols = MetaType((0, 1, 2))
        out_col_meta = ColNamesMetaType(("O", "S", "RN", "NT", "LG"))
        partition_global = bodo.utils.typing.MetaType((0,))
        order_global = bodo.utils.typing.MetaType((1,))
        true_global = bodo.utils.typing.MetaType((True,))
        input_indices = bodo.utils.typing.MetaType((0, 1, 2))
        kept_indices = bodo.utils.typing.MetaType((1, 2))
        func_names = bodo.utils.typing.MetaType(("row_number", "ntile", "lag"))
        func_input_indices = bodo.utils.typing.MetaType(((), (), (2,)))
        func_scalar_args = bodo.utils.typing.MetaType(((), (100,), (3, "fizzbuzz")))

        # Use the streaming window calculations to compute
        # ROW_NUMBER(), NTILE(100), and LAG(S, 3, "fizzbuzz")
        def impl(df, op_pool_size_bytes):
            T1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
                (),
                input_indices,
                3,
            )
            window_state = bodo.libs.streaming.window.init_window_state(
                4001,
                partition_global,
                order_global,
                true_global,
                true_global,
                func_names,
                func_input_indices,
                kept_indices,
                True,
                3,
                func_scalar_args,
                op_pool_size_bytes=op_pool_size_bytes,
            )
            is_last1 = False
            _iter_1 = 0
            _temp1 = bodo.hiframes.table.local_len(T1)
            while not is_last1:
                T2 = bodo.hiframes.table.table_local_filter(
                    T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
                )
                is_last1 = (_iter_1 * batch_size) >= _temp1
                T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
                is_last1, _ = bodo.libs.streaming.window.window_build_consume_batch(
                    window_state, T3, is_last1
                )
                ### Uncomment for debugging purposes ###
                # bytes_pinned = bodo.libs.streaming.groupby.get_op_pool_bytes_pinned(window_state)
                # bytes_allocated = bodo.libs.streaming.groupby.get_op_pool_bytes_allocated(window_state)
                # bodo.parallel_print(
                #     f"Build Iter {_iter_1}: bytes_pinned: {bytes_pinned}, bytes_allocated: {bytes_allocated}"
                # )
                # partition_state = get_partition_state(window_state)
                # bodo.parallel_print(
                #     f"Build Iter {_iter_1} partition_state: ", partition_state
                # )
                ###
                _iter_1 = _iter_1 + 1
            final_partition_state = get_partition_state(window_state)
            is_last2 = False
            _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
            while not is_last2:
                (
                    out_table,
                    is_last2,
                ) = bodo.libs.streaming.window.window_produce_output_batch(
                    window_state, True
                )
                bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            final_bytes_pinned = bodo.libs.streaming.groupby.get_op_pool_bytes_pinned(
                window_state
            )
            final_bytes_allocated = (
                bodo.libs.streaming.groupby.get_op_pool_bytes_allocated(window_state)
            )
            bodo.libs.streaming.window.delete_window_state(window_state)
            out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, out_col_meta
            )
            return (
                df,
                final_partition_state,
                final_bytes_pinned,
                final_bytes_allocated,
            )

        try:
            (
                output,
                final_partition_state,
                final_bytes_pinned,
                final_bytes_allocated,
            ) = bodo.jit(distributed=["df"])(impl)(df, op_pool_size_bytes)
        except Exception:
            out, err = capfd.readouterr()
            with capfd.disabled():
                print(f"STDOUT:\n{out}")
                print(f"STDERR:\n{err}")
            raise

    out, err = capfd.readouterr()

    ### Uncomment for debugging purposes ###
    # with capfd.disabled():
    #     for i in range(bodo.get_size()):
    #         if bodo.get_rank() == i:
    #             print(f"output:\n{output}")
    #             print(f"expected_out:\n{expected_out}")
    #             print(f"stdout:\n{out}")
    #             print(f"stderr:\n{err}")
    #             print(f"final_partition_state: {final_partition_state}")
    ###

    # Verify that the expected log messages are present.
    for expected_log_message in expected_log_messages:
        assert_success = True
        if expected_log_message is not None:
            assert_success = expected_log_message in err
        assert assert_success

    assert output.shape[0] == expected_output_size, (
        f"Final output size ({output.shape[0]}) is not as expected ({expected_output_size})"
    )

    # After the build step, all memory should've been released:
    assert final_bytes_pinned == 0, (
        f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"
    )

    assert final_bytes_allocated == 0, (
        f"Final bytes allocated by the Operator BufferPool ({final_bytes_allocated}) is not 0!"
    )

    assert final_partition_state == expected_partition_state, (
        f"Final partition state ({final_partition_state}) is not as expected ({expected_partition_state})"
    )

    pd.testing.assert_frame_equal(
        output.sort_values(list(expected_out.columns)).reset_index(drop=True),
        expected_out.sort_values(list(expected_out.columns)).reset_index(drop=True),
        check_dtype=False,
        check_index_type=False,
        atol=0.1,
    )


@pytest.mark.parametrize(
    "func_name, order_keys, in_df, out_df",
    [
        pytest.param(
            "row_number",
            (0,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        list(range(1, 1001)), dtype=pd.ArrowDtype(pa.int64())
                    ),
                }
            ),
            id="row_number-integer",
        ),
        pytest.param(
            "rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [i // 10 for i in range(1000)], dtype=pd.ArrowDtype(pa.int64())
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + 10 * (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="rank-integer",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [i // 10 for i in range(1000)], dtype=pd.ArrowDtype(pa.int64())
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-integer",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [f"{(i // 10):02}" for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.large_string()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-strings",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [f"{(i // 10):02}" for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
                    ),
                },
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                },
            ),
            id="dense_rank-dictionary",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [[i // 10, None] for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-array_item",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [
                            {"A": i // 10, "B": 7, "C": ["A", "B", "A"]}
                            for i in range(1000)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("A", pa.int32()),
                                    pa.field("B", pa.int32()),
                                    pa.field(
                                        "C",
                                        pa.large_list(
                                            pa.dictionary(pa.int32(), pa.string())
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-struct",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [{"A": i // 10, "B": 7, "C": None} for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int64())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-map",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [
                            [[], None, [{"A": 0}, {"B": 7 * (i // 10), "C": i // 100}]]
                            for i in range(1000)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.large_list(
                                pa.large_list(pa.map_(pa.large_string(), pa.int16()))
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-multi_nested",
        ),
        pytest.param(
            "dense_rank",
            (1,),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "O": pd.array(
                        [
                            bodo.types.TimestampTZ(
                                pd.Timestamp("2018-10-1")
                                + pd.DateOffset(months=5 * (i // 10)),
                                i,
                            )
                            for i in range(1000)
                        ]
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(1000)), dtype=pd.ArrowDtype(pa.int64())),
                    "OUT": pd.array(
                        [1 + (i // 10) for i in range(1000)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="dense_rank-timestamptz",
        ),
    ],
)
def test_partitionless_rank_family(
    func_name, order_keys, in_df, out_df, memory_leak_check
):
    # Randomize the order of the input data
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(in_df))
    in_df = in_df.iloc[perm, :]

    n_inputs = len(in_df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(n_inputs)))
    col_meta = bodo.utils.typing.ColNamesMetaType(("IDX", "OUT"))
    empty_global = bodo.utils.typing.MetaType(())
    order_global = bodo.utils.typing.MetaType(order_keys)
    true_global = bodo.utils.typing.MetaType(tuple([True] * len(order_keys)))
    kept_indices = bodo.utils.typing.MetaType((0,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            n_inputs,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            empty_global,
            order_global,
            true_global,
            true_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            n_inputs,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "func_name, df, answer",
    [
        pytest.param(
            "sum",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, 1, 1, 8, 8, 8, 8]
                        + [40] * 8
                        + [176] * 16
                        + [736] * 32
                        + [3008] * 64,
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="sum",
        ),
        pytest.param(
            "count",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [0, 1, 1, 2, 2, 2, 2]
                        + [4] * 8
                        + [8] * 16
                        + [16] * 32
                        + [32] * 64,
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="count",
        ),
        pytest.param(
            "count_if",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [
                            [None, False, True, False, None, None, True][
                                min(i % 7, i % 10)
                            ]
                            for i in range(127)
                        ],
                        dtype=pd.ArrowDtype(pa.bool_()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [0, 1, 1, 1, 1, 1, 1]
                        + [2] * 8
                        + [3] * 16
                        + [7] * 32
                        + [14] * 64,
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="count_if",
        ),
        pytest.param(
            "boolor_agg",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else (i - 1) % 3 for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, False, False] + [True] * 124,
                        dtype=pd.ArrowDtype(pa.bool_()),
                    ),
                }
            ),
            id="boolor_agg",
        ),
        pytest.param(
            "booland_agg",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else i % 27 for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None]
                        + [True] * 14
                        + [False] * 16
                        + [True] * 32
                        + [False] * 64,
                        dtype=pd.ArrowDtype(pa.bool_()),
                    ),
                }
            ),
            id="booland_agg",
        ),
        pytest.param(
            "bitand_agg",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 <= i % 3 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, None, None, 3, 3, 3, 3]
                        + [9] * 8
                        + [1] * 16
                        + [33] * 32
                        + [1] * 64,
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            id="bitand_agg",
        ),
        pytest.param(
            "bitor_agg",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 <= i % 3 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, None, None, 3, 3, 3, 3]
                        + [9] * 8
                        + [31] * 16
                        + [63] * 32
                        + [127] * 64,
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            id="bitor_agg",
        ),
        pytest.param(
            "bitxor_agg",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 3 == 0 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, 3, 3, 1, 1, 1, 1]
                        + [13] * 8
                        + [5] * 16
                        + [53] * 32
                        + [21] * 64,
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            id="bitxor_agg",
        ),
        pytest.param(
            "min",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, 1, 1, 3, 3, 3, 3]
                        + [7] * 8
                        + [15] * 16
                        + [31] * 32
                        + [63] * 64,
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="min",
        ),
        pytest.param(
            "max",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [None if i % 2 == 0 else i for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int16()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None, 1, 1, 5, 5, 5, 5]
                        + [13] * 8
                        + [29] * 16
                        + [61] * 32
                        + [125] * 64,
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="max",
        ),
    ],
)
def test_streaming_window_aggfunc_impl(func_name, df, answer, memory_leak_check):
    """
    Tests the streaming window code for simple aggregations on functions
    that support the new sort based implementation.
    """

    kept_cols = bodo.utils.typing.MetaType((0, 1, 2))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "IDX",
            "WIN",
        )
    )
    output_col_order = bodo.utils.typing.MetaType(
        (
            0,
            1,
        )
    )
    partition_global = bodo.utils.typing.MetaType((0,))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType((1,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((2,),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            2,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            partition_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            3,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        out_table = bodo.hiframes.table.table_subset(
            window_output, output_col_order, False
        )

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    check_func(
        impl,
        (df,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
        only_1DVar=True,
    )


@pytest.mark.parametrize(
    "func_name, df, answer",
    [
        pytest.param(
            "first",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "O": pd.array(
                        [np.tan(i) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.float64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [
                            None
                            if i % 2 == 0
                            else datetime.date.fromordinal(738886 + i)
                            for i in range(127)
                        ],
                        dtype=pd.ArrowDtype(pa.date32()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [None] * 3
                        + [datetime.date(2024, 1, 6)] * 4
                        + [datetime.date(2024, 1, 12)] * 8
                        + [None] * 16
                        + [datetime.date(2024, 2, 3)] * 32
                        + [datetime.date(2024, 3, 18)] * 64,
                        dtype=pd.ArrowDtype(pa.date32()),
                    ),
                }
            ),
            id="first",
        ),
        pytest.param(
            "last",
            pd.DataFrame(
                {
                    "P": pd.array(
                        [int(np.log2(i + 1)) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "O": pd.array(
                        [np.tan(i) for i in range(127)],
                        dtype=pd.ArrowDtype(pa.float64()),
                    ),
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "S": pd.array(
                        [
                            None
                            if i % 3 == 1
                            else bodo.types.Time(microsecond=int(1.5**i))
                            for i in range(127)
                        ],
                        dtype=pd.ArrowDtype(pa.time64("us")),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "IDX": pd.array(list(range(127)), dtype=pd.ArrowDtype(pa.int64())),
                    "WIN": pd.array(
                        [bodo.types.Time(microsecond=1)]
                        + [None] * 6
                        + [bodo.types.Time(microsecond=291)] * 8
                        + [bodo.types.Time(microsecond=985)] * 16
                        + [None] * 96,
                        dtype=pd.ArrowDtype(pa.time64("us")),
                    ),
                }
            ),
            id="last",
        ),
    ],
)
def test_streaming_window_value_fn(func_name, df, answer, memory_leak_check):
    """
    Tests the streaming window code for simple aggregations on value functions
    that support the new sort based implementation.
    """

    kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "IDX",
            "WIN",
        )
    )
    output_col_order = bodo.utils.typing.MetaType(
        (
            0,
            1,
        )
    )
    partition_global = bodo.utils.typing.MetaType((0,))
    order_global = bodo.utils.typing.MetaType((1,))
    true_global = bodo.utils.typing.MetaType((True, True))
    kept_indices = bodo.utils.typing.MetaType((2,))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((3,),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            2,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            partition_global,
            order_global,
            true_global,
            true_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            4,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        out_table = bodo.hiframes.table.table_subset(
            window_output, output_col_order, False
        )

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    check_func(
        impl,
        (df,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
        only_1DVar=True,
    )


def test_size_over_nothing(memory_leak_check):
    """
    Tests the streaming window code for `COUNT(*) OVER ()`
    """
    func_name = "size"

    kept_cols = bodo.utils.typing.MetaType((0, 1))
    col_meta = bodo.utils.typing.ColNamesMetaType(("win",))
    output_col_order = bodo.utils.typing.MetaType((2,))
    empty_global = bodo.utils.typing.MetaType(())
    kept_indices = bodo.utils.typing.MetaType(
        (
            0,
            1,
        )
    )
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((),))
    func_scalar_args = bodo.utils.typing.MetaType(((),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            2,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            empty_global,
            empty_global,
            empty_global,
            empty_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            2,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        out_table = bodo.hiframes.table.table_subset(
            window_output, output_col_order, False
        )

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    data = [None] * 10

    in_df = pd.DataFrame(
        {
            "idx": pd.array(
                [str(i) for i in range(len(data))],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "data": pd.array(data, dtype=pd.ArrowDtype(pa.int32())),
        }
    )
    out_df = pd.DataFrame(
        {
            "win": pd.array([len(in_df)] * len(data), dtype=pd.ArrowDtype(pa.int64())),
        }
    )

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


# test window_init with scalar args passed.
def test_ntile(memory_leak_check):
    func_name = "ntile"
    in_df = pd.DataFrame(
        {
            "A": pd.array([1, 1, 1, 2, 2, 2], dtype=pd.ArrowDtype(pa.int64())),
            "B": pd.array([1, 2, 3, 1, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
        }
    )

    n_inputs = len(in_df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(n_inputs)))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "OUT",
        )
    )
    partition_global = bodo.utils.typing.MetaType((0,))
    order_global = bodo.utils.typing.MetaType((1,))
    true_global = bodo.utils.typing.MetaType((True,))
    kept_indices = bodo.utils.typing.MetaType((0, 1))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((),))
    func_scalar_args = bodo.utils.typing.MetaType(((2,),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            n_inputs,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            partition_global,
            order_global,
            true_global,
            true_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            n_inputs,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    out_df = pd.DataFrame(
        {
            "A": pd.array([1, 1, 1, 2, 2, 2], dtype=pd.ArrowDtype(pa.int64())),
            "B": pd.array([1, 2, 3, 1, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
            "OUT": pd.array([1, 1, 2, 1, 1, 2], dtype=pd.ArrowDtype(pa.int64())),
        }
    )

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "default_val, in_col, out_col",
    [
        pytest.param(
            np.int64(-1),
            pd.array([40, 50, 60, 70, 80, 90], dtype=pd.ArrowDtype(pa.int32())),
            pd.array([70, 90, -1, 60, 50, -1], dtype=pd.ArrowDtype(pa.int64())),
            id="upcast_input",
        ),
        pytest.param(
            np.int8(-1),
            pd.array([40, 50, 60, 70, 80, 90], dtype=pd.ArrowDtype(pa.int32())),
            pd.array([70, 90, -1, 60, 50, -1], dtype=pd.ArrowDtype(pa.int32())),
            id="upcast_default",
        ),
        pytest.param(
            None,
            pd.array([40, 50, 60, 70, 80, 90], dtype=pd.ArrowDtype(pa.int32())),
            pd.array([70, 90, None, 60, 50, None], dtype=pd.ArrowDtype(pa.int32())),
            id="null_default",
        ),
        pytest.param(
            None,
            pd.array(
                ["apple", "orange", "banana", "pear", "lemon", "lime"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            pd.array(
                ["pear", "lime", None, "banana", "orange", None],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            id="string-null",
        ),
        pytest.param(
            "hi",
            pd.array(
                ["apple", "orange", "banana", "pear", "lemon", "lime"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            pd.array(
                ["pear", "lime", "hi", "banana", "orange", "hi"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            id="string-dict",
        ),
        pytest.param(
            None,
            pd.array(
                [
                    {
                        "x": i,
                        "y": "hi" + str(i),
                    }
                    for i in range(6)
                ],
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("x", pa.int64()), pa.field("y", pa.string())])
                ),
            ),
            pd.array(
                [
                    (
                        None
                        if name is None
                        else {
                            "x": int(name),
                            "y": "hi" + name,
                        }
                    )
                    for name in ["3", "5", None, "2", "1", None]
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [pa.field("x", pa.int64(), True), pa.field("y", pa.string())]
                    )
                ),
            ),
            id="struct_null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal(str(i)) for i in [40, 50, 60, 70, 80, 90]],
                dtype=pd.ArrowDtype(pa.decimal128(32, 0)),
            ),
            pd.array(
                [
                    None if i == None else Decimal(str(i))
                    for i in [70, 90, None, 60, 50, None]
                ],
                dtype=pd.ArrowDtype(pa.decimal128(32, 0)),
            ),
            id="decimal",
        ),
        pytest.param(
            np.float32(-1),
            pd.array([40, 50, 60, 70, 80, 90], dtype=pd.ArrowDtype(pa.float64())),
            pd.array([70, 90, -1, 60, 50, -1], dtype=pd.ArrowDtype(pa.float64())),
            id="float",
        ),
        pytest.param(
            None,
            pd.array(
                [
                    ["40", "1", "2", "3"],
                    ["50", "2", "3"],
                    ["60", "6", "7"],
                    ["70", "0", "1"],
                    ["80", "2"],
                    ["90", "0"],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
            ),
            pd.array(
                [
                    ["70", "0", "1"],
                    ["90", "0"],
                    None,
                    ["60", "6", "7"],
                    ["50", "2", "3"],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
            ),
            id="array_null",
        ),
        pytest.param(
            None,
            pd.array(
                [{x: x} for x in [40, 50, 60, 70, 80, 90]],
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.int64())),
            ),
            pd.array(
                [None if x is None else {x: x} for x in [70, 90, None, 60, 50, None]],
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.int64())),
            ),
            id="map_null",
        ),
    ],
)
def test_lead_lag(default_val, in_col, out_col, memory_leak_check):
    """
    Test three basic senarios for casting (w/out Nulls):
    1. Cast input table
    2. Cast default val
    3. Cast input table and default val

    SELECT A, B, C, LEAD(C, 1, default_val) OVER (PARTITION BY A ORDER BY B) FROM TABLE
    """
    func_name = "lead"
    in_df = pd.DataFrame(
        {
            "A": pd.array(
                ["A", "B", "A", "A", "B", "B"], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "B": pd.array([1, 5, 4, 2, 3, 6], dtype=pd.ArrowDtype(pa.int64())),
            "C": in_col,
        }
    )

    n_inputs = len(in_df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(n_inputs)))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "OUT",
        )
    )
    partition_global = bodo.utils.typing.MetaType((0,))
    order_global = bodo.utils.typing.MetaType((1,))
    true_global = bodo.utils.typing.MetaType((True,))
    kept_indices = bodo.utils.typing.MetaType((0, 1, 2))
    func_names = bodo.utils.typing.MetaType((func_name,))
    func_input_indices = bodo.utils.typing.MetaType(((2,),))
    func_scalar_args = bodo.utils.typing.MetaType(((1, default_val),))

    def impl(in_df):
        in_table = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(in_df),
            (),
            kept_cols,
            n_inputs,
        )
        window_state = bodo.libs.streaming.window.init_window_state(
            4001,
            partition_global,
            order_global,
            true_global,
            true_global,
            func_names,
            func_input_indices,
            kept_indices,
            True,
            n_inputs,
            func_scalar_args,
        )
        iteration = 0
        local_len = bodo.hiframes.table.local_len(in_table)
        is_last_1 = False
        is_last_2 = False
        while not (is_last_2):
            table_section = bodo.hiframes.table.table_local_filter(
                in_table, slice((iteration * 4096), ((iteration + 1) * 4096))
            )
            is_last_1 = (iteration * 4096) >= local_len
            (
                is_last_2,
                _,
            ) = bodo.libs.streaming.window.window_build_consume_batch(
                window_state, table_section, is_last_1
            )
            iteration = iteration + 1
        is_last_3 = False
        table_builder_state = bodo.libs.table_builder.init_table_builder_state(5001)
        while not (is_last_3):
            (
                window_output_batch,
                is_last_3,
            ) = bodo.libs.streaming.window.window_produce_output_batch(
                window_state, True
            )
            bodo.libs.table_builder.table_builder_append(
                table_builder_state, window_output_batch
            )
        bodo.libs.streaming.window.delete_window_state(window_state)
        window_output = bodo.libs.table_builder.table_builder_finalize(
            table_builder_state
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(window_output), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (window_output,), index_var, col_meta
        )
        return out_df

    out_df = pd.DataFrame(
        {
            "A": pd.array(
                ["A", "B", "A", "A", "B", "B"], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "B": pd.array([1, 5, 4, 2, 3, 6], dtype=pd.ArrowDtype(pa.int64())),
            "C": in_col,
            "OUT": out_col,
        }
    )

    check_func(
        impl,
        (in_df,),
        py_output=out_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )
