import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.libs.stream_join import (
    delete_join_state,
    get_op_pool_bytes_allocated,
    get_op_pool_bytes_pinned,
    get_partition_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import set_broadcast_join, temp_env_override


@pytest.fixture(params=[True, False])
def broadcast(request):
    return request.param


@pytest.fixture(
    params=[
        # (build_outer, probe_outer)
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
)
def build_probe_outer(request):
    return request.param


def hash_join_common_impl(df1, df2, op_pool_size_bytes, build_outer, probe_outer):
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def impl(df1, df2, op_pool_size_bytes):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
            op_pool_size_bytes,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)
            ### Uncomment for debugging purposes ###
            # bytes_pinned = get_op_pool_bytes_pinned(join_state)
            # bytes_allocated = get_op_pool_bytes_allocated(join_state)
            # partition_state = get_partition_state(join_state)
            # bodo.parallel_print(
            #     f"Build Iter {_temp1}: bytes_pinned: {bytes_pinned}, bytes_allocated: {bytes_allocated}"
            # )
            # bodo.parallel_print(f"Build Iter {_temp1} partition_state: ", partition_state)
            ###
            _temp1 = _temp1 + 1

        final_partition_state = get_partition_state(join_state)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

            _temp2 = _temp2 + 1

        final_bytes_pinned = get_op_pool_bytes_pinned(join_state)
        final_bytes_allocated = get_op_pool_bytes_allocated(join_state)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return (
            df,
            final_partition_state,
            final_bytes_pinned,
            final_bytes_allocated,
        )

    # We need a wrapper so that build_outer and probe_outer are treated
    # as globals.
    return bodo.jit(distributed=["df1", "df2"])(impl)(df1, df2, op_pool_size_bytes)


@pytest.mark.skipif(bodo.get_size() > 1, reason="Only calibrated for single core case")
def test_split_during_append_table(build_probe_outer, broadcast, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during AppendBuildBatch on an input batch.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    This test is very specific to the implementation at this point
    and should be replaced with a more granular C++ unit test at some point.
    """
    build_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 2500, dtype="Int64"),
            "B": np.array([1, 2, 3, 4, 5] * 2500, dtype=np.int32),
        }
    )

    probe_df = pd.DataFrame(
        {
            "C": pd.array([2, 6] * 2500, dtype="Int64"),
            "D": np.array([6, 2] * 2500, dtype=np.int8),
        }
    )

    build_outer, probe_outer = build_probe_outer

    expected_partition_state = [(2, 0), (2, 1), (1, 1)] if not broadcast else [(0, 0)]
    expected_output_size = 2500 * 2500

    if build_outer:
        expected_output_size += 4 * 2500  # 4 unmatched keys on the build side
    if probe_outer:
        expected_output_size += 1 * 2500  # 1 unmatched keys on the probe side

    # This will cause partition split during the "AppendBuildBatch"
    # In the broadcast case, we don't want it to re-partition.
    op_pool_size_bytes = 768 * 1024 if not broadcast else 2 * 1024 * 1024
    with set_broadcast_join(broadcast), temp_env_override(
        {
            "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING": "1",
        }
    ):
        (
            output,
            final_partition_state,
            final_bytes_pinned,
            final_bytes_allocated,
        ) = hash_join_common_impl(
            build_df, probe_df, op_pool_size_bytes, build_outer, probe_outer
        )

    assert (
        output.shape[0] == expected_output_size
    ), f"Final output size ({output.shape[0]}) is not as expected ({expected_output_size})"

    # By the time we're done with the probe step, all memory should've been
    # released:
    assert (
        final_bytes_pinned == 0
    ), f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"
    assert (
        final_bytes_allocated == 0
    ), f"Final bytes allocated by the Operator BufferPool ({final_bytes_allocated}) is not 0!"

    assert (
        final_partition_state == expected_partition_state
    ), f"Final partition state ({final_partition_state}) is not as expected ({expected_partition_state})"


@pytest.mark.skipif(bodo.get_size() > 1, reason="Only calibrated for single core case")
def test_split_during_finalize_build(build_probe_outer, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    This test is very specific to the implementation at this point
    and should be replaced with a more granular C++ unit test at some point.
    """
    build_df = pd.DataFrame(
        {
            "A": pd.array(([1, 2] * 2500) + ([34, 67, 35] * 3500), dtype="Int64"),
            "B": np.array(([1, 2] * 2500) + ([3, 4, 5] * 3500), dtype=np.int32),
        }
    )

    probe_df = pd.DataFrame(
        {
            "C": pd.array([2, 6] * 2500, dtype="Int64"),
            "D": np.array([6, 2] * 2500, dtype=np.int8),
        }
    )

    build_outer, probe_outer = build_probe_outer

    expected_partition_state = [(1, 0), (2, 2), (3, 6), (3, 7)]
    expected_output_size = 2500 * 2500
    if build_outer:
        expected_output_size += (1 * 2500) + (
            3 * 3500
        )  # 4 unmatched keys on the build side
    if probe_outer:
        expected_output_size += 2_500  # 1 unmatched keys on the probe side

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 896 * 1024
    with set_broadcast_join(False), temp_env_override(
        {
            "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING": "1",
        }
    ):
        (
            output,
            final_partition_state,
            final_bytes_pinned,
            final_bytes_allocated,
        ) = hash_join_common_impl(
            build_df, probe_df, op_pool_size_bytes, build_outer, probe_outer
        )

    assert (
        output.shape[0] == expected_output_size
    ), f"Final output size ({output.shape[0]}) is not as expected ({expected_output_size})"

    # By the time we're done with the probe step, all memory should've been
    # released:
    assert (
        final_bytes_pinned == 0
    ), f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"
    assert (
        final_bytes_allocated == 0
    ), f"Final bytes allocated by the Operator BufferPool ({final_bytes_allocated}) is not 0!"

    assert (
        final_partition_state == expected_partition_state
    ), f"Final partition state ({final_partition_state}) is not as expected ({expected_partition_state})"


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_split_during_shuffle_append_table_and_diff_part_state(
    build_probe_outer, memory_leak_check
):
    """
    Test that re-partitioning works correctly when it happens
    during AppendBuildBatch on the output of a shuffle operation.
    This test also tests that the overall algorithm works correctly
    and without hangs when the partitioning state is different on
    different ranks. In particular, in this case, rank 0 will end up
    with a single partition, but rank 1 will end up with 3 partitions.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    This test is very specific to the implementation at this point
    and should be replaced with a more granular C++ unit test at some point.
    """

    comm = MPI.COMM_WORLD

    build_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 2500, dtype="Int64"),
            "B": np.array([1, 2, 3, 4, 5] * 2500, dtype=np.int32),
        }
    )

    probe_df = pd.DataFrame(
        {
            "C": pd.array([2, 6] * 2500, dtype="Int64"),
            "D": np.array([6, 2] * 2500, dtype=np.int8),
        }
    )

    build_outer, probe_outer = build_probe_outer

    # Different partitioning state on the two ranks.
    expected_partition_state = (
        [(2, 0), (2, 1), (1, 1)] if bodo.get_rank() == 1 else [(0, 0)]
    )
    expected_output_size = 5000 * 5000
    if build_outer:
        expected_output_size += (
            4 * 2500 * 2
        )  # 4 unmatched keys on the build side (on each rank)
    if probe_outer:
        expected_output_size += (
            1 * 2500 * 2
        )  # 1 unmatched keys on the probe side (on each rank)

    # This will cause partition split during the "AppendBuildBatch"
    # after the shuffle on rank 1.
    op_pool_size_bytes = 1024 * 1024
    with set_broadcast_join(False), temp_env_override(
        {
            "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING": "1",
        }
    ):
        (
            output,
            final_partition_state,
            final_bytes_pinned,
            final_bytes_allocated,
        ) = hash_join_common_impl(
            build_df, probe_df, op_pool_size_bytes, build_outer, probe_outer
        )

    output_size = comm.allreduce(output.shape[0], op=MPI.SUM)
    assert (
        output_size == expected_output_size
    ), f"Final output size ({output_size}) is not as expected ({expected_output_size})"

    # By the time we're done with the probe step, all memory should've been
    # released:
    assert_success = final_bytes_pinned == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert (
        assert_success
    ), f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"

    assert_success = final_bytes_allocated == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert (
        assert_success
    ), f"Final bytes allocated by the Operator BufferPool ({final_bytes_allocated}) is not 0!"

    assert_success = final_partition_state == expected_partition_state
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert (
        assert_success
    ), f"Final partition state ({final_partition_state}) is not as expected ({expected_partition_state})"
