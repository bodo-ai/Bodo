import sys

import numpy as np
import pandas as pd
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
from mpi4py import MPI

import bodo.io.snowflake
import bodo.tests.utils
from bodo.libs.streaming.join import (
    delete_join_state,
    get_op_pool_budget_bytes,
    get_op_pool_bytes_allocated,
    get_op_pool_bytes_pinned,
    get_partition_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import pytest_mark_one_rank, set_broadcast_join, temp_env_override

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32", reason="TODO[BSE-4556]: enable buffer pool on Windows"
    ),
    pytest.mark.slow,
]


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
    build_interval_cols = bodo.utils.typing.MetaType(())

    def impl(df1, df2, op_pool_size_bytes):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
            build_interval_cols,
            False,
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
            is_last1, _ = join_build_consume_batch(join_state, T2, is_last1)
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
        op_pool_budget_after_build = get_op_pool_budget_bytes(join_state)

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
        final_op_pool_budget = get_op_pool_budget_bytes(join_state)
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
            op_pool_budget_after_build,
            final_op_pool_budget,
        )

    # We need a wrapper so that build_outer and probe_outer are treated
    # as globals.
    return bodo.jit(distributed=["df1", "df2"])(impl)(df1, df2, op_pool_size_bytes)


def _test_helper(
    build_df,
    probe_df,
    build_outer,
    probe_outer,
    expected_partition_state,
    expected_output_size,
    op_pool_size_bytes,
    broadcast,
    capfd,
    expected_log_messages,
    expected_op_pool_budget_after_build=None,
):
    """
    Helper for testing.

    expected_op_pool_budget_after_build: If not None, it must be a tuple that the value is expected
     to be in the middle of.
    """
    with (
        set_broadcast_join(broadcast),
        temp_env_override(
            {
                "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
                # Enable partitioning even though spilling is not setup
                "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING": "1",
            }
        ),
    ):
        try:
            (
                output,
                final_partition_state,
                final_bytes_pinned,
                final_bytes_allocated,
                op_pool_budget_after_build,
                final_op_pool_budget,
            ) = hash_join_common_impl(
                build_df, probe_df, op_pool_size_bytes, build_outer, probe_outer
            )
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
    #             print(f"final_bytes_pinned: {final_bytes_pinned}")
    #             print(f"final_op_pool_budget: {final_op_pool_budget}")
    #             print(f"final_bytes_allocated: {final_bytes_allocated}")
    #             print(f"final_partition_state: {final_partition_state}")
    #             print(f"op_pool_budget_after_build: {op_pool_budget_after_build}")
    #             print(f"stdout:\n{out}")
    #             print(f"stderr:\n{err}")
    #         bodo.barrier()
    ###

    comm = MPI.COMM_WORLD

    # Verify that the expected log messages are present.
    for expected_log_message in expected_log_messages:
        assert_success_loc = True
        if expected_log_message is not None:
            assert_success_loc = expected_log_message in err
        assert_success = comm.allreduce(assert_success_loc, op=MPI.LAND)
        assert assert_success, (
            f"Expected log message ('{expected_log_message}') not found in logs!"
            if (not assert_success_loc)
            else "See other rank(s)."
        )

    output_size = comm.allreduce(output.shape[0], op=MPI.SUM)
    assert output_size == expected_output_size, (
        f"Final output size ({output_size}) is not as expected ({expected_output_size})"
    )

    # By the time we're done with the probe step, all memory should've been
    # released:
    assert_success = final_bytes_pinned == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"
    )

    assert_success = final_op_pool_budget == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Final operator pool budget ({final_op_pool_budget}) is not 0!"
    )

    assert_success = final_bytes_allocated == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Final bytes allocated by the Operator BufferPool ({final_bytes_allocated}) is not 0!"
    )

    assert_success = final_partition_state == expected_partition_state
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Final partition state ({final_partition_state}) is not as expected ({expected_partition_state})"
    )

    assert_success = (expected_op_pool_budget_after_build is None) or (
        expected_op_pool_budget_after_build[0]
        <= op_pool_budget_after_build
        <= expected_op_pool_budget_after_build[1]
    )
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Operator pool budget after build ({op_pool_budget_after_build}) is not as expected ({expected_op_pool_budget_after_build})"
    )


@pytest_mark_one_rank
def test_split_during_append_table(
    build_probe_outer, broadcast, memory_leak_check, capfd
):
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
            "A": pd.array([1, 2, 3, 4, 5] * 5000, dtype="Int64"),
            "B": np.array([1, 2, 3, 4, 5] * 5000, dtype=np.int32),
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
    expected_output_size = 5000 * 2500

    if build_outer:
        expected_output_size += 4 * 5000  # 4 unmatched keys on the build side
    if probe_outer:
        expected_output_size += 1 * 2500  # 1 unmatched keys on the probe side

    # This will cause partition split during the "AppendBuildBatch"
    # In the broadcast case, we don't want it to re-partition.
    op_pool_size_bytes = 768 * 1024 if not broadcast else 2 * 1024 * 1024

    expected_log_msgs = []
    expected_op_pool_budget_after_build = None
    if broadcast:
        expected_log_msgs = [
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 0. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 1.",
            "Estimated max partition size: ",
            "Total size of all partitions: ",
            "Estimated required size of Op-Pool: ",
        ]
        # It's between 620KiB and 630KiB depending on build_table_outer
        expected_op_pool_budget_after_build = (620 * 1024, 630 * 1024)
    else:
        expected_log_msgs = [
            "[DEBUG] HashJoinState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
            "[DEBUG] Splitting partition 0.",
            "[DEBUG] HashJoinState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
            "[DEBUG] Splitting partition 0.",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 0. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 1. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 2. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 3.",
            "Estimated max partition size: ",
            "Total size of all partitions: ",
            "Estimated required size of Op-Pool: ",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 1.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 1.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 2.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 2.",
        ]
        # It's between 500KiB and 510KiB depending on build_table_outer
        expected_op_pool_budget_after_build = (500 * 1024, 510 * 1024)

    _test_helper(
        build_df,
        probe_df,
        build_outer,
        probe_outer,
        expected_partition_state,
        expected_output_size,
        op_pool_size_bytes,
        broadcast,
        capfd,
        expected_log_msgs,
        expected_op_pool_budget_after_build,
    )


@pytest_mark_one_rank
def test_split_during_finalize_build(build_probe_outer, memory_leak_check, capfd):
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
            "A": pd.array(([1, 2] * 5000) + ([34, 67, 35] * 7000), dtype="Int64"),
            "B": np.array(([1, 2] * 5000) + ([3, 4, 5] * 7000), dtype=np.int32),
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
    expected_output_size = 5000 * 2500
    if build_outer:
        expected_output_size += (1 * 5000) + (
            3 * 7000
        )  # 4 unmatched keys on the build side
    if probe_outer:
        expected_output_size += 2_500  # 1 unmatched keys on the probe side

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 768 * 1024
    expected_log_msgs = [
        "[DEBUG] HashJoinState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 0. Estimated partition size: ",
        "[DEBUG] HashJoinState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 1.",
        "[DEBUG] Splitting partition 1.",
        "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 1. Estimated partition size: ",
        "[DEBUG] HashJoinState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 2.",
        "[DEBUG] Splitting partition 2.",
        "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 2. Estimated partition size: ",
        "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 3. Estimated partition size: ",
        "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 4. Estimated max partition size: ",
        "Total size of all partitions: ",
        "Estimated required size of Op-Pool: ",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 1.",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 1.",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 2.",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 2.",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 3.",
        "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 3.",
    ]
    # It's between 545KiB and 570KiB depending on build_table_outer
    expected_op_pool_budget_after_build = (545 * 1024, 570 * 1024)

    _test_helper(
        build_df,
        probe_df,
        build_outer,
        probe_outer,
        expected_partition_state,
        expected_output_size,
        op_pool_size_bytes,
        False,
        capfd,
        expected_log_msgs,
        expected_op_pool_budget_after_build,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_split_during_shuffle_append_table_and_diff_part_state(
    build_probe_outer, memory_leak_check, capfd
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

    build_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 5000, dtype="Int64"),
            "B": np.array([1, 2, 3, 4, 5] * 5000, dtype=np.int32),
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
    expected_output_size = 10000 * 5000
    if build_outer:
        expected_output_size += (
            4 * 5000 * 2
        )  # 4 unmatched keys on the build side (on each rank)
    if probe_outer:
        expected_output_size += (
            1 * 2500 * 2
        )  # 1 unmatched keys on the probe side (on each rank)

    # This will cause partition split during the "AppendBuildBatch"
    # after the shuffle on rank 1.
    op_pool_size_bytes = 1024 * 1024

    expected_log_msgs = []
    expected_op_pool_budget_after_build = None
    if bodo.get_rank() == 0:
        expected_log_msgs = [
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 0. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 1. Estimated max partition size: ",
            "Total size of all partitions: ",
            "Estimated required size of Op-Pool: ",
        ] + ([None] * 10)
        # It's between 300KiB and 400KiB depending on build_table_outer
        expected_op_pool_budget_after_build = (300 * 1024, 400 * 1024)
    else:
        expected_log_msgs = [
            "[DEBUG] HashJoinState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
            "[DEBUG] Splitting partition 0.",
            # This is the primary one we're looking for.
            "[DEBUG] HashJoinState::AppendBuildBatch[2]: Encountered OperatorPoolThresholdExceededError.",
            "[DEBUG] Splitting partition 0.",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 0. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 1. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Successfully finalized partition 2. Estimated partition size: ",
            "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 3. Estimated max partition size: ",
            "Total size of all partitions: ",
            "Estimated required size of Op-Pool: ",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 1.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 1.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Starting probe finalization for partition 2.",
            "[DEBUG] HashJoinState::FinalizeProbeForInactivePartitions: Finalized probe for partition 2.",
        ]
        # It's between 600KiB and 700KiB depending on build_table_outer
        expected_op_pool_budget_after_build = (600 * 1024, 700 * 1024)

    _test_helper(
        build_df,
        probe_df,
        build_outer,
        probe_outer,
        expected_partition_state,
        expected_output_size,
        op_pool_size_bytes,
        False,
        capfd,
        expected_log_msgs,
        expected_op_pool_budget_after_build,
    )
