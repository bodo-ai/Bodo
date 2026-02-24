import gc
import sys

import numpy as np
import pandas as pd
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
from mpi4py import MPI

import bodo.io.snowflake
from bodo.libs.streaming.groupby import (
    delete_groupby_state,
    get_op_pool_bytes_allocated,
    get_op_pool_bytes_pinned,
    get_partition_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.memory import default_buffer_pool_bytes_allocated
from bodo.tests.utils import (
    _gather_output,
    _get_dist_arg,
    _test_equal_guard,
    pytest_mark_one_rank,
    temp_env_override,
)
from bodo.tests.utils_jit import reduce_sum

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32", reason="TODO[BSE-4556]: enable buffer pool on Windows"
    ),
    pytest.mark.slow,
]


##################### COMMON HELPERS #####################


def groupby_common_impl(
    df, key_inds_list, func_names, f_in_offsets, f_in_cols, op_pool_size_bytes
):
    keys_inds = bodo.utils.typing.MetaType(tuple(key_inds_list))
    out_col_meta_l = (
        ["key"]
        if (len(key_inds_list) == 1)
        else [f"key_{i}" for i in range(len(key_inds_list))]
    ) + [f"out_{i}" for i in range(len(func_names))]
    out_col_meta = bodo.utils.typing.ColNamesMetaType(tuple(out_col_meta_l))
    len_kept_cols = len(df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(len_kept_cols)))
    batch_size = 4000
    fnames = bodo.utils.typing.MetaType(tuple(func_names))
    f_in_offsets = bodo.utils.typing.MetaType(tuple(f_in_offsets))
    f_in_cols = bodo.utils.typing.MetaType(tuple(f_in_cols))

    def impl(df, op_pool_size_bytes):
        groupby_state = init_groupby_state(
            -1,
            keys_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            op_pool_size_bytes=op_pool_size_bytes,
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            kept_cols,
            len_kept_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
            ### Uncomment for debugging purposes ###
            # bytes_pinned = get_op_pool_bytes_pinned(groupby_state)
            # bytes_allocated = get_op_pool_bytes_allocated(groupby_state)
            # bodo.parallel_print(
            #     f"Build Iter {_iter_1}: bytes_pinned: {bytes_pinned}, bytes_allocated: {bytes_allocated}"
            # )
            # partition_state = get_partition_state(groupby_state)
            # bodo.parallel_print(
            #     f"Build Iter {_iter_1} partition_state: ", partition_state
            # )
            ###
            _iter_1 = _iter_1 + 1

        final_partition_state = get_partition_state(groupby_state)
        is_last2 = False
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        final_bytes_pinned = get_op_pool_bytes_pinned(groupby_state)
        final_bytes_allocated = get_op_pool_bytes_allocated(groupby_state)
        delete_groupby_state(groupby_state)
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

    # We need a wrapper so that fnames, etc. are treated as globals.
    return bodo.jit(distributed=["df"])(impl)(df, op_pool_size_bytes)


def _test_helper(
    df,
    expected_out,
    expected_partition_state,
    expected_output_size,
    key_inds_list,
    func_names,
    f_in_offsets,
    f_in_cols,
    op_pool_size_bytes,
    expected_log_messages,
    capfd,
    multi_rank,
    max_partition_depth=None,
):
    """
    Helper for testing.

    Args:
        multi_rank (bool, optional): Whether this is a
            multi-rank test. If it is, we use allgather
            the output on all ranks before comparing
            with expected output.
    """
    comm = MPI.COMM_WORLD
    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_GROUPBY_ENABLE_PARTITIONING": "1",
            "BODO_STREAM_GROUPBY_MAX_PARTITION_DEPTH": str(max_partition_depth)
            if max_partition_depth is not None
            else None,
        }
    ):
        try:
            (
                output,
                final_partition_state,
                final_bytes_pinned,
                final_bytes_allocated,
            ) = groupby_common_impl(
                _get_dist_arg(df) if multi_rank else df,
                key_inds_list,
                func_names,
                f_in_offsets,
                f_in_cols,
                op_pool_size_bytes,
            )
            if multi_rank:
                global_output = bodo.allgatherv(output)
            else:
                global_output = output
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
    #         bodo.barrier()
    ###

    # Verify that the expected log messages are present.
    for expected_log_message in expected_log_messages:
        assert_success = True
        if expected_log_message is not None:
            assert_success = expected_log_message in err
        assert_success = comm.allreduce(assert_success, op=MPI.LAND)
        assert assert_success

    assert global_output.shape[0] == expected_output_size, (
        f"Final output size ({global_output.shape[0]}) is not as expected ({expected_output_size})"
    )

    # After the build step, all memory should've been released:
    assert_success = final_bytes_pinned == 0
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)
    assert assert_success, (
        f"Final bytes pinned by the Operator BufferPool ({final_bytes_pinned}) is not 0!"
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

    pd.testing.assert_frame_equal(
        global_output.sort_values(
            "key"
            if (len(key_inds_list) == 1)
            else [f"key_{i}" for i in range(len(key_inds_list))]
        ).reset_index(drop=True),
        expected_out.sort_values(
            "key"
            if (len(key_inds_list) == 1)
            else [f"key_{i}" for i in range(len(key_inds_list))]
        ).reset_index(drop=True),
        check_dtype=False,
        check_index_type=False,
        atol=0.1,
    )


##########################################################

########### TESTS FOR ACCUMULATE INPUT STRATEGY ##########


@pytest_mark_one_rank
def test_split_during_append_table_acc_funcs(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during AppendBuildBatch on an input batch.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    In particular, we use functions that always go through Accumulate
    path regardless of the dtypes of the running values.
    """

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    func_names = ["median", "sum", "nunique"]
    f_in_offsets = [0, 1, 2, 3]
    f_in_cols = [
        1,
        1,
        2,
    ]
    expected_out = df.groupby("A", as_index=False).agg(
        {"B": ["median", "sum"], "C": ["nunique"]}
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(3)]
    expected_output_size = 6

    # This will cause partition split during the "AppendBuildBatch[3]"
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(2, 0), (2, 1), (1, 1)]

    # Verify that we split a partition during AppendBuildBatch.
    expected_log_msg = "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_split_during_append_table_str_running_vals(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during AppendBuildBatch on an input batch.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    In particular, we use functions that usually go through the
    Aggregation path, but won't because one or more running values
    are STRING/DICT
    """

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
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
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
            "C": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "D": np.arange(32000, dtype=np.int32),
        }
    )
    func_names = [
        "max",
        "min",
        "sum",
        "count",
        "mean",
        "var",
        "std",
        "kurtosis",
        "skew",
    ]
    f_in_cols = [1, 1, 2, 2, 2, 2, 2, 3, 3]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["max", "min"],
            "C": ["sum", "count", "mean", "var", "std"],
            "D": [pd.Series.kurt, "skew"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(9)]
    expected_output_size = 6

    # This will cause partition split during the "AppendBuildBatch[3]"
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(2, 0), (2, 1), (1, 1)]

    # Verify that we split a partition during AppendBuildBatch.
    expected_log_msg = "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_split_during_acc_finalize_build_acc_funcs(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild.
    In particular, we use functions that always go through Accumulate
    path regardless of the dtypes of the running values.
    To actually be able to invoke repartitioning during FinalizeBuild
    (which is not easy since only the main table buffer allocates from
    the main mem portion and all the rest goes through the scratch mem
    portion), we need to output a lot of columns with a lot of unique
    groups.
    """

    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(4000)) * 16, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 4000,
                dtype=np.float32,
            ),
            "C": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 8000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 2000
            ),
        }
    )
    func_names = [
        "median",
        "max",
        "min",
        "sum",
        "count",
        "mean",
        "var",
        "std",
        "kurtosis",
        "skew",
        "nunique",
        "sum",
    ]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    f_in_cols = ([1] * 10) + ([2] * 2)
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": [
                "median",
                "max",
                "min",
                "sum",
                "count",
                "mean",
                "var",
                "std",
                pd.Series.kurt,
                "skew",
            ],
            "C": ["nunique", "sum"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(len(func_names))]
    expected_output_size = 4000

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 1.5 * 1024 * 1024
    expected_partition_state = [(3, 0), (3, 1), (2, 1), (2, 2), (2, 3)]
    # Verify that we split a partition during FinalizeBuild.
    expected_log_msgs = [
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        # This is the main one we're looking for:
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 3.",
        "[DEBUG] Splitting partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 4.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 5.",
    ]

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_split_during_acc_finalize_build_str_running_vals(
    capfd,
    # TODO Need to find and fix the memory leak
    # (https://bodo.atlassian.net/browse/BSE-2271)
    # memory_leak_check,
):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild.
    In particular, we use functions that usually go through the
    Aggregation path, but won't because one or more running values
    are STRING/DICT
    """

    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(4000)) * 16, dtype="Int64"),
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
                * 8000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 2000
            ),
            "C": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 4000,
                dtype=np.float32,
            ),
            "D": np.arange(64000, dtype=np.int32),
        }
    )
    func_names = [
        "max",
        "min",
        "sum",
        "sum",
        "count",
        "mean",
        "var",
        "std",
        "kurtosis",
        "skew",
        "sum",
        "count",
        "mean",
        "var",
        "std",
        "kurtosis",
        "skew",
    ]
    f_in_cols = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["max", "min", "sum"],
            "C": ["sum", "count", "mean", "var", "std", pd.Series.kurt, "skew"],
            "D": ["sum", "count", "mean", "var", "std", pd.Series.kurt, "skew"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(len(func_names))]
    expected_output_size = 4000

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(2, 0), (2, 1), (2, 2), (2, 3)]

    # Verify that we split a partition during FinalizeBuild.
    expected_log_msgs = [
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 2.",
        "[DEBUG] Splitting partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 4.",
    ]

    # memory_leak_check seems to indicate that we don't
    # "free" an allocation (both the meminfo struct and the
    # underlying allocation). We have confirmed that it's
    # not an actual leak, i.e. no bytes from the buffer pool
    # are leaked. Until we figure out the MemInfo leak
    # (https://bodo.atlassian.net/browse/BSE-2271), we
    # will just verify that there's no bytes leaking from the
    # buffer pool itself.

    # Ensure that all unused allocations are free-d before
    # measuring bytes_allocated.
    gc.collect()
    bytes_before = default_buffer_pool_bytes_allocated()
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
    )
    # Ensure that all unused allocations are free-d before
    # measuring bytes_allocated.
    gc.collect()
    bytes_after = default_buffer_pool_bytes_allocated()
    assert bytes_before == bytes_after, (
        f"Potential memory leak! bytes_before ({bytes_before}) != bytes_after ({bytes_after})"
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_split_during_shuffle_append_table_and_diff_part_state(
    capfd, memory_leak_check
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
    """

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 6400, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    expected_out = df.groupby("A", as_index=False).agg(
        {"B": ["median", "sum"], "C": ["nunique"]}
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(3)]

    expected_partition_state = [(0, 0)] if (bodo.get_rank() == 0) else [(1, 0), (1, 1)]
    expected_output_size = 5
    func_names = ["median", "sum", "nunique"]
    f_in_offsets = [0, 1, 2, 3]
    f_in_cols = [1, 1, 2]

    # This will cause partition split during the "AppendBuildBatch[2]"
    op_pool_size_bytes = 1.5 * 1024 * 1024

    # Verify that we split a partition during AppendBuildBatch.
    expected_log_msg = (
        "[DEBUG] GroupbyState::AppendBuildBatch[2]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
        if bodo.get_rank() == 1
        else None
    )

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        True,
    )


@pytest_mark_one_rank
def test_max_partition_depth_fallback_acc_finalize(capfd, memory_leak_check):
    """
    Test that we fall back to disabling partitioning while
    finalizing partitions at max depth. This primarily tests
    that the expected warnings are printed in debug mode.
    We cannot really test for the case where the OOM killer
    is invoked in a unit test, so we just test for the case
    where it succeeds.
    """
    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(1000)) * 32, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )
    func_names = ["median", "sum", "nunique"]
    f_in_offsets = [0, 1, 2, 3]
    f_in_cols = [
        1,
        1,
        2,
    ]
    expected_out = df.groupby("A", as_index=False).agg(
        {"B": ["median", "sum"], "C": ["nunique"]}
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(3)]
    expected_output_size = 1000

    # This will cause partition split during the "FinalizeBuild"
    # and would usually lead to 3 partitions. Setting max partition
    # depth of 1 force just 2 partitions at most.
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(1, 0), (1, 1)]
    max_partition_depth = 1
    # Verify that we split a partition during FinalizeBuild and we log
    # the expected warnings.
    expected_log_msgs = [
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] WARNING: Disabling partitioning and threshold enforcement temporarily to finalize partition 0 which is at max allowed partition depth (1). This may invoke the OOM killer.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] WARNING: Disabling partitioning and threshold enforcement temporarily to finalize partition 1 which is at max allowed partition depth (1). This may invoke the OOM killer.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 2.",
    ]

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
        max_partition_depth,
    )


##########################################################

####### TESTS FOR INCREMENTAL AGGREGATION STRATEGY #######


@pytest_mark_one_rank
def test_split_during_update_combine(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during UpdateGroupsAndCombine (AGG path) on an input batch.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(8000, dtype=np.int32)) * 4, dtype=np.int32),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "D": np.arange(32000, 64000, dtype=np.float64),
            "E": ([True] * 5) + ([False] * (32000 - 5)),
            "F": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    func_names = [
        "sum",
        "mean",
        "min",
        "max",
        "skew",
        "kurtosis",
        "count",
        "var",
        "std",
        "boolxor_agg",
        "count",
    ]
    f_in_cols = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["sum", "mean"],
            "C": ["min", "max", "skew", pd.Series.kurt, "count"],
            "D": ["var", "std"],
            "E": [(lambda x: sum(x) == 1)],
            "F": ["count"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(11)]
    expected_output_size = 8000

    # This will cause partition split during the "UpdateGroupsAndCombine[4]"
    op_pool_size_bytes = 4 * 1024 * 1024
    expected_partition_state = [(1, 0), (2, 2), (2, 3)]

    # Verify that we split a partition during UpdateGroupsAndCombine.
    expected_log_msg = "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_drop_duplicates_split_during_update_combine(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly for drop_duplicates
    when it happens during UpdateGroupsAndCombine (AGG path) on
    an input batch. We trigger this by using specific key column
    values, array sizes and the size of the operator pool.
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(8000, dtype=np.int32)) * 4, dtype=np.int32),
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
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )
    key_inds_list = [0, 1]
    func_names = []
    f_in_cols = []
    f_in_offsets = [0]

    expected_out = df.drop_duplicates()
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key_0", "key_1"]
    expected_output_size = 32000

    # This will cause partition split during the "UpdateGroupsAndCombine[4]"
    op_pool_size_bytes = 2 * 1024 * 1024
    expected_partition_state = [(1, 0), (1, 1)]

    # Verify that we split a partition during UpdateGroupsAndCombine.
    expected_log_msg = "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_split_during_agg_finalize(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild in the AGG case. Specifically, this
    will occur while activating inactive partitions.
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(8000, dtype=np.int32)) * 4, dtype=np.int32),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "D": np.arange(32000, 64000, dtype=np.float64),
            "E": ([True] * 5) + ([False] * (32000 - 5)),
            "F": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    func_names = [
        "sum",
        "mean",
        "min",
        "max",
        "skew",
        "kurtosis",
        "count",
        "var",
        "std",
        "boolxor_agg",
        "count",
    ]
    f_in_cols = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["sum", "mean"],
            "C": ["min", "max", "skew", pd.Series.kurt, "count"],
            "D": ["var", "std"],
            "E": [(lambda x: sum(x) == 1)],
            "F": ["count"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(11)]
    expected_output_size = 8000

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 3 * 1024 * 1024
    expected_partition_state = [(1, 0), (2, 2), (2, 3)]

    # Verify that we split a (inactive) partition during FinalizeBuild (during ActivatePartition).
    expected_log_msgs = [
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        # This is the main one we're looking for:
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 1.",
        "[DEBUG] Splitting partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 3.",
    ]
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
    )


@pytest_mark_one_rank
def test_drop_duplicates_split_during_agg_finalize(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly for drop_duplicates
    when it happens during FinalizeBuild in the AGG case.
    Specifically, this will occur while activating inactive
    partitions.
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(16000, dtype=np.int32)) * 2, dtype=np.int32),
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
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )
    key_inds_list = [0, 1]
    func_names = []
    f_in_cols = []
    f_in_offsets = [0]

    expected_out = df.drop_duplicates()
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key_0", "key_1"]
    expected_output_size = 32000

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 768 * 1024
    expected_partition_state = [(2, 0), (2, 1), (2, 2), (2, 3)]

    # Verify that we split (inactive) partitions during FinalizeBuild (during ActivatePartition).
    expected_log_msgs = [
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        # This is the main one we're looking for:
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 2.",
        "[DEBUG] Splitting partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 4.",
    ]

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_spilt_during_shuffle_out_update_combine_and_diff_part_state(
    capfd, memory_leak_check
):
    """
    Test that re-partitioning works correctly when it happens
    during UpdateGroupsAndCombine on the output of a shuffle operation.
    This test also tests that the overall algorithm works correctly
    and without hangs when the partitioning state is different on
    different ranks. In particular, in this case, rank 0 will end up
    with a single partition, but rank 1 will end up with 3 partitions.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(8000, dtype=np.int32)) * 4, dtype=np.int32),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "D": np.arange(32000, 64000, dtype=np.float64),
            "E": ([True] * 5) + ([False] * (32000 - 5)),
            "F": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    func_names = [
        "sum",
        "mean",
        "min",
        "max",
        "skew",
        "kurtosis",
        "count",
        "var",
        "std",
        "boolxor_agg",
        "count",
    ]
    f_in_cols = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["sum", "mean"],
            "C": ["min", "max", "skew", pd.Series.kurt, "count"],
            "D": ["var", "std"],
            "E": [(lambda x: sum(x) == 1)],
            "F": ["count"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(11)]
    expected_output_size = 8000

    # This will cause partition split during the "UpdateGroupsAndCombine[3]"
    op_pool_size_bytes = 4 * 1024 * 1024
    expected_partition_state = (
        [(0, 0)] if (bodo.get_rank() == 0) else [(2, 0), (2, 1), (1, 1)]
    )

    # Verify that we split a partition during UpdateGroupsAndCombine on the shuffle output.
    expected_log_msg = (
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[3]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
        if bodo.get_rank() == 1
        else None
    )
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        True,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
def test_drop_duplicates_spilt_during_shuffle_out_update_combine_and_diff_part_state(
    capfd, memory_leak_check
):
    """
    Test that re-partitioning works correctly for drop_duplicates when it happens
    during UpdateGroupsAndCombine on the output of a shuffle operation.
    This test also tests that the overall algorithm works correctly
    and without hangs when the partitioning state is different on
    different ranks. In particular, in this case, rank 0 will end up
    with 2 partitions, but rank 1 will end up with 5 partitions.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool (diff on diff ranks).
    """
    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(8000, dtype=np.int32)) * 4, dtype=np.int32),
            "B": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza" * 100,
                    "omelette",
                    "salad",
                    "spinach" * 100,
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )
    key_inds_list = [0, 1]
    func_names = []
    f_in_cols = []
    f_in_offsets = [0]

    expected_out = df.drop_duplicates()
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key_0", "key_1"]
    expected_output_size = 32000

    # This will cause partition split during the "UpdateGroupsAndCombine[3]" on rank-1.
    # We need to set different pool sizes on the different ranks because otherwise
    # it's hard to invoke partition split on one rank and not the other.
    op_pool_size_bytes = 4 * 1024 * 1024 if bodo.get_rank() == 1 else 8 * 1024 * 1024
    expected_partition_state = (
        [(3, 0), (3, 1), (2, 1), (1, 1)] if bodo.get_rank() == 1 else [(1, 0), (1, 1)]
    )

    # Verify that we split a partition during UpdateGroupsAndCombine on the
    # shuffle output on rank 1.
    # On rank 0, we split during UpdateGroupsAndCombine on an input batch.
    expected_log_msg = (
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[3]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
        if bodo.get_rank() == 1
        else "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0."
    )

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        [expected_log_msg],
        capfd,
        True,
    )


@pytest_mark_one_rank
def test_max_partition_depth_fallback_agg_finalize(capfd, memory_leak_check):
    """
    Test that we fall back to disabling partitioning while
    finalizing partitions at max depth. This primarily tests
    that the expected warnings are printed in debug mode.
    We cannot really test for the case where the OOM killer
    is invoked in a unit test, so we just test for the case
    where it succeeds.
    """

    df = pd.DataFrame(
        {
            "A": np.array(list(np.arange(16000, dtype=np.int32)) * 2, dtype=np.int32),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
            "C": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "D": np.arange(32000, 64000, dtype=np.float64),
            "E": ([True] * 5) + ([False] * (32000 - 5)),
            "F": pd.array(
                [
                    "tapas",
                    "bravas",
                    "pizza",
                    "omelette",
                    "salad",
                    "spinach",
                    "celery",
                ]
                * 4000
                + ["sandwich", "burrito", "ramen", "carrot-cake"] * 1000
            ),
        }
    )

    func_names = [
        "sum",
        "mean",
        "min",
        "max",
        "skew",
        "kurtosis",
        "count",
        "var",
        "std",
        "boolxor_agg",
        "count",
    ]
    f_in_cols = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5]
    f_in_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_out = df.groupby("A", as_index=False).agg(
        {
            "B": ["sum", "mean"],
            "C": ["min", "max", "skew", pd.Series.kurt, "count"],
            "D": ["var", "std"],
            "E": [(lambda x: sum(x) == 1)],
            "F": ["count"],
        }
    )
    expected_out.reset_index(inplace=True, drop=True)
    expected_out.columns = ["key"] + [f"out_{i}" for i in range(11)]
    expected_output_size = 16000

    # This will cause partition split during the "FinalizeBuild".
    op_pool_size_bytes = 4 * 1024 * 1024
    expected_partition_state = [(3, 0), (3, 1), (2, 1), (2, 2), (2, 3)]
    max_partition_depth = 3

    # Verify that we split (inactive) partitions during FinalizeBuild (during ActivatePartition)
    # and that we display the expected warnings.
    expected_log_msgs = [
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::UpdateGroupsAndCombine[4]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] WARNING: Disabling partitioning and threshold enforcement temporarily to finalize partition 0 which is at max allowed partition depth (3). This may invoke the OOM killer.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] WARNING: Disabling partitioning and threshold enforcement temporarily to finalize partition 1 which is at max allowed partition depth (3). This may invoke the OOM killer.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
        # This verifies that partitioning was re-enabled successfully after disabling it temporarily
        # for finalizing partitions 0 and 1.
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 3.",
        "[DEBUG] Splitting partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 3.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 4.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 5.",
    ]
    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        [0],
        func_names,
        f_in_offsets,
        f_in_cols,
        op_pool_size_bytes,
        expected_log_msgs,
        capfd,
        False,
        max_partition_depth,
    )


##########################################################


######## TESTS FOR STREAMING WINDOW SKEW HANDLING ########

WINDOW_PARTITION_ROWS = 16 * 1024
WINDOW_BYTES_PER_ROW = 8 * 4
WINDOW_PARTITION_SIZE = WINDOW_PARTITION_ROWS * WINDOW_BYTES_PER_ROW
WINDOW_PARTITION_BUDGET = (WINDOW_PARTITION_SIZE * 7) // 2

import numpy as np
import pandas as pd

import bodo
from bodo.utils.typing import ColNamesMetaType, MetaType

global_2 = MetaType((0,))
global_3 = MetaType((1,))
global_1 = MetaType((0, 1, 2, 3))
global_6 = MetaType((0, 2, 3))
global_7 = ColNamesMetaType(("A", "C", "D", "RANK"))
global_4 = MetaType((True,))
global_5 = MetaType(("rank",))
global_8 = MetaType(((),))


@bodo.jit(distributed=["df"])
def window_skew_impl(df):
    T1 = bodo.hiframes.table.logical_table_to_table(
        bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), global_1, 4
    )
    T2 = T1
    __bodo_is_last_streaming_output_1 = False
    _iter_1 = 0
    _temp1 = bodo.hiframes.table.local_len(T2)
    state_1 = bodo.libs.streaming.window.init_window_state(
        4001,
        global_2,
        global_3,
        global_4,
        global_4,
        global_5,
        global_6,
        True,
        4,
        global_8,
        op_pool_size_bytes=WINDOW_PARTITION_BUDGET,
    )
    __bodo_is_last_streaming_output_2 = False
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
    __bodo_is_last_streaming_output_3 = False
    _produce_output_1 = True
    __bodo_streaming_batches_table_builder_1 = (
        bodo.libs.table_builder.init_table_builder_state(5001)
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
    T7 = bodo.hiframes.table.table_subset(T6, global_1, False)
    index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
    df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index_1, global_7)
    bodo.libs.query_profile_collector.finalize()
    return df2


@pytest.mark.skip(
    reason="[BSE-3610] We need a new way to test reparitioning that doesn't involve rank (which now takes a sort path)"
)
@pytest_mark_one_rank
def test_rank_skew_repartition(capfd):
    """
    Test that we avoid repartitioning unnecessarily when we
    have a large partition. This test is very sensitive to the associated
    data. With the histogram based partitioning disabled rank 0 should have
    15 partitions, but with it there are only 2 partitions.
    """
    # memory_leak_check seems to indicate that we don't
    # "free" an allocation (both the meminfo struct and the
    # underlying allocation). We have confirmed that it's
    # not an actual leak, i.e. no bytes from the buffer pool
    # are leaked. Until we figure out the MemInfo leak
    # (https://bodo.atlassian.net/browse/BSE-2271), we
    # will just verify that there's no bytes leaking from the
    # buffer pool itself.

    df = pd.DataFrame(
        {
            "A": np.array(
                [7790324] * (2 * WINDOW_PARTITION_ROWS) + (list(range(0, 5))),
                dtype=np.int64,
            ),
            "B": np.arange(0, (2 * WINDOW_PARTITION_ROWS) + 5),
            "C": np.arange(0, (2 * WINDOW_PARTITION_ROWS) + 5),
            "D": np.arange(0, (2 * WINDOW_PARTITION_ROWS) + 5),
        }
    )
    py_output = pd.DataFrame(
        {
            "A": df["A"],
            "C": df["C"],
            "D": df["D"],
            "RANK": list(range(1, (2 * WINDOW_PARTITION_ROWS) + 1)) + ([1] * 5),
        }
    )
    df = _get_dist_arg(df)
    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
            # Enable partitioning even though spilling is not setup
            "BODO_STREAM_GROUPBY_ENABLE_PARTITIONING": "1",
        }
    ):
        # Ensure that all unused allocations are free-d before
        # measuring bytes_allocated.
        gc.collect()
        bytes_before = default_buffer_pool_bytes_allocated()
        output_df = window_skew_impl(df)
        # Ensure that all unused allocations are free-d before
        # measuring bytes_allocated.
        gc.collect()
        bytes_after = default_buffer_pool_bytes_allocated()
        output_df = _gather_output(output_df)
        passed = 1
        if bodo.get_rank() == 0:
            passed = _test_equal_guard(
                output_df,
                py_output,
                sort_output=True,
                reset_index=True,
            )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()
        assert bytes_before == bytes_after, (
            f"Potential memory leak! bytes_before ({bytes_before}) != bytes_after ({bytes_after})"
        )

    output, err = capfd.readouterr()
    # Uncomment to view the output for debugging.
    # with capfd.disabled():
    #     for i in range(bodo.get_size()):
    #         if bodo.get_rank() == i:
    #             print(f"stdout:\n{output}")
    #             print(f"stderr:\n{err}")
    #         bodo.barrier()

    expected_log_msgs = [
        "[DEBUG] WARNING: Disabling partitioning and threshold enforcement temporarily to finalize partition 1 which is determined based on the histogram to retain at least 90% of its data after repartitioning. This may invoke the OOM killer.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 2.",
    ]
    # Verify that the expected log messages are present.
    comm = MPI.COMM_WORLD
    for expected_log_message in expected_log_msgs:
        assert_success = True
        if expected_log_message is not None:
            assert_success = expected_log_message in err
        assert_success = comm.allreduce(assert_success, op=MPI.LAND)
        assert assert_success


##########################################################
