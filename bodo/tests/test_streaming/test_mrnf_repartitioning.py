import sys

import numpy as np
import pandas as pd
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
from mpi4py import MPI

from bodo.libs.streaming.groupby import (
    delete_groupby_state,
    get_op_pool_bytes_allocated,
    get_op_pool_bytes_pinned,
    get_partition_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.tests.utils import _get_dist_arg, pytest_mark_one_rank, temp_env_override
from bodo.utils.typing import ColNamesMetaType, MetaType

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32", reason="TODO[BSE-4556]: enable buffer pool on Windows"
    ),
    pytest.mark.slow,
]


##################### COMMON HELPERS #####################


def mrnf_common_impl(
    df,
    key_inds_list,
    sort_inds_list,
    sort_asc_list,
    sort_na_list,
    keep_inds_list,
    op_pool_size_bytes,
):
    keys_inds = bodo.utils.typing.MetaType(tuple(key_inds_list))
    fnames = MetaType(("min_row_number_filter",))
    n_cols = len(df.columns)
    batch_size = 4000
    f_in_cols_list = [i for i in range(n_cols) if i not in key_inds_list]
    f_in_cols = MetaType(tuple(f_in_cols_list))
    f_in_offsets = MetaType((0, n_cols - len(key_inds_list)))
    mrnf_sort_col_inds = MetaType(tuple(sort_inds_list))
    mrnf_sort_col_asc = MetaType(tuple(sort_asc_list))
    mrnf_sort_col_na = MetaType(tuple(sort_na_list))
    mrnf_col_inds_keep = MetaType(tuple(keep_inds_list))
    kept_cols = MetaType(tuple(range(n_cols)))
    len_kept_cols = n_cols
    out_col_meta = ColNamesMetaType(tuple(np.take(list(df.columns), keep_inds_list)))

    def impl(df, op_pool_size_bytes):
        groupby_state = init_groupby_state(
            -1,
            keys_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
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
    sort_inds_list,
    sort_asc_list,
    sort_na_list,
    keep_inds_list,
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
            ) = mrnf_common_impl(
                _get_dist_arg(df) if multi_rank else df,
                key_inds_list,
                sort_inds_list,
                sort_asc_list,
                sort_na_list,
                keep_inds_list,
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
    #             print(f"final_partition_state: {final_partition_state}")
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
        global_output.sort_values(list(expected_out.columns)).reset_index(drop=True),
        expected_out.sort_values(list(expected_out.columns)).reset_index(drop=True),
        check_dtype=False,
        check_index_type=False,
        atol=0.1,
    )


##########################################################


@pytest_mark_one_rank
def test_split_during_append_table(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during AppendBuildBatch on an input batch.
    We trigger this by using specific key column values, array
    sizes and the size of the operator pool.
    """
    np.random.seed(543)  # Fix seed for deterministic output on all ranks.
    df = pd.DataFrame(
        {
            "A": pd.Series([1] * 32000, dtype="Int64"),
            "B": pd.array([1, 2, 3, 4, 5, 6, 5, 4] * 4000, dtype="Int64"),
            "C": pd.Series(
                np.random.choice(
                    [
                        pd.NA,
                        "apple",
                        "pie",
                        "egg",
                        "salad",
                        "banana",
                        "tapas",
                        "bravas",
                        "pizza",
                        "omelette",
                        "salad",
                        "spinach",
                        "celery",
                    ],
                    32000,
                )
            ),
            "D": pd.Series(["abc"] * 32000),
            "E": pd.Series(
                np.random.choice(
                    [
                        pd.NA,
                        4.5,
                        23.43,
                        67.64,
                        234.0,
                        0.32,
                        -4.43,
                        -10.98,
                        6543.32,
                        -98.43,
                    ],
                    32000,
                ),
                dtype="Float64",
            ),
        }
    )

    key_inds_list = [1]
    sort_inds_list = [2, 4]
    sort_asc_list = [True, False]
    sort_na_list = [False, True]
    keep_inds_list = [0, 1, 3, 4]

    @bodo.jit(distributed=False)
    def py_mrnf(x: pd.DataFrame, asc: list[bool], na: list[bool]):
        return x.sort_values(by=["C", "E"], ascending=asc, na_position=na)

    expected_out = df.groupby(["B"], as_index=False, dropna=False).apply(
        lambda x: py_mrnf(
            x,
            sort_asc_list,
            [("last" if f else "first") for f in sort_na_list],
        ).iloc[0]
    )
    expected_out = expected_out[list(np.take(list(df.columns), keep_inds_list))]
    expected_output_size = expected_out.shape[0]

    # This will cause partition split during the "AppendBuildBatch[3]"
    op_pool_size_bytes = 3 * 1024 * 1024
    expected_partition_state = [(2, 0), (2, 1), (1, 1)]
    # Verify that we split a partition during AppendBuildBatch.
    expected_log_messages = [
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::AppendBuildBatch[3]: Encountered OperatorPoolThresholdExceededError.",
        "[DEBUG] Splitting partition 0.",
    ]

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        sort_inds_list,
        sort_asc_list,
        sort_na_list,
        keep_inds_list,
        op_pool_size_bytes,
        expected_log_messages,
        capfd,
        multi_rank=False,
    )


@pytest_mark_one_rank
def test_split_during_acc_finalize_build(capfd, memory_leak_check):
    """
    Test that re-partitioning works correctly when it happens
    during FinalizeBuild.
    """
    np.random.seed(543)  # Fix seed for deterministic output on all ranks.

    # We need to keep the data small so that the transient state is relatively
    # larger and we can induce re-partitioning during Finalize.
    # Inducing re-partitioning is especially hard in the MRNF case since
    # the only allocations are:
    # - Hash-map, etc. to figure out the grouping information.
    # - A uint64 array to keep track of the output index of each group
    # The output is directly written to the output_buffer from the
    # build-table, so there are no allocations from the Operator Pool
    # there.
    df = pd.DataFrame(
        {
            "A": pd.Series([True] * 32000, dtype="bool"),
            "B": pd.array(list(np.arange(16000)) * 2, dtype="Int32"),
            "C": pd.Series(np.random.choice([1, 2, 3, 4, 5, 6], 32000), dtype="int32"),
        }
    )

    key_inds_list = [1]
    sort_inds_list = [2]
    sort_asc_list = [True]
    sort_na_list = [False]
    keep_inds_list = [0, 1, 2]

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by=["C"], ascending=True, na_position="first").iloc[0]

    expected_out = df.groupby(["B"], as_index=False, dropna=False).apply(py_mrnf)
    expected_out = expected_out[list(np.take(list(df.columns), keep_inds_list))]
    expected_output_size = expected_out.shape[0]

    # This will cause partition split during the "FinalizeBuild"
    op_pool_size_bytes = 1 * 1024 * 1024
    expected_partition_state = [(1, 0), (1, 1)]
    # Verify that we split a partition during FinalizeBuild.
    expected_log_messages = [
        "[DEBUG] GroupbyState::FinalizeBuild: Encountered OperatorPoolThresholdExceededError while finalizing partition 0.",
        "[DEBUG] Splitting partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
        "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 2.",
    ]

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        sort_inds_list,
        sort_asc_list,
        sort_na_list,
        keep_inds_list,
        op_pool_size_bytes,
        expected_log_messages,
        capfd,
        multi_rank=False,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for two cores case")
@pytest.mark.slow
@pytest.mark.timeout(1000)
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
    np.random.seed(543)  # Fix seed for deterministic output on all ranks.
    df = pd.DataFrame(
        {
            "A": pd.array([1] * 32000, dtype="Int64"),
            "B": pd.array(list(range(8000)) * 4, dtype="Int64"),
            "C": pd.Series(
                np.random.choice(
                    [
                        pd.NA,
                        "apple",
                        "pie",
                        "egg",
                        "salad",
                        "banana",
                        "tapas",
                        "bravas",
                        "pizza",
                        "omelette",
                        "salad",
                        "spinach",
                        "celery",
                    ],
                    32000,
                )
            ),
            "D": pd.Series(["abc"] * 32000),
            "E": pd.Series(
                np.random.choice(
                    [
                        pd.NA,
                        4.5,
                        23.43,
                        67.64,
                        234.0,
                        0.32,
                        -4.43,
                        -10.98,
                        6543.32,
                        -98.43,
                    ],
                    32000,
                ),
                dtype="Float64",
            ),
        }
    )

    key_inds_list = [1]
    sort_inds_list = [2, 4]
    sort_asc_list = [True, False]
    sort_na_list = [False, True]
    keep_inds_list = [0, 1, 3, 4]

    @bodo.jit(distributed=False)
    def py_mrnf(x: pd.DataFrame, asc: list[bool], na: list[bool]):
        return x.sort_values(by=["C", "E"], ascending=asc, na_position=na)

    expected_out = df.groupby(["B"], as_index=False, dropna=False).apply(
        lambda x: py_mrnf(
            x,
            sort_asc_list,
            [("last" if f else "first") for f in sort_na_list],
        ).iloc[0]
    )
    expected_out = expected_out[list(np.take(list(df.columns), keep_inds_list))]
    expected_output_size = expected_out.shape[0]

    # This will cause partition split during the "AppendBuildBatch[2]"
    op_pool_size_bytes = 1 * 1024 * 1024
    expected_partition_state = (
        [(2, 0), (2, 1), (1, 1)]
        if (bodo.get_rank() == 0)
        else [(3, 0), (3, 1), (2, 1), (1, 1)]
    )
    # Verify that we split a partition during AppendBuildBatch.
    expected_log_messages = (
        [
            "[DEBUG] GroupbyState::AppendBuildBatch[2]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0.",
            "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
            "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
            "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
            "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 3.",
            "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 4.",
        ]
        if bodo.get_rank() == 1
        else (
            [
                "[DEBUG] GroupbyState::AppendBuildBatch[2]: Encountered OperatorPoolThresholdExceededError.\n[DEBUG] Splitting partition 0.",
                "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 0.",
                "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 1.",
                "[DEBUG] GroupbyState::FinalizeBuild: Successfully finalized partition 2.",
                "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 3.",
                None,
            ]
        )
    )

    _test_helper(
        df,
        expected_out,
        expected_partition_state,
        expected_output_size,
        key_inds_list,
        sort_inds_list,
        sort_asc_list,
        sort_na_list,
        keep_inds_list,
        op_pool_size_bytes,
        expected_log_messages,
        capfd,
        multi_rank=True,
    )
