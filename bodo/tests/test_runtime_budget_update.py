import sys

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import pytest_mark_one_rank, temp_env_override

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32", reason="TODO[BSE-4556]: enable buffer pool on Windows"
    ),
    pytest.mark.slow,
]


def hash_join_impl(df1, df2):
    """
    Helper impl for test_hash_join_dynamic_budget_increase.
    """
    from bodo.libs.memory_budget import (
        OperatorType,
    )
    from bodo.libs.streaming.join import (
        delete_join_state,
        init_join_state,
        join_build_consume_batch,
        join_probe_consume_batch,
    )
    from bodo.libs.streaming.join import (
        get_op_pool_budget_bytes as join_get_op_pool_budget_bytes,
    )

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

    def impl(df1, df2):
        # Assign a 2MiB budget
        bodo.libs.memory_budget.init_operator_comptroller_with_budget(2 * 1024 * 1024)
        bodo.libs.memory_budget.register_operator(0, OperatorType.JOIN, 0, 1, 100)

        # Dummy operator to force splitting the budget.
        bodo.libs.memory_budget.register_operator(1, OperatorType.JOIN, 0, 1, 100)
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        # Reduce budget of operator 1 to 0 to make more memory available to operator 0
        bodo.libs.memory_budget.reduce_operator_budget(1, 0)

        join_state = init_join_state(
            0,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            build_interval_cols,
            False,
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
            _temp1 = _temp1 + 1

        op_pool_budget_after_build = join_get_op_pool_budget_bytes(join_state)

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

        final_op_pool_budget = join_get_op_pool_budget_bytes(join_state)
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
            op_pool_budget_after_build,
            final_op_pool_budget,
        )

    return bodo.jit(distributed=["df1", "df2"])(impl)(df1, df2)


@pytest_mark_one_rank
def test_hash_join_dynamic_budget_increase(memory_leak_check, capfd):
    """
    Test that HashJoin is able to dynamically increase its budget
    at runtime (if there's budget available in its pipelines).
    """

    build_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 7500, dtype="Int64"),
            "B": np.array([1, 2, 3, 4, 5] * 7500, dtype=np.int32),
        }
    )

    probe_df = pd.DataFrame(
        {
            "C": pd.array([2, 6] * 2500, dtype="Int64"),
            "D": np.array([6, 2] * 2500, dtype=np.int8),
        }
    )

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_HASH_JOIN_PARTITIONING": "1",
            # Enable partitioning and budgets even though spilling is not setup
            "BODO_STREAM_HASH_JOIN_ENABLE_PARTITIONING": "1",
            "BODO_USE_MEMORY_BUDGETS": "1",
            "BODO_MEMORY_BUDGETS_DEBUG_LEVEL": "1",
        }
    ):
        try:
            (
                output,
                op_pool_budget_after_build,
                final_op_pool_budget,
            ) = hash_join_impl(build_df, probe_df)
        except Exception:
            out, err = capfd.readouterr()
            with capfd.disabled():
                print(f"STDOUT:\n{out}")
                print(f"STDERR:\n{err}")
            raise

    out, err = capfd.readouterr()

    ### Uncomment for debugging purposes ###
    # with capfd.disabled():
    #     print(f"output:\n{output}")
    #     print(f"op_pool_budget_after_build: {op_pool_budget_after_build}")
    #     print(f"final_op_pool_budget: {final_op_pool_budget}")
    #     print(f"stdout:\n{out}")
    #     print(f"stderr:\n{err}")
    ###

    # Verify that the expected log messages are present.
    expected_log_messages = [
        # Initial assignments
        "JOIN (Operator ID: 0, Original estimate: 100 bytes (relative), Allocated budget: 1.0MiB)",
        "JOIN (Operator ID: 1, Original estimate: 100 bytes (relative), Allocated budget: 1.0MiB)",
        # Dummy operator reducing its budget to 0
        "[DEBUG] OperatorComptroller::ReduceOperatorBudget: Reduced budget for operator 1 from 1.0MiB to 0 bytes.",
        # Operator 0 increasing its budget dynamically -- This is the main one we're looking for
        "[DEBUG] OperatorComptroller::RequestAdditionalBudget: Increased budget for operator 0 by 1.0MiB (from 1.0MiB to 2.0MiB).",
        # The rest are just sanity checks:
        "[DEBUG] HashJoinState::FinalizeBuild: Total number of partitions: 2. Estimated max partition size: ",
        "Total size of all partitions: ",
        "Estimated required size of Op-Pool: ",
        "[DEBUG] OperatorComptroller::ReduceOperatorBudget: Reduced budget for operator 0 from 2.0MiB to 1.13MiB",
        "[DEBUG] OperatorComptroller::ReduceOperatorBudget: Reduced budget for operator 0 from 1.13MiB to 0 bytes.",
    ]
    for expected_log_message in expected_log_messages:
        assert expected_log_message in err, (
            f"Expected log message ('{expected_log_message}') not in logs!"
        )

    # Verify that the output size is as expected
    expected_out_size = 18750000
    assert output.shape[0] == expected_out_size, (
        f"Final output size ({output.shape[0]}) is not as expected ({expected_out_size})"
    )

    # Verify that the op pool budget is as expected after the build and probe stages
    expected_op_pool_budget_after_build = (1190000, 1190500)
    assert (
        expected_op_pool_budget_after_build[0]
        <= op_pool_budget_after_build
        <= expected_op_pool_budget_after_build[1]
    ), (
        f"Operator pool budget after build ({op_pool_budget_after_build}) is not as expected ({expected_op_pool_budget_after_build})"
    )
    assert final_op_pool_budget == 0, (
        f"Final operator pool budget ({final_op_pool_budget}) is not 0!"
    )


def groupby_impl(df, key_inds_list, func_names, f_in_offsets, f_in_cols):
    """
    Helper impl for test_groupby_dynamic_budget_increase.
    """
    from bodo.libs.memory_budget import (
        OperatorType,
    )
    from bodo.libs.streaming.groupby import (
        delete_groupby_state,
        groupby_build_consume_batch,
        groupby_produce_output_batch,
        init_groupby_state,
    )

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

    def impl(df):
        # Assign a 3MiB total budget
        bodo.libs.memory_budget.init_operator_comptroller_with_budget(3 * 1024 * 1024)
        # This will get ~1MiB assigned initially
        bodo.libs.memory_budget.register_operator(0, OperatorType.GROUPBY, 0, 0, 100)
        # Dummy operator to force splitting the budget. This will get ~2MiB initially.
        bodo.libs.memory_budget.register_operator(1, OperatorType.GROUPBY, 0, 0, 200)
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        # Reduce budget of operator 1 to 0 to make more memory available to operator 0
        bodo.libs.memory_budget.reduce_operator_budget(1, 0)

        groupby_state = init_groupby_state(
            0, keys_inds, fnames, f_in_offsets, f_in_cols
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
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, out_col_meta
        )
        return df

    # We need a wrapper so that fnames, etc. are treated as globals.
    return bodo.jit(distributed=["df"])(impl)(df)


@pytest_mark_one_rank
def test_groupby_dynamic_budget_increase(memory_leak_check, capfd):
    """
    Test that Groupby is able to dynamically increase its budget
    at runtime (if there's budget available in its pipelines).
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

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
            # Enable partitioning and budgets even though spilling is not setup
            "BODO_STREAM_GROUPBY_ENABLE_PARTITIONING": "1",
            "BODO_USE_MEMORY_BUDGETS": "1",
            "BODO_MEMORY_BUDGETS_DEBUG_LEVEL": "1",
        }
    ):
        try:
            output = groupby_impl(df, [0], func_names, f_in_offsets, f_in_cols)
        except Exception:
            out, err = capfd.readouterr()
            with capfd.disabled():
                print(f"STDOUT:\n{out}")
                print(f"STDERR:\n{err}")
            raise

    out, err = capfd.readouterr()

    ### Uncomment for debugging purposes ###
    # with capfd.disabled():
    #     print(f"output:\n{output}")
    #     print(f"stdout:\n{out}")
    #     print(f"stderr:\n{err}")
    ###

    # Verify that the expected log messages are present.
    expected_log_messages = [
        # Initial assignments
        "GROUPBY (Operator ID: 0, Original estimate: 100 bytes (relative), Allocated budget: 1.0MiB)",
        "GROUPBY (Operator ID: 1, Original estimate: 200 bytes (relative), Allocated budget: 1.98MiB)",
        # Dummy operator reducing its budget to 0
        "[DEBUG] OperatorComptroller::ReduceOperatorBudget: Reduced budget for operator 1 from 1.98MiB to 0 bytes.",
        # Operator 0 increasing its budget dynamically -- This is the main one we're looking for
        "[DEBUG] OperatorComptroller::RequestAdditionalBudget: Increased budget for operator 0 by 1.98MiB (from 1.0MiB to 3.0MiB).",
        # Sanity check:
        "[DEBUG] GroupbyState::FinalizeBuild: Total number of partitions: 1.",
    ]
    for expected_log_message in expected_log_messages:
        assert expected_log_message in err, (
            f"Expected log message ('{expected_log_message}') not in logs!"
        )

    # Verify that the output size is as expected
    expected_output_size = 1000
    assert output.shape[0] == expected_output_size, (
        f"Final output size ({output.shape[0]}) is not as expected ({expected_output_size})"
    )
