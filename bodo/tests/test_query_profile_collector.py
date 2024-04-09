import os
import tempfile

import numpy as np
import pandas as pd

import bodo
from bodo.libs.stream_join import (
    delete_join_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import _get_dist_arg, reduce_sum, temp_env_override


def test_query_profile_collection_compiles():
    """Check that all query profile collector functions compile"""

    @bodo.jit
    def impl():
        bodo.libs.query_profile_collector.init()
        bodo.libs.query_profile_collector.start_pipeline(1)
        bodo.libs.query_profile_collector.end_pipeline(1, 10)
        bodo.libs.query_profile_collector.finalize()
        return

    impl()


def test_query_profile_join_row_count_collection(tmp_path):
    """
    Check that Join submits its row counts to the QueryProfileCollector
    as expected.
    """

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

    @bodo.jit(distributed=["df1", "df2"])
    def impl(df1, df2):
        bodo.libs.query_profile_collector.init()

        join_state = init_join_state(
            0,  # op_id
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)
            _temp1 = _temp1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _temp1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        bodo.libs.query_profile_collector.start_pipeline(1)
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
        bodo.libs.query_profile_collector.end_pipeline(1, _temp2)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        bodo.libs.query_profile_collector.finalize()
        return df

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

    _ = impl(_get_dist_arg(build_df), _get_dist_arg(probe_df))
    build_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 0)
    )
    probe_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 1)
    )
    build_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 0)
    )
    probe_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    build_input_row_count = reduce_sum(build_input_row_count)
    probe_input_row_count = reduce_sum(probe_input_row_count)
    build_output_row_count = reduce_sum(build_output_row_count)
    probe_output_row_count = reduce_sum(probe_output_row_count)
    assert (
        build_input_row_count == 37500
    ), f"Expected build_input_row_count to be 37500 but it was {build_input_row_count} instead"
    assert (
        probe_input_row_count == 5000
    ), f"Expected probe_input_row_count to be 5000 but it was {probe_input_row_count} instead"
    assert (
        build_output_row_count == 0
    ), f"Expected build_output_row_count to be 0 but it was {build_output_row_count} instead"
    assert (
        probe_output_row_count == 18750000
    ), f"Expected probe_output_row_count to be 18750000 but it was {probe_output_row_count} instead"


def test_output_directory_can_be_set():
    """Check that the output directory can be set"""

    with tempfile.TemporaryDirectory() as test_dir:
        with temp_env_override(
            {"BODO_TRACING_OUTPUT_DIR": test_dir, "BODO_TRACING_LEVEL": "1"}
        ):

            @bodo.jit
            def impl():
                bodo.libs.query_profile_collector.init()
                bodo.libs.query_profile_collector.start_pipeline(1)
                bodo.libs.query_profile_collector.end_pipeline(1, 10)
                bodo.libs.query_profile_collector.finalize()
                return

            impl()
            for f in os.listdir(test_dir):
                assert f.startswith("query_profile")
                assert f.endswith(".json")
