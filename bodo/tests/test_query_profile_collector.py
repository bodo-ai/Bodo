import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (
    _get_dist_arg,
    get_query_profile_location,
    get_snowflake_connection_string,
    pytest_mark_snowflake,
    temp_env_override,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="TODO[BSE-4580]: fix nightly test hangs on Windows"
)


def test_query_profile_collection_compiles(memory_leak_check):
    """Check that all query profile collector functions compile"""

    @bodo.jit
    def impl():
        bodo.libs.query_profile_collector.init()
        bodo.libs.query_profile_collector.start_pipeline(1)
        bodo.libs.query_profile_collector.submit_operator_stage_row_counts(1, 0, 0)
        bodo.libs.query_profile_collector.submit_operator_stage_time(1, 0, 100.0)
        bodo.libs.query_profile_collector.get_operator_duration(1)
        bodo.libs.query_profile_collector.end_pipeline(1, 10)
        bodo.libs.query_profile_collector.finalize()
        return

    impl()


def test_output_directory_can_be_set(tmp_path):
    """Check that the output directory can be set"""

    comm = MPI.COMM_WORLD
    tmp_path_rank0 = comm.bcast(str(tmp_path))
    num_ranks = comm.Get_size()
    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        runs = os.listdir(tmp_path_rank0)
        assert len(runs) == 0
        bodo.barrier()

        @bodo.jit
        def impl():
            bodo.libs.query_profile_collector.init()
            bodo.libs.query_profile_collector.start_pipeline(1)
            bodo.libs.query_profile_collector.end_pipeline(1, 10)
            bodo.libs.query_profile_collector.finalize()
            return

        impl()
        bodo.barrier()
        runs = os.listdir(tmp_path_rank0)
        assert len(runs) == 1
        out_path = f"{tmp_path_rank0}/{runs[0]}"
        profiles = os.listdir(out_path)
        assert len(profiles) == num_ranks
        for f in profiles:
            assert f.startswith("query_profile")
            assert f.endswith(".json")
        bodo.barrier()


def test_join_row_count_collection(memory_leak_check):
    """
    Check that Join submits its row counts to the QueryProfileCollector
    as expected.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.join import (
        delete_join_state,
        init_join_state,
        join_build_consume_batch,
        join_probe_consume_batch,
    )
    from bodo.tests.utils_jit import reduce_sum

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
            build_interval_cols,
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
            is_last1, _ = join_build_consume_batch(join_state, T2, is_last1)
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

    build_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    probe_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 2)
    )
    build_output_row_count = reduce_sum(build_output_row_count)
    probe_output_row_count = reduce_sum(probe_output_row_count)
    assert build_output_row_count == 0, (
        f"Expected build_output_row_count to be 0 but it was {build_output_row_count} instead"
    )
    assert probe_output_row_count == 18750000, (
        f"Expected probe_output_row_count to be 18750000 but it was {probe_output_row_count} instead"
    )


def test_groupby_row_count_collection(memory_leak_check):
    """
    Check that Groupby submits its row counts to the QueryProfileCollector
    as expected.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.groupby import (
        delete_groupby_state,
        groupby_build_consume_batch,
        groupby_produce_output_batch,
        init_groupby_state,
    )
    from bodo.tests.utils_jit import reduce_sum

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

    keys_inds = bodo.utils.typing.MetaType((0,))
    out_col_meta_l = ["key"] + [f"out_{i}" for i in range(len(func_names))]
    out_col_meta = bodo.utils.typing.ColNamesMetaType(tuple(out_col_meta_l))
    len_kept_cols = len(df.columns)
    kept_cols = bodo.utils.typing.MetaType(tuple(range(len_kept_cols)))
    batch_size = 4000
    fnames = bodo.utils.typing.MetaType(tuple(func_names))
    f_in_offsets = bodo.utils.typing.MetaType(tuple(f_in_offsets))
    f_in_cols = bodo.utils.typing.MetaType(tuple(f_in_cols))

    @bodo.jit(distributed=["df"])
    def impl(df):
        bodo.libs.query_profile_collector.init()

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
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, kept_cols, False)
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)
        is_last2 = False
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        _iter_2 = 0
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            _iter_2 = _iter_2 + 1
        bodo.libs.query_profile_collector.end_pipeline(1, _iter_2)
        delete_groupby_state(groupby_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, out_col_meta
        )
        bodo.libs.query_profile_collector.finalize()
        return df

    _ = impl(_get_dist_arg(df))

    produce_output_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 2)
    )
    produce_output_output_row_count = reduce_sum(produce_output_output_row_count)
    assert produce_output_output_row_count == 1000, (
        f"Expected produce_output_output_row_count to be 1000 but it was {produce_output_output_row_count} instead"
    )


@pytest_mark_snowflake
def test_snowflake_read_row_count_collection(memory_leak_check):
    """
    Check that Snowflake Reader submits its row counts to the QueryProfileCollector
    as expected.
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
    from bodo.tests.utils_jit import reduce_sum

    @bodo.jit()
    def impl(conn):
        bodo.libs.query_profile_collector.init()
        total_max = 0

        reader = pd.read_sql(
            "SELECT * FROM LINEITEM",
            conn,
            _bodo_chunksize=4000,
            _bodo_read_as_table=True,
            _bodo_sql_op_id=0,
        )
        is_last_global = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            # Perform more compute in between to see caching speedup
            local_max = pd.Series(bodo.hiframes.table.get_table_data(table, 1)).max()
            total_max = max(total_max, local_max)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        arrow_reader_del(reader)
        bodo.libs.query_profile_collector.finalize()
        return total_max

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    _ = impl(conn)
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert reader_output_row_count == 6001215, (
        f"Expected reader_output_row_count to be 6001215, but it was {reader_output_row_count} instead."
    )


@pytest.mark.iceberg
def test_iceberg_read_row_count_collection(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Check that Iceberg Reader submits its row counts to the QueryProfileCollector
    as expected.
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
    from bodo.tests.utils_jit import reduce_sum

    col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D"))

    @bodo.jit()
    def impl(table_name, conn, db_schema):
        bodo.libs.query_profile_collector.init()
        total_max = pd.Timestamp(year=1970, month=1, day=1, tz="UTC")
        is_last_global = False
        reader = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_chunksize=4096,
            _bodo_sql_op_id=0,
        )
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (table,), index_var, col_meta
            )
            local_max = df["A"].max()
            total_max = max(local_max, total_max)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)
        arrow_reader_del(reader)
        bodo.libs.query_profile_collector.finalize()
        return total_max

    table_name = "SIMPLE_PRIMITIVES_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    _ = impl(table_name, conn, db_schema)
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert reader_output_row_count == 200, (
        f"Expected reader_output_row_count to be 200, but it was {reader_output_row_count} instead."
    )


@pytest.mark.parquet
def test_parquet_read_row_count_collection(datapath, memory_leak_check):
    """
    Check that Parquet Reader submits its row counts to the QueryProfileCollector
    as expected.
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
    from bodo.tests.utils_jit import reduce_sum

    @bodo.jit()
    def impl(path):
        bodo.libs.query_profile_collector.init()
        total_max = 0
        is_last_global = False
        reader = pd.read_parquet(
            path,
            _bodo_use_index=False,
            _bodo_chunksize=4096,
            _bodo_sql_op_id=0,
            dtype_backend="pyarrow",
        )
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last_global:
            T1, is_last = read_arrow_next(reader, True)
            local_max = pd.Series(bodo.hiframes.table.get_table_data(T1, 1)).max()
            total_max = max(total_max, local_max)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)
        arrow_reader_del(reader)
        bodo.libs.query_profile_collector.finalize()
        return total_max

    _ = impl(datapath("tpch-test_data/parquet/lineitem.pq"))
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert reader_output_row_count == 120515, (
        f"Expected reader_output_row_count to be 120515, but it was {reader_output_row_count} instead."
    )


def test_hash_join_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by hash join.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.join import (
        delete_join_state,
        init_join_state,
        join_build_consume_batch,
        join_probe_consume_batch,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_pes = comm.Get_size()
    tmp_path_rank0 = comm.bcast(str(tmp_path))
    build_keys_inds = bodo.utils.typing.MetaType((0, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D"))
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(("E", "F", "G", "H"))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        ("E", "F", "G", "H", "A", "B", "C", "D")
    )
    build_interval_cols = bodo.utils.typing.MetaType(())

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
            build_interval_cols,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 4
        )
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            is_last1, _ = join_build_consume_batch(join_state, T2, is_last1)
            _temp1 = _temp1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _temp1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 4
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
            # Non-dict key column
            "A": pd.array([1, 2, 3, 4, 5] * 7500, dtype="Int64"),
            # Dict key column
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    ["A", "bc", "De", "FGh", "Ij"] * 7500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
            # Non-key non-dict column
            "C": pd.array([1, 2, 3, 4, 5] * 7500, dtype="Int64"),
            # Non-key dict column
            "D": pd.arrays.ArrowStringArray(
                pa.array(
                    ["pizza", "potatoes", "cilantro", "cucumber", "jalapeno"] * 7500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    probe_df = pd.DataFrame(
        {
            # Non-dict key column
            "E": pd.array([2, 6] * 2500, dtype="Int64"),
            # Dict key column
            "F": pd.arrays.ArrowStringArray(
                pa.array(
                    ["bc", "Ij"] * 2500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
            # Non-key non-dict column
            "G": pd.array([2, 6] * 2500, dtype="Int64"),
            # Non-key dict column
            "H": pd.arrays.ArrowStringArray(
                pa.array(
                    ["cilantro", "cucumber"] * 2500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(build_df), _get_dist_arg(probe_df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]

    # Verify build metrics
    stage_1_metrics = operator_report["stage_1"]["metrics"]
    build_metrics: list = stage_1_metrics
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}
    if rank == 0:
        assert "bcast_join" in build_metrics_dict
        assert "bloom_filter_enabled" in build_metrics_dict
        assert "n_key_dict_builders" in build_metrics_dict
        assert "n_non_key_dict_builders" in build_metrics_dict
        assert "n_shuffle_send" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_send"] >= (0 if n_pes == 1 else 1)
        assert "n_shuffle_recv" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_recv"] >= (0 if n_pes == 1 else 1)
    assert "shuffle_buffer_append_time" in build_metrics_dict
    assert "ht_hashing_time" in build_metrics_dict
    assert "repartitioning_time_total" in build_metrics_dict

    # Verify probe metrics
    stage_2_metrics = operator_report["stage_2"]["metrics"]
    probe_metrics: list = stage_2_metrics
    probe_metrics_dict = {x["name"]: x["stat"] for x in probe_metrics}
    if rank == 0:
        assert "n_key_dict_builders" in probe_metrics_dict
        assert "n_non_key_dict_builders" in probe_metrics_dict
        assert "n_shuffle_send" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_send"] >= (0 if n_pes == 1 else 1)
        assert "n_shuffle_recv" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_recv"] >= (0 if n_pes == 1 else 1)
    assert "output_append_time" in probe_metrics_dict
    assert "output_total_nrows" in probe_metrics_dict
    assert "output_total_nrows_rem_at_finalize" in probe_metrics_dict
    assert "output_peak_nrows" in probe_metrics_dict
    assert "shuffle_buffer_append_time" in probe_metrics_dict
    assert "ht_probe_time" in probe_metrics_dict
    assert "finalize_inactive_partitions_total_time" in probe_metrics_dict


def test_nested_loop_join_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by nested loop join.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.join import (
        delete_join_state,
        init_join_state,
        join_build_consume_batch,
        join_probe_consume_batch,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B"))
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(("C", "D"))
    col_meta = bodo.utils.typing.ColNamesMetaType(("C", "D", "A", "B"))
    non_equi_condition = "right.A >= left.C"
    build_interval_cols = bodo.utils.typing.MetaType(())

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
            build_interval_cols,
            False,
            non_equi_condition=non_equi_condition,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 4
        )
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            is_last1, _ = join_build_consume_batch(join_state, T2, is_last1)
            _temp1 = _temp1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _temp1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 4
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
            # Non-dict column
            "A": pd.array([1, 2, 3, 4, 5] * 2000, dtype="Int64"),
            # Dict column
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    ["A", "bc", "De", "FGh", "Ij"] * 2000,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    probe_df = pd.DataFrame(
        {
            # Non-dict column
            "C": pd.array([2, 6] * 2500, dtype="Int64"),
            # Dict column
            "D": pd.arrays.ArrowStringArray(
                pa.array(
                    ["bc", "Ij"] * 2500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(build_df), _get_dist_arg(probe_df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]

    # Verify build metrics
    stage_1_metrics = operator_report["stage_1"]["metrics"]
    build_metrics: list = stage_1_metrics
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}
    if rank == 0:
        assert "bcast_join" in build_metrics_dict
        assert "block_size_bytes" in build_metrics_dict
        assert "chunk_size_nrows" in build_metrics_dict
        assert "n_dict_builders" in build_metrics_dict
    assert "append_time" in build_metrics_dict
    assert "num_chunks" in build_metrics_dict

    # Verify probe metrics
    stage_2_metrics = operator_report["stage_2"]["metrics"]
    probe_metrics: list = stage_2_metrics
    probe_metrics_dict = {x["name"]: x["stat"] for x in probe_metrics}
    if rank == 0:
        assert "n_dict_builders" in probe_metrics_dict
    assert "output_append_time" in probe_metrics_dict
    assert "output_total_nrows" in probe_metrics_dict
    assert "output_total_nrows_rem_at_finalize" in probe_metrics_dict
    assert "output_peak_nrows" in probe_metrics_dict
    assert "global_dict_unification_time" in probe_metrics_dict
    assert "bcast_size_bytes" in probe_metrics_dict
    assert "bcast_table_time" in probe_metrics_dict
    assert "compute_matches_time" in probe_metrics_dict


def test_groupby_agg_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by groupby in the incremental aggregation case.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.groupby import (
        delete_groupby_state,
        groupby_build_consume_batch,
        groupby_produce_output_batch,
        init_groupby_state,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_pes = comm.Get_size()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    keys_inds = bodo.utils.typing.MetaType((0,))
    in_kept_cols = bodo.utils.typing.ColNamesMetaType((0, 1))
    fnames = bodo.utils.typing.MetaType(("max",))
    f_in_cols = bodo.utils.typing.MetaType((1,))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    batch_size = 1000
    out_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B_max"))

    @bodo.jit(distributed=["df"])
    def impl(df):
        bodo.libs.query_profile_collector.init()
        groupby_state = init_groupby_state(
            0, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            in_kept_cols,
            2,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        is_last2 = False
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        _iter_2 = 0
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            _iter_2 = _iter_2 + 1
        bodo.libs.query_profile_collector.end_pipeline(1, _iter_2)
        delete_groupby_state(groupby_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, out_col_meta
        )
        bodo.libs.query_profile_collector.finalize()
        return out_df

    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(1000)) * 32, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]
    stage_0 = operator_report["stage_0"]
    stage_1 = operator_report["stage_1"]
    assert "stage_2" in operator_report

    build_metrics = stage_1["metrics"]
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}

    if rank == 0:
        initialization_metrics = stage_0["metrics"]
        initialization_metrics_names: list[str] = [
            x["name"] for x in initialization_metrics
        ]
        assert "acc_or_agg" in initialization_metrics_names
        assert (
            initialization_metrics[initialization_metrics_names.index("acc_or_agg")][
                "stat"
            ]
            == "AGG"
        )
        assert "n_shuffle_send" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_send"] >= (0 if n_pes == 1 else 1)
        assert "n_shuffle_recv" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_recv"] >= (0 if n_pes == 1 else 1)

    assert "pre_agg_total_time" in build_metrics_dict
    assert "n_repartitions_in_append" in build_metrics_dict
    assert "input_groupby_hashing_time" in build_metrics_dict
    assert "update_logical_ht_time" in build_metrics_dict
    assert "combine_input_time" in build_metrics_dict
    assert "shuffle_update_logical_ht_time" in build_metrics_dict
    assert "finalize_time_total" in build_metrics_dict
    assert "shuffle_send_time" in build_metrics_dict
    assert "shuffle_recv_time" in build_metrics_dict
    assert "shuffle_n_local_reductions" in build_metrics_dict
    assert "key_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "non_key_build_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "non_key_output_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "output_append_time" in build_metrics_dict
    assert "final_partitioning_state" in build_metrics_dict


def test_groupby_acc_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by groupby in the accumulate input case.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.groupby import (
        delete_groupby_state,
        groupby_build_consume_batch,
        groupby_produce_output_batch,
        init_groupby_state,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_pes = comm.Get_size()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    keys_inds = bodo.utils.typing.MetaType((0,))
    in_kept_cols = bodo.utils.typing.MetaType((0, 1))
    fnames = bodo.utils.typing.MetaType(("median",))
    f_in_cols = bodo.utils.typing.MetaType((1,))
    f_in_offsets = bodo.utils.typing.MetaType((0, 1))
    batch_size = 1000
    out_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B_median"))

    @bodo.jit(distributed=["df"])
    def impl(df):
        bodo.libs.query_profile_collector.init()
        groupby_state = init_groupby_state(
            0, keys_inds, fnames, f_in_offsets, f_in_cols
        )
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            in_kept_cols,
            2,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        is_last2 = False
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        _iter_2 = 0
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(groupby_state, True)
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            _iter_2 = _iter_2 + 1
        bodo.libs.query_profile_collector.end_pipeline(1, _iter_2)
        delete_groupby_state(groupby_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, out_col_meta
        )
        bodo.libs.query_profile_collector.finalize()
        return out_df

    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(1000)) * 32, dtype="Int64"),
            "B": np.array(
                [1, 3, 5, 11, 1, 3, 5, 3, 4, 78, 23, 120, 87, 34, 52, 34] * 2000,
                dtype=np.float32,
            ),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]
    stage_0 = operator_report["stage_0"]
    stage_1 = operator_report["stage_1"]
    assert "stage_2" in operator_report
    build_metrics = stage_1["metrics"]
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}
    if rank == 0:
        initialization_metrics = stage_0["metrics"]
        initialization_metrics_names: list[str] = [
            x["name"] for x in initialization_metrics
        ]
        assert "acc_or_agg" in initialization_metrics_names
        assert (
            initialization_metrics[initialization_metrics_names.index("acc_or_agg")][
                "stat"
            ]
            == "ACC"
        )
        assert "n_shuffle_send" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_send"] >= (0 if n_pes == 1 else 1)
        assert "n_shuffle_recv" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_recv"] >= (0 if n_pes == 1 else 1)

    assert "pre_agg_total_time" not in build_metrics_dict
    assert "n_repartitions_in_append" in build_metrics_dict
    assert "input_groupby_hashing_time" not in build_metrics_dict
    assert "input_part_hashing_time" in build_metrics_dict
    assert "finalize_time_total" in build_metrics_dict
    assert "shuffle_send_time" in build_metrics_dict
    assert "shuffle_recv_time" in build_metrics_dict
    assert "shuffle_n_local_reductions" in build_metrics_dict
    assert "key_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "non_key_build_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "non_key_output_dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "output_append_time" in build_metrics_dict
    assert "final_partitioning_state" in build_metrics_dict


def test_mrnf_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by MRNF.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.streaming.groupby import (
        delete_groupby_state,
        groupby_build_consume_batch,
        groupby_produce_output_batch,
        init_groupby_state,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_pes = comm.Get_size()
    tmp_path_rank0 = comm.bcast(str(tmp_path))
    keys_inds = bodo.utils.typing.MetaType((0,))
    fnames = bodo.utils.typing.MetaType(("min_row_number_filter",))
    n_cols = 3
    f_in_cols = bodo.utils.typing.MetaType((1, 2))
    f_in_offsets = bodo.utils.typing.MetaType((0, n_cols - 1))
    mrnf_sort_col_inds = bodo.utils.typing.MetaType((1,))
    mrnf_sort_col_asc = bodo.utils.typing.MetaType((False,))
    mrnf_sort_col_na = bodo.utils.typing.MetaType((True,))
    mrnf_col_inds_keep = bodo.utils.typing.MetaType((0, 1, 2))
    input_table_kept_cols = bodo.utils.typing.MetaType(tuple(range(n_cols)))
    output_table_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C"))
    batch_size = 1000

    @bodo.jit(distributed=["df"])
    def impl(df):
        bodo.libs.query_profile_collector.init()
        mrnf_state = init_groupby_state(
            0,
            keys_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        )

        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            input_table_kept_cols,
            n_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, input_table_kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(mrnf_state, T3, is_last1, True)
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)
        out_dfs = []
        is_last2 = False
        _iter_2 = 0
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(mrnf_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, output_table_col_meta
            )
            out_dfs.append(df_final)
            _iter_2 = _iter_2 + 1
        bodo.libs.query_profile_collector.end_pipeline(1, _iter_2)
        delete_groupby_state(mrnf_state)
        out_df = pd.concat(out_dfs)
        bodo.libs.query_profile_collector.finalize()
        return out_df

    df = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(200)) * 160, dtype="Int64"),
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    [None, "apple", "pie", "egg", "salad", "banana", "kiwi", "pudding"]
                    * 4000,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
            "C": pd.array(list(np.arange(100)) * 320, dtype="Int64"),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]
    stage_0 = operator_report["stage_0"]
    stage_1 = operator_report["stage_1"]
    assert "stage_2" in operator_report
    build_metrics = stage_1["metrics"]
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}

    if rank == 0:
        initialization_metrics = stage_0["metrics"]
        initialization_metrics_names: list[str] = [
            x["name"] for x in initialization_metrics
        ]
        assert "acc_or_agg" in initialization_metrics_names
        assert (
            initialization_metrics[initialization_metrics_names.index("acc_or_agg")][
                "stat"
            ]
            == "ACC"
        )
        assert "aggregation_type" in initialization_metrics_names
        assert (
            initialization_metrics[
                initialization_metrics_names.index("aggregation_type")
            ]["stat"]
            == "MIN ROW NUMBER FILTER"
        )
        assert "n_shuffle_send" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_send"] >= (0 if n_pes == 1 else 1)
        assert "n_shuffle_recv" in build_metrics_dict
        # Async shuffle code avoids any shuffle when not necessary
        assert build_metrics_dict["n_shuffle_recv"] >= (0 if n_pes == 1 else 1)

    assert "pre_agg_total_time" not in build_metrics_dict
    assert "n_repartitions_in_append" in build_metrics_dict
    assert "appends_active_time" in build_metrics_dict
    assert "input_part_hashing_time" in build_metrics_dict
    assert "finalize_time_total" in build_metrics_dict
    assert "finalize_compute_mrnf_time" in build_metrics_dict
    assert "finalize_colset_update_time" in build_metrics_dict
    assert "shuffle_send_time" in build_metrics_dict
    assert "shuffle_recv_time" in build_metrics_dict
    assert "shuffle_local_reduction_mrnf_colset_update_time" in build_metrics_dict
    assert "dict_builders_unify_cache_id_misses" in build_metrics_dict
    assert "output_append_time" in build_metrics_dict
    assert "final_partitioning_state" in build_metrics_dict


def test_union_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by Union.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    in_kept_cols = bodo.utils.typing.MetaType((0, 1))
    batch_size = 1000
    out_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B "))

    @bodo.jit(distributed=["df1", "df2", "df3"])
    def impl(df1, df2, df3):
        bodo.libs.query_profile_collector.init()
        union_state = bodo.libs.streaming.union.init_union_state(0, all=False)

        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1),
            (),
            in_kept_cols,
            2,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        is_last1 = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                union_state, T3, is_last1, False
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2),
            (),
            in_kept_cols,
            2,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        is_last1 = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                union_state, T3, is_last1, False
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(1, _iter_1)

        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df3),
            (),
            in_kept_cols,
            2,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        is_last1 = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(2)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, in_kept_cols, False)
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                union_state, T3, is_last1, True
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(2, _iter_1)

        is_last3 = False
        _iter_3 = 0
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        bodo.libs.query_profile_collector.start_pipeline(3)
        while not is_last3:
            T5, is_last3 = bodo.libs.streaming.union.union_produce_batch(
                union_state, True
            )
            bodo.libs.table_builder.table_builder_append(table_builder, T5)
            _iter_3 = _iter_3 + 1
        bodo.libs.query_profile_collector.end_pipeline(3, _iter_3)

        bodo.libs.streaming.union.delete_union_state(union_state)
        T6 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (T6,), index_1, out_col_meta
        )
        bodo.libs.query_profile_collector.finalize()
        return out_df

    df1 = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(90)) * 100, dtype="Int64"),
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    [
                        None,
                        "apple",
                        "pie",
                        "egg",
                        "salad",
                        "banana",
                        "kiwi",
                        "pudding",
                        "caramel",
                    ]
                    * 1000,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(900)) * 20, dtype="Int64"),
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    [
                        "apple",
                        "kiwi",
                        "pudding",
                        None,
                        "caramel",
                    ]
                    * 3600,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )
    df3 = pd.DataFrame(
        {
            "A": pd.array(list(np.arange(10000, 10010)) * 10, dtype="Int64"),
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    [
                        "apple",
                        "pie",
                    ]
                    * 50,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(_get_dist_arg(df1), _get_dist_arg(df2), _get_dist_arg(df3))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    operator_report = profile_json["operator_reports"]["0"]
    stage_0 = operator_report["stage_0"]
    stage_1 = operator_report["stage_1"]
    stage_2 = operator_report["stage_2"]
    stage_3 = operator_report["stage_3"]
    stage_4 = operator_report["stage_4"]
    if rank == 0:
        initialization_metrics = stage_0["metrics"]
        initialization_metrics_names: list[str] = [
            x["name"] for x in initialization_metrics
        ]
        assert "acc_or_agg" in initialization_metrics_names
        assert (
            initialization_metrics[initialization_metrics_names.index("acc_or_agg")][
                "stat"
            ]
            == "AGG"
        )

    build_stage1_metrics = stage_1["metrics"]
    build_stage1_metrics_names: set[str] = {x["name"] for x in build_stage1_metrics}
    build_stage2_metrics = stage_2["metrics"]
    build_stage2_metrics_names: set[str] = {x["name"] for x in build_stage2_metrics}
    build_stage3_metrics = stage_3["metrics"]
    build_stage3_metrics_names: set[str] = {x["name"] for x in build_stage3_metrics}
    output_stage_metrics = stage_4["metrics"]
    output_metrics_dict = {x["name"]: x["stat"] for x in output_stage_metrics}

    # Some metrics should be reported in all stages
    for k in [
        "pre_agg_total_time",
        "repartitioning_time_total",
        "update_logical_ht_time",
        "key_dict_builders_unify_cache_id_misses",
    ]:
        assert k in build_stage1_metrics_names, k
        assert k in build_stage2_metrics_names, k
        assert k in build_stage3_metrics_names, k

    # Some metrics should only be reported in the final stage
    for k in [
        "finalize_time_total",
        "non_key_output_dict_builders_unify_cache_id_misses",
        "output_append_time",
        "final_partitioning_state",
    ]:
        assert k not in build_stage1_metrics_names, k
        assert k not in build_stage2_metrics_names, k
        assert k in build_stage3_metrics_names, k

    assert "work_stealing_enabled" in output_metrics_dict
    assert output_metrics_dict["work_stealing_enabled"] == 0


@pytest_mark_snowflake
def test_snowflake_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by Snowflake Reader.
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    @bodo.jit()
    def impl(conn):
        bodo.libs.query_profile_collector.init()
        total_max = 0

        reader = pd.read_sql(
            "SELECT * FROM LINEITEM",
            conn,
            _bodo_chunksize=4000,
            _bodo_read_as_table=True,
            _bodo_sql_op_id=0,
        )
        is_last_global = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            local_max = pd.Series(bodo.hiframes.table.get_table_data(table, 1)).max()
            total_max = max(total_max, local_max)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        arrow_reader_del(reader)
        bodo.libs.query_profile_collector.finalize()
        return total_max

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(conn)

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    assert "operator_reports" in profile_json
    assert "0" in profile_json["operator_reports"]
    operator_report = profile_json["operator_reports"]["0"]
    assert "stage_0" in operator_report
    assert "stage_1" in operator_report
    init_metrics = operator_report["stage_0"]["metrics"]
    init_metrics_dict = {x["name"]: x["stat"] for x in init_metrics}
    read_metrics = operator_report["stage_1"]["metrics"]
    read_metrics_dict = {x["name"]: x["stat"] for x in read_metrics}

    if rank == 0:
        assert "sf_data_prep_time" in init_metrics_dict
        assert "limit_nrows" in init_metrics_dict
        assert "get_ds_time" in init_metrics_dict
        assert "global_nrows_to_read" in init_metrics_dict
        assert "global_n_pieces" in init_metrics_dict
        assert "create_dict_encoding_from_strings" in init_metrics_dict
        assert "n_str_as_dict_cols" in init_metrics_dict
        assert "n_dict_builders" in read_metrics_dict

    assert "local_rows_to_read" in init_metrics_dict
    assert "local_n_pieces_to_read_from" in init_metrics_dict

    assert "to_arrow_time" in read_metrics_dict
    assert "cast_arrow_table_time" in read_metrics_dict
    assert "total_append_time" in read_metrics_dict
    assert "arrow_rb_to_bodo_time" in read_metrics_dict
    assert "ctb_pop_chunk_time" in read_metrics_dict
    assert "output_append_time" in read_metrics_dict
    assert "output_total_nrows" in read_metrics_dict
    assert "output_peak_nrows" in read_metrics_dict
    assert "dict_builders_unify_cache_id_misses" in read_metrics_dict
    assert "read_batch_total_time" in read_metrics_dict


@pytest.mark.iceberg
def test_iceberg_metrics_collection(
    memory_leak_check, tmp_path, iceberg_database, iceberg_table_conn
):
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @bodo.jit()
    def impl(conn):
        bodo.libs.query_profile_collector.init()
        reader = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_chunksize=4000,
            _bodo_read_as_table=True,
            _bodo_sql_op_id=0,
        )
        is_last_global = False
        _iter_1 = 0
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            _iter_1 = _iter_1 + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter_1)

        arrow_reader_del(reader)
        bodo.libs.query_profile_collector.finalize()

    bodo.set_verbose_level(2)
    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = impl(conn)

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)

    assert "operator_reports" in profile_json
    assert "0" in profile_json["operator_reports"]
    operator_report = profile_json["operator_reports"]["0"]
    assert "stage_0" in operator_report
    assert "stage_1" in operator_report
    init_metrics = operator_report["stage_0"]["metrics"]
    init_metrics_dict = {x["name"]: x["stat"] for x in init_metrics}
    read_metrics = operator_report["stage_1"]["metrics"]
    read_metrics_dict = {x["name"]: x["stat"] for x in read_metrics}

    assert "limit_nrows" in init_metrics_dict
    assert "create_dict_encoding_from_strings" in init_metrics_dict
    assert "n_str_as_dict_cols" in init_metrics_dict
    assert "force_row_level" in init_metrics_dict
    assert "row_level" in init_metrics_dict
    assert "get_ds_exact_row_counts_time" in init_metrics_dict
    assert "get_ds_file_frags_creation_time" in init_metrics_dict
    assert "get_ds_get_row_counts_nrgs" in init_metrics_dict
    assert "get_ds_get_row_counts_total_bytes" in init_metrics_dict

    for k in [
        "get_ds_time",
        "global_nrows_to_read",
        "global_n_pieces",
        "local_rows_to_read",
        "local_n_pieces_to_read_from",
        "get_ds_file_list_time",
        "get_ds_file_to_schema_time_us",
        "get_ds_get_fs_time",
        "get_ds_n_files_analyzed",
        "get_ds_get_sg_id_time",
        "get_ds_sort_by_sg_id_time",
        "get_ds_nunique_sgs_seen",
        "get_ds_get_row_counts_nrows",
        "get_ds_pieces_allgather_time",
        "get_ds_sort_all_pieces_time",
        "get_ds_assemble_ds_time",
        "ds_nunique_schema_groups",
        "init_scanners_get_pa_datasets_time",
        "n_scanners",
        "create_scanners_time",
        "init_arrow_reader_total_time",
        "distribute_pieces_or_rows_time",
    ]:
        assert k in init_metrics_dict, f"Metric {k} is missing"
        assert init_metrics_dict[k] > 0, f"Metric {k} is {init_metrics_dict[k]}, not >0"

    for k in [
        "get_batch_time",
        "evolve_time",
        "arrow_rb_to_bodo_time",
        "n_batches",
        "read_batch_total_time",
    ]:
        assert k in read_metrics_dict, k
        assert read_metrics_dict[k] > 0, k

    assert "unify_time" in read_metrics_dict
    assert "unify_append_small_time" in read_metrics_dict
    assert "output_pop_chunk_time" in read_metrics_dict
    assert "n_small_batches" in read_metrics_dict


@pytest.mark.parametrize(
    "limit_offset",
    [
        pytest.param((-1, -1), id="no_limit_offset"),
        pytest.param((10, 10), id="small_limit_offset"),
        pytest.param((10_000_000_000, 5_000), id="large_limit"),
    ],
)
def test_sort_metrics_collection(memory_leak_check, tmp_path, limit_offset):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by sort.
    """
    from bodo.utils.typing import ColNamesMetaType, MetaType

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_pes = comm.Get_size()
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    input_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5] * 7500, dtype="Int64"),
            "B": pd.arrays.ArrowStringArray(
                pa.array(
                    ["A", "bc", "De", "FGh", "Ij"] * 7500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
            "C": pd.array([1, 2, 3, 4, 5] * 7500, dtype="Int64"),
            "D": pd.arrays.ArrowStringArray(
                pa.array(
                    ["pizza", "potatoes", "cilantro", "cucumber", "jalapeno"] * 7500,
                    type=pa.dictionary(pa.int32(), pa.string()),
                )
            ),
        }
    )

    global_1 = MetaType((0, 1, 2, 3))
    global_2 = ColNamesMetaType(("A", "B", "C", "D"))
    limit, offset = limit_offset
    limit_offset_case = bool(limit != -1)
    small_limit_case = limit_offset_case and bool(limit < 100)

    def impl(df):
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            0, bodo.libs.memory_budget.OperatorType.SORT, 0, 0, 268435456
        )
        bodo.libs.memory_budget.register_operator(
            1, bodo.libs.memory_budget.OperatorType.ACCUMULATE_TABLE, 1, 1, 1800000
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        bodo.libs.query_profile_collector.init()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), global_1, 4
        )

        is_last = False
        _iter = 0
        df_len = bodo.hiframes.table.local_len(T1)
        sort_state = bodo.libs.streaming.sort.init_stream_sort_state(
            0,  # Op ID
            limit,
            offset,
            ["A", "D"],
            [True, True],
            ["last", "last"],
            ("A", "B", "C", "D"),
        )
        bodo.libs.query_profile_collector.start_pipeline(0)
        while not (is_last):
            T3 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter * 4096), ((_iter + 1) * 4096))
            )
            is_last = (_iter * 4096) >= df_len
            is_last = bodo.libs.streaming.sort.sort_build_consume_batch(
                sort_state, T3, is_last
            )
            _iter = _iter + 1
        bodo.libs.query_profile_collector.end_pipeline(0, _iter)

        is_last = False
        _iter = 0
        out_tb = bodo.libs.table_builder.init_table_builder_state(1)
        bodo.libs.query_profile_collector.start_pipeline(1)
        while not (is_last):
            (T4, is_last) = bodo.libs.streaming.sort.produce_output_batch(
                sort_state, True
            )
            bodo.libs.table_builder.table_builder_append(out_tb, T4)
            _iter = _iter + 1
        bodo.libs.streaming.sort.delete_stream_sort_state(sort_state)
        T6 = bodo.libs.table_builder.table_builder_finalize(out_tb)
        bodo.libs.query_profile_collector.end_pipeline(1, _iter)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T6,), index_1, global_2)
        bodo.libs.query_profile_collector.finalize()
        return df2

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0}
    ):
        _ = bodo.jit(distributed=["df"])(impl)(_get_dist_arg(input_df))

    profile_path = get_query_profile_location(tmp_path_rank0, rank)
    with open(profile_path) as f:
        profile_json = json.load(f)
    comm.barrier()

    operator_report = profile_json["operator_reports"]["0"]
    # Verify build metrics
    stage_1_metrics = operator_report["stage_1"]["metrics"]
    build_metrics: list = stage_1_metrics
    build_metrics_dict = {x["name"]: x["stat"] for x in build_metrics}

    if rank == 0:
        # Global metric that will only be reported on rank 0
        assert "final_bytes_per_row" in build_metrics_dict
        assert build_metrics_dict["final_bytes_per_row"] > 0

    # Metrics to expect and whether they are expected to be non-zero
    for exp_metric, exp_nz in [
        ("input_chunks_size_bytes_total", True),
        ("n_input_chunks", not small_limit_case),
        ("input_append_time", not small_limit_case),
        ("sampling_process_input_time", not small_limit_case),
        ("n_samples_taken", not small_limit_case),
        ("final_budget_bytes", True),
        ("global_dict_unification_time", True),
        ("merge_chunks_processing_chunk_size", not small_limit_case),
        ("merge_chunks_K", not small_limit_case),
        ("total_finalize_time", True),
        ("kway_merge_sorter_append_time", not small_limit_case),
        ("get_bounds_total_time", not small_limit_case),
        ("get_bounds_dict_unify_time", not small_limit_case),
        ("get_bounds_gather_samples_time", not small_limit_case),
        ("get_bounds_compute_bounds_time", not small_limit_case),
        ("partition_chunks_total_time", not small_limit_case),
        ("partition_chunks_pin_time", not small_limit_case),
        ("partition_chunks_append_time", not small_limit_case),
        ("partition_chunks_sort_time", not small_limit_case),
        ("partition_chunks_sort_copy_time", not small_limit_case),
        ("partition_chunks_compute_dest_rank_time", not small_limit_case),
        ("n_rows_after_shuffle", not small_limit_case),
        ("kway_merge_sort_total_time", not small_limit_case),
        ("n_merge_levels", False),
        ("merge_chunks_total_time", False),
        ("merge_chunks_make_heap_time", False),
        ("merge_input_builder_finalize_time", not limit_offset_case),
        ("merge_input_builder_total_sort_time", not limit_offset_case),
        ("merge_input_builder_total_sort_copy_time", not limit_offset_case),
        ("merge_n_input_chunks", not limit_offset_case),
        ("merge_approx_input_chunks_total_bytes", not limit_offset_case),
        ("merge_approx_max_input_chunk_size_bytes", not limit_offset_case),
        ("performed_inmem_concat_sort", not limit_offset_case),
        ("finalize_inmem_concat_time", False),
        ("finalize_inmem_sort_time", False),
        ("finalize_inmem_output_append_time", not limit_offset_case),
        ("merge_chunks_output_append_time", False),
        ("n_dict_builders", True),
        ("dict_builders_unify_cache_id_misses", True),
        ("dict_builders_unify_build_transpose_map_time", True),
        ("dict_builders_unify_transpose_time", True),
        ("n_sampling_buffer_rebuilds", False),
        ("n_chunk_merges", False),
        ("merge_chunks_pop_heap_time", False),
        ("merge_chunks_push_heap_time", False),
        ("dict_builders_unify_cache_length_misses", False),
        ("shuffle_barrier_test_time", False),
        # Shuffle related metrics that are expected to be non-zero
        # only when there are multiple processes
        ("shuffle_chunk_size", not small_limit_case and n_pes > 1),
        ("shuffle_total_time", not small_limit_case and n_pes > 1),
        ("shuffle_issend_time", not small_limit_case and n_pes > 1),
        ("shuffle_send_done_check_time", not small_limit_case and n_pes > 1),
        ("shuffle_n_send_done_checks", not small_limit_case and n_pes > 1),
        ("shuffle_irecv_time", not small_limit_case and n_pes > 1),
        ("shuffle_n_irecvs", not small_limit_case and n_pes > 1),
        ("shuffle_recv_done_check_time", not small_limit_case and n_pes > 1),
        ("shuffle_n_recv_done_checks", not small_limit_case and n_pes > 1),
        ("shuffle_n_barrier_tests", not small_limit_case and n_pes > 1),
        ("n_shuffle_send", not small_limit_case and n_pes > 1),
        ("n_shuffle_recv", not small_limit_case and n_pes > 1),
        ("max_concurrent_sends", not small_limit_case and n_pes > 1),
        ("max_concurrent_recvs", not small_limit_case and n_pes > 1),
        ("shuffle_total_sent_nrows", not small_limit_case and n_pes > 1),
        ("shuffle_total_recv_nrows", not small_limit_case and n_pes > 1),
        ("shuffle_total_approx_sent_size_bytes", not small_limit_case and n_pes > 1),
        ("shuffle_approx_sent_size_bytes_dicts", not small_limit_case and n_pes > 1),
        ("shuffle_total_recv_size_bytes", not small_limit_case and n_pes > 1),
        ("approx_recv_size_bytes_dicts", not small_limit_case and n_pes > 1),
        ("dict_unification_time", not small_limit_case and n_pes > 1),
    ]:
        assert exp_metric in build_metrics_dict, exp_metric
        if exp_nz:
            assert build_metrics_dict[exp_metric] > 0, exp_metric

    if limit_offset_case:
        assert "is_limit_offset_case" in build_metrics_dict
        assert build_metrics_dict["is_limit_offset_case"] == 1

        assert "is_small_limit_case" in build_metrics_dict
        assert build_metrics_dict["is_small_limit_case"] == int(small_limit_case)

        if small_limit_case:
            for exp_metric, exp_nz in [
                ("topk_heap_append_chunk_time", True),
                ("topk_heap_update_time", True),
                ("small_limit_local_concat_time", True),
                ("small_limit_gather_time", True),
                ("small_limit_rank0_sort_time", False),
                ("small_limit_rank0_output_append_time", False),
            ]:
                assert exp_metric in build_metrics_dict, exp_metric
                if exp_nz:
                    assert build_metrics_dict[exp_metric] > 0, exp_metric
        else:
            assert "compute_local_limit_time" in build_metrics_dict
            assert build_metrics_dict["compute_local_limit_time"] > 0
