import json
import os
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
from bodo.libs.stream_groupby import (
    delete_groupby_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.libs.stream_join import (
    delete_join_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import (
    _get_dist_arg,
    get_snowflake_connection_string,
    pytest_mark_snowflake,
    reduce_sum,
    temp_env_override,
)


def test_query_profile_collection_compiles(memory_leak_check):
    """Check that all query profile collector functions compile"""

    @bodo.jit
    def impl():
        bodo.libs.query_profile_collector.init()
        bodo.libs.query_profile_collector.start_pipeline(1)
        bodo.libs.query_profile_collector.end_pipeline(1, 10)
        bodo.libs.query_profile_collector.finalize()
        return

    impl()


def test_join_row_count_collection(memory_leak_check):
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


def test_groupby_row_count_collection(memory_leak_check):
    """
    Check that Groupby submits its row counts to the QueryProfileCollector
    as expected.
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

    keys_inds = bodo.utils.typing.MetaType(tuple([0]))
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
            is_last1 = groupby_build_consume_batch(groupby_state, T3, is_last1, True)
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
    build_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 0)
    )
    produce_output_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 1)
    )
    build_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 0)
    )
    produce_output_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    build_input_row_count = reduce_sum(build_input_row_count)
    produce_output_input_row_count = reduce_sum(produce_output_input_row_count)
    build_output_row_count = reduce_sum(build_output_row_count)
    produce_output_output_row_count = reduce_sum(produce_output_output_row_count)
    assert (
        build_input_row_count == 32000
    ), f"Expected build_input_row_count to be 32000 but it was {build_input_row_count} instead"
    assert (
        produce_output_input_row_count == 0
    ), f"Expected produce_output_input_row_count to be 0 but it was {produce_output_input_row_count} instead"
    assert (
        build_output_row_count == 0
    ), f"Expected build_output_row_count to be 0 but it was {build_output_row_count} instead"
    assert (
        produce_output_output_row_count == 1000
    ), f"Expected produce_output_output_row_count to be 1000 but it was {produce_output_output_row_count} instead"


@pytest_mark_snowflake
def test_snowflake_read_row_count_collection(memory_leak_check):
    """
    Check that Snowflake Reader submits its row counts to the QueryProfileCollector
    as expected.
    """

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
    reader_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_input_row_count = reduce_sum(reader_input_row_count)
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert (
        reader_input_row_count == 0
    ), f"Expected reader_input_row_count to be 0, but it was {reader_input_row_count} instead."
    assert (
        reader_output_row_count == 6001215
    ), f"Expected reader_output_row_count to be 6001215, but it was {reader_output_row_count} instead."


@pytest.mark.iceberg
def test_iceberg_read_row_count_collection(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Check that Iceberg Reader submits its row counts to the QueryProfileCollector
    as expected.
    """

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
    reader_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_input_row_count = reduce_sum(reader_input_row_count)
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert (
        reader_input_row_count == 0
    ), f"Expected reader_input_row_count to be 0, but it was {reader_input_row_count} instead."
    assert (
        reader_output_row_count == 200
    ), f"Expected reader_output_row_count to be 200, but it was {reader_output_row_count} instead."


@pytest.mark.parquet
def test_parquet_read_row_count_collection(datapath, memory_leak_check):
    """
    Check that Parquet Reader submits its row counts to the QueryProfileCollector
    as expected.
    """

    @bodo.jit()
    def impl(path):
        bodo.libs.query_profile_collector.init()
        total_max = 0
        is_last_global = False
        reader = pd.read_parquet(
            path, _bodo_use_index=False, _bodo_chunksize=4096, _bodo_sql_op_id=0
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

    _ = impl(datapath("tpch-test_data/parquet/lineitem.parquet"))
    reader_input_row_count = (
        bodo.libs.query_profile_collector.get_input_row_counts_for_op_stage(0, 1)
    )
    reader_output_row_count = (
        bodo.libs.query_profile_collector.get_output_row_counts_for_op_stage(0, 1)
    )
    reader_input_row_count = reduce_sum(reader_input_row_count)
    reader_output_row_count = reduce_sum(reader_output_row_count)
    assert (
        reader_input_row_count == 0
    ), f"Expected reader_input_row_count to be 0, but it was {reader_input_row_count} instead."
    assert (
        reader_output_row_count == 120515
    ), f"Expected reader_output_row_count to be 120515, but it was {reader_output_row_count} instead."


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


def test_hash_join_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by hash join.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tmp_path_rank0 = comm.bcast(str(tmp_path))
    build_keys_inds = bodo.utils.typing.MetaType((0, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D"))
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(("E", "F", "G", "H"))
    col_meta = bodo.utils.typing.ColNamesMetaType(
        ("E", "F", "G", "H", "A", "B", "C", "D")
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
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 4
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

    assert os.path.isfile(os.path.join(tmp_path_rank0, f"query_profile_{rank}.json"))
    with open(os.path.join(tmp_path_rank0, f"query_profile_{rank}.json"), "r") as f:
        profile_json = json.load(f)

    assert "operator_stages" in profile_json

    # Verify build metrics
    assert "0" in profile_json["operator_stages"]
    assert "metrics" in profile_json["operator_stages"]["0"]
    build_metrics: list = profile_json["operator_stages"]["0"]["metrics"]
    build_metrics_names: set[str] = set([x["name"] for x in build_metrics])
    if rank == 0:
        assert "bcast_join" in build_metrics_names
        assert "bloom_filter_enabled" in build_metrics_names
        assert "n_key_dict_builders" in build_metrics_names
        assert "n_non_key_dict_builders" in build_metrics_names
    assert "shuffle_buffer_append_time" in build_metrics_names
    assert "ht_hashing_time" in build_metrics_names
    assert "repartitioning_time_total" in build_metrics_names

    # Verify probe metrics
    assert "1" in profile_json["operator_stages"]
    assert "metrics" in profile_json["operator_stages"]["1"]
    probe_metrics: list = profile_json["operator_stages"]["1"]["metrics"]
    probe_metrics_names: set[str] = set([x["name"] for x in probe_metrics])
    if rank == 0:
        assert "n_key_dict_builders" in probe_metrics_names
        assert "n_non_key_dict_builders" in probe_metrics_names
    assert "output_append_time" in probe_metrics_names
    assert "output_total_nrows" in probe_metrics_names
    assert "output_total_nrows_rem_at_finalize" in probe_metrics_names
    assert "output_peak_nrows" in probe_metrics_names
    assert "shuffle_buffer_append_time" in probe_metrics_names
    assert "ht_probe_time" in probe_metrics_names
    assert "finalize_inactive_partitions_total_time" in probe_metrics_names
    assert "join_filter_materialization_time" in probe_metrics_names


def test_nested_loop_join_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by nested loop join.
    """

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
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)
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
        {
            "BODO_TRACING_LEVEL": "1",
            "BODO_TRACING_OUTPUT_DIR": tmp_path_rank0,
        }
    ):
        _ = impl(_get_dist_arg(build_df), _get_dist_arg(probe_df))

    assert os.path.isfile(os.path.join(tmp_path_rank0, f"query_profile_{rank}.json"))
    with open(os.path.join(tmp_path_rank0, f"query_profile_{rank}.json"), "r") as f:
        profile_json = json.load(f)

    assert "operator_stages" in profile_json

    # Verify build metrics
    assert "0" in profile_json["operator_stages"]
    assert "metrics" in profile_json["operator_stages"]["0"]
    build_metrics: list = profile_json["operator_stages"]["0"]["metrics"]
    build_metrics_names: set[str] = set([x["name"] for x in build_metrics])
    if rank == 0:
        assert "bcast_join" in build_metrics_names
        assert "block_size_bytes" in build_metrics_names
        assert "chunk_size_nrows" in build_metrics_names
        assert "n_dict_builders" in build_metrics_names
    assert "append_time" in build_metrics_names
    assert "num_chunks" in build_metrics_names

    # Verify probe metrics
    assert "1" in profile_json["operator_stages"]
    assert "metrics" in profile_json["operator_stages"]["1"]
    probe_metrics: list = profile_json["operator_stages"]["1"]["metrics"]
    probe_metrics_names: set[str] = set([x["name"] for x in probe_metrics])
    if rank == 0:
        assert "n_dict_builders" in probe_metrics_names
    assert "output_append_time" in probe_metrics_names
    assert "output_total_nrows" in probe_metrics_names
    assert "output_total_nrows_rem_at_finalize" in probe_metrics_names
    assert "output_peak_nrows" in probe_metrics_names
    assert "global_dict_unification_time" in probe_metrics_names
    assert "bcast_size_bytes" in probe_metrics_names
    assert "bcast_table_time" in probe_metrics_names
    assert "compute_matches_time" in probe_metrics_names
