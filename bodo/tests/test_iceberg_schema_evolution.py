import io
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.partition_schema_evolution_tables import (
    PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP,
    create_partition_schema_evolution_tables,
)
from bodo.tests.iceberg_database_helpers.partition_tables import (
    create_partition_tables,
)
from bodo.tests.iceberg_database_helpers.schema_evolution_tables import (
    LIST_UNSUPPORTED_OPERATIONS_TABLES_MAP,
    LIST_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR,
    MAP_UNSUPPORTED_OPERATIONS_TABLES_MAP,
    MAP_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR,
    SCHEMA_EVOLUTION_TABLE_NAME_MAP,
    STRUCT_UNSUPPORTED_OPERATIONS_TABLES_MAP,
    STRUCT_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR,
    create_schema_evolution_tables,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLE_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    append_to_iceberg_table,
    create_iceberg_table,
    get_spark,
)
from bodo.tests.test_iceberg import _check_for_sql_read_head_only
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DistTestPipeline,
    _gather_output,
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    convert_non_pandas_columns,
    reduce_sum,
    run_rank0,
)
from bodo.utils.typing import BodoError

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.slow,
]


@pytest.mark.parametrize(
    "table_name",
    list(SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys()),
)
def test_read_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Basic test for reading tables that have gone through one or more
    schema evolutions. This covers many different data types and many
    different types of evolutions (promotion, nullability, reordering,
    renaming, dropping, etc.) and multiple combinations of these.
    """
    if "STRUCT_FIELD_TYPE_PROMOTION" in table_name:
        pytest.skip(
            reason="Schema evolution within struct fields is not yet supported."
        )
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = run_rank0(spark_reader.read_iceberg_table)(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        # When Spark reads TZ aware columns and converts it to
        # Pandas, the datatype is 'datetime64[ns]' whereas when
        # Bodo reads it, it's 'datetime64[ns, UTC]'.
        check_dtype=(
            False if ("TZ_AWARE" in table_name) or ("DT_TSZ" in table_name) else True
        ),
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "table_name",
    list(SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys()),
)
def test_write_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Basic test for verifying that we can write to tables that have
    undergone various types of schema evolutions and then read data back.
    This covers many different data types and many
    different types of evolutions (promotion, nullability, reordering,
    renaming, dropping, etc.) and multiple combinations of these.
    """
    if ("DECIMALS_TABLE" in table_name) or (
        "DECIMALS_PRECISION_PROMOTION_TABLE" in table_name
    ):
        pytest.skip(reason="Bodo only supports decimals with type (38,18).")
    if "STRUCT_FIELD_TYPE_PROMOTION" in table_name:
        pytest.skip(
            reason="Schema evolution within struct fields is not yet supported."
        )

    db_schema, warehouse_loc = iceberg_database()
    postfix = "_WRITE_TEST"
    run_rank0(create_schema_evolution_tables, bcast_result=False)(
        [table_name], spark=None, postfix=postfix
    )
    table_name = f"{table_name}{postfix}"
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if "DICT_ENCODED" in table_name:
        bodo.hiframes.boxing._use_dict_str_type = True

    @bodo.jit(distributed=["df"])
    def write_then_read_impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")
        return pd.read_sql_table(table_name, conn, db_schema)

    @bodo.jit
    def read_impl(table_name, conn, db_schema):
        return pd.read_sql_table(table_name, conn, db_schema)

    try:
        df_to_append = read_impl(table_name, conn, db_schema)
        bodo_out = write_then_read_impl(
            _get_dist_arg(df_to_append), table_name, conn, db_schema
        )
        bodo_out = _gather_output(bodo_out)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    passed = 1
    if bodo.get_rank() == 0:
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.iceberg_db.{table_name};")
        py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)

        # For easier comparison, convert semi-structured columns in
        # both tables to comparable types:
        py_out = convert_non_pandas_columns(py_out)
        bodo_out = convert_non_pandas_columns(bodo_out)

        passed = _test_equal_guard(
            bodo_out,
            py_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size()


@pytest.mark.parametrize(
    "table_name",
    [
        key
        for key, value in SCHEMA_EVOLUTION_TABLE_NAME_MAP.items()
        if value == "NUMERIC_TABLE"
    ],
)
def test_filter_pushdown(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test for verifying that we can successfully perform filter pushdown
    even when the tables have gone through one or more levels
    of schema evolution.
    """
    if table_name == "NUMERIC_TABLE_RENAME_COLUMN":
        pytest.skip(
            "NUMERIC_TABLE_RENAME_COLUMN test will be fixed in schema evolution PR"
        )

    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @run_rank0
    def get_py_out():
        spark = get_spark()
        df_schema = spark.sql(
            f"select * from hadoop_prod.{db_schema}.{table_name} LIMIT 1"
        )
        col_name = [col for col in df_schema.columns if "B" in col][0]

        py_out = spark.sql(
            f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        WHERE {col_name} = 2
        """
        )
        py_out = py_out.toPandas()
        return py_out, col_name

    py_out, col_name = get_py_out()

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df[col_name] == 2]
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(py_out.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_filter_pushdown_adversarial_renamed_and_swapped_cols(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test for verifying that we can successfully perform filter pushdown
    even when the tables have gone through an adversarial case
    of schema evolution. E.g. Columns B and C have swapped names
    over time and have been re-ordered such that columns originally
    at position 2 & 3 in a table have both swapped names and locations.
    """
    table_name = "filter_pushdown_adversarial"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    @run_rank0
    def setup():
        spark = get_spark()
        sql_schema: list[tuple[str, str, bool]] = [
            ("A", "bigint", True),
            ("B", "bigint", True),
        ]
        create_iceberg_table(
            pd.DataFrame({"A": list(range(5)) * 5, "B": list((range(20, 25))) * 5}),
            sql_schema,
            table_name,
            spark,
        )
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} RENAME COLUMN A to TEMP"
        )
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} RENAME COLUMN B to A"
        )
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} RENAME COLUMN TEMP to B"
        )
        append_to_iceberg_table(
            pd.DataFrame({"A": list(range(5, 10)) * 5, "B": list(range(10, 15)) * 5}),
            sql_schema,
            table_name,
            spark,
        )
        py_out = spark.sql(
            f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        WHERE B = 2
        """
        )
        py_out = py_out.toPandas()
        return py_out

    py_out = setup()

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.B == 2]
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(py_out.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.parametrize("filter", ["IS_NULL", "IS_NOT_NULL", "IS_IN"])
@pytest.mark.parametrize(
    "dtype_cols_to_add_isin_tuple",
    [
        pytest.param(
            (
                "boolean",
                False,
                pd.Series([False, True, pd.NA, False, True] * 20, dtype="boolean"),
                pd.Series([True, False] * 10, dtype="boolean"),
                pd.Series([True, pd.NA, False] * 5, dtype="boolean"),
                pd.Series([pd.NA] * 20, dtype="boolean"),
                (True, False),
            ),
            id="boolean",
        ),
        pytest.param(
            (
                "int",
                False,
                pd.Series([pd.NA, 43, 2445, 67] * 25, dtype="Int32"),
                pd.Series([5, 34, 78, 23] * 5, dtype="Int32"),
                pd.Series([pd.NA, 43, 2445] * 5, dtype="Int32"),
                pd.Series([pd.NA] * 20, dtype="Int32"),
                (5, 43, 23),
            ),
            id="int",
        ),
        pytest.param(
            (
                "long",
                False,
                pd.Series([pd.NA, 43, 2445, 9076] * 25, dtype="Int64"),
                pd.Series([5, 34, 78, 23] * 5, dtype="Int64"),
                pd.Series([pd.NA, 43, 2445] * 5, dtype="Int64"),
                pd.Series([pd.NA] * 20, dtype="Int64"),
                (5, 43, 23),
            ),
            id="long",
        ),
        pytest.param(
            (
                "float",
                False,
                pd.Series([pd.NA, 43.56, 24.45, 908.65] * 25, dtype="Float32"),
                pd.Series([0.64, 0.0, 7.8, 23.3] * 5, dtype="Float32"),
                pd.Series([pd.NA, 43.56, 24.45] * 5, dtype="Float32"),
                pd.Series([pd.NA] * 20, dtype="Float32"),
                # PySpark seems to have a bug where if we add 0.64 to the list
                # it won't actually include the rows in the output while Bodo
                # does. 0.0 seems to match correctly though.
                (0.0, 23.0),
            ),
            id="float",
        ),
        pytest.param(
            (
                "double",
                False,
                pd.Series([pd.NA, 43.56, 24.45, 876.23] * 25, dtype="Float64"),
                pd.Series([0.64, 0.0, 7.8, 23.3] * 5, dtype="Float64"),
                pd.Series([pd.NA, 43.56, 24.45] * 5, dtype="Float64"),
                pd.Series([pd.NA] * 20, dtype="Float64"),
                (0.0, 23.0),
            ),
            id="double",
        ),
        pytest.param(
            (
                "date",
                False,
                pd.Series([pd.NA, date(2033, 12, 12), date(2019, 12, 12), pd.NA] * 25),
                pd.Series(
                    [
                        date(2018, 11, 12),
                        date(2017, 11, 16),
                        date(2019, 11, 12),
                        date(2017, 12, 12),
                    ]
                    * 5
                ),
                pd.Series([pd.NA, date(2033, 12, 12), date(2019, 12, 12)] * 5),
                pd.Series([pd.NA] * 20),
                (
                    date(2017, 11, 16),
                    date(1992, 11, 16),
                    date(2017, 12, 12),
                    date(2019, 12, 12),
                ),
            ),
            id="date",
        ),
        pytest.param(
            (
                "timestamp",
                False,
                pd.Series(
                    [
                        pd.NA,
                        datetime.strptime("11/12/2020", "%d/%m/%Y"),
                        datetime.strptime("11/11/2025", "%d/%m/%Y"),
                        datetime.strptime("11/11/2010", "%d/%m/%Y"),
                    ]
                    * 25
                ),
                pd.Series(
                    [
                        datetime.strptime("12/11/2018", "%d/%m/%Y"),
                        datetime.strptime("11/11/2020", "%d/%m/%Y"),
                        datetime.strptime("12/11/2019", "%d/%m/%Y"),
                        datetime.strptime("13/11/2018", "%d/%m/%Y"),
                    ]
                    * 5
                ),
                pd.Series(
                    [
                        pd.NA,
                        datetime.strptime("11/12/2020", "%d/%m/%Y"),
                        datetime.strptime("11/11/2025", "%d/%m/%Y"),
                    ]
                    * 5
                ),
                pd.Series([pd.NA] * 20),
                (
                    datetime.strptime("12/11/2019", "%d/%m/%Y"),
                    datetime.strptime("12/11/2030", "%d/%m/%Y"),
                ),
            ),
            id="timestamp",
        ),
        pytest.param(
            (
                "string",
                False,
                pd.Series(["medicine", pd.NA, "geography", "political_science"] * 25),
                pd.Series(["history", "literature"] * 10),
                pd.Series(["medicine", pd.NA, "geography"] * 5),
                pd.Series([pd.NA] * 20),
                ("history", "philosophy", "geography"),
            ),
            id="string",
        ),
        pytest.param(
            (
                "string",
                True,
                pd.Series(["medicine", pd.NA, "geography", "political_science"] * 25),
                pd.Series(["history", "literature"] * 10),
                pd.Series(["medicine", pd.NA, "geography"] * 5),
                pd.Series([pd.NA] * 20),
                ("history", "philosophy", "geography"),
            ),
            id="dict_encoded_string",
        ),
        pytest.param(
            (
                "MAP<string, long>",
                False,
                pd.Series([{"78": 908}, pd.NA, {"a": 15}, {"io": 89}] * 25),
                pd.Series([{"a": 10}, {"c": 13}] * 10, dtype=object),
                pd.Series([{"78": 908}, pd.NA, {"a": 15}] * 5),
                pd.Series([pd.NA] * 20),
                None,
            ),
            id="map",
        ),
        pytest.param(
            (
                "ARRAY<long>",
                False,
                pd.Series(
                    [[89, 100002, 43], pd.NA, [890, 23], [890, 23, 1425, 8923]] * 25,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
                pd.Series(
                    [[0, 1, 2], [3, 4]] * 10,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
                pd.Series(
                    [[89, 100002, 43], pd.NA, [890, 23]] * 5,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
                pd.Series([pd.NA] * 20, dtype=pd.ArrowDtype(pa.large_list(pa.int64()))),
                None,
            ),
            id="list",
        ),
        pytest.param(
            (
                "STRUCT<a: string, b: long>",
                False,
                pd.Series(
                    [{"a": "heater", "b": 987}, pd.NA, {"a": "mac", "b": 12}, pd.NA]
                    * 25,
                    dtype=pd.ArrowDtype(
                        pa.struct([("a", pa.string()), ("b", pa.int64())])
                    ),
                ),
                pd.Series(
                    [{"a": "tv", "b": 87}, {"a": "dishwasher", "b": 220}] * 10,
                    dtype=pd.ArrowDtype(
                        pa.struct([("a", pa.string()), ("b", pa.int64())])
                    ),
                ),
                pd.Series(
                    [{"a": "heater", "b": 987}, pd.NA, {"a": "mac", "b": 12}] * 5,
                    dtype=pd.ArrowDtype(
                        pa.struct([("a", pa.string()), ("b", pa.int64())])
                    ),
                ),
                pd.Series(
                    [pd.NA] * 20,
                    dtype=pd.ArrowDtype(
                        pa.struct([("a", pa.string()), ("b", pa.int64())])
                    ),
                ),
                None,
            ),
            id="struct",
        ),
    ],
)
@pytest.mark.parametrize("evol_type", ["ADD", "DROP_THEN_READD"])
def test_filter_pushdown_on_newly_added_column(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
    filter: str,
    dtype_cols_to_add_isin_tuple: tuple[
        str, pd.Series, pd.Series, pd.Series, pd.Series, tuple[Any]
    ],
    evol_type: str,
):
    """
    Test that we can successfully perform filter pushdown even when
    the filter is on a column that didn't exist in previous schemas
    (the ADD case).
    We also test the case where a column was dropped and then re-added
    with the same name (DROP_THEN_READD). The new column will have
    a different field ID.
    Note that in this case, the old parquet files have the old column
    and we *shouldn't* filter on it in those files.
    We test this with various dtypes and three commonly problematic
    filters to verify that the null handling and equality semantics
    are correct.
    """
    (
        dtype,
        use_dict_encoding,
        init_col,
        no_null_col,
        mixed_col,
        all_null_col,
        isin_tuple,
    ) = dtype_cols_to_add_isin_tuple
    if (dtype == "date") and (filter == "IS_IN"):
        # TODO Open a task for this.
        pytest.skip(
            reason="Gap in DNF filter conversion where datetime.date objects are not handled properly when passing them to Java."
        )
    if (dtype == "timestamp") and (filter == "IS_IN"):
        pytest.skip(reason="Series.isin() on Timezone-aware series not yet supported.")
    if (filter == "IS_IN") and (isin_tuple is None):
        pytest.skip(reason="isin_tuple is None")

    db_schema, warehouse_loc = iceberg_database()
    # Start with a simple dataframe
    orig_df = pd.DataFrame(
        {
            "index_col": pd.Series(np.arange(100), dtype="Int64"),
            "dish": pd.Series(["salad", "avocado", "sushi", "pasta"] * 25),
        }
    )
    orig_spark_schema = [
        ("index_col", "long", True),
        ("dish", "string", True),
    ]

    new_column_name = "some_new_column"
    # In the Drop then readd case, we need an existing column
    if evol_type == "DROP_THEN_READD":
        orig_df[new_column_name] = init_col
        orig_spark_schema = orig_spark_schema + [(new_column_name, dtype, True)]

    dtype_san = (
        dtype.replace(" ", "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace(",", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    table_name = f"FILTER_PUSHDOWN_TEST_NEWLY_ADDED_COL_{evol_type}_{dtype_san}_{use_dict_encoding}_{filter}"
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    @run_rank0
    def setup(orig_spark_schema):
        spark = get_spark()
        create_iceberg_table(
            orig_df,
            orig_spark_schema,
            table_name,
            spark,
        )
        if evol_type == "DROP_THEN_READD":
            # Drop the column
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} DROP COLUMN {new_column_name}"
            )

            # Add some data
            append_df = pd.DataFrame(
                {
                    "index_col": pd.Series(np.arange(230, 240), dtype="Int64"),
                    "dish": pd.Series(["eggs", "pancake"] * 5),
                }
            )
            orig_spark_schema = orig_spark_schema[:-1]
            append_to_iceberg_table(append_df, orig_spark_schema, table_name, spark)

        # Add the column (re-add in the DROP_THEN_READD case)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} ADD COLUMN ({new_column_name} {dtype})"
        )
        # Add some data (no nulls).
        append_df = pd.DataFrame(
            {
                "index_col": pd.Series(np.arange(100, 110), dtype="Int64"),
                "dish": pd.Series(["noodles", "kebab"] * 5),
                new_column_name: no_null_col,
            }
        )
        append_sql_schema = orig_spark_schema + [(new_column_name, dtype, True)]
        append_to_iceberg_table(append_df, append_sql_schema, table_name, spark)
        # Add some more data (with nulls).
        append_df = pd.DataFrame(
            {
                "index_col": pd.Series(np.arange(150, 165), dtype="Int64"),
                "dish": pd.Series(["hummus", "tacos", "pizza"] * 5),
                new_column_name: mixed_col,
            }
        )
        append_to_iceberg_table(append_df, append_sql_schema, table_name, spark)
        # Add some more data (all nulls).
        append_df = pd.DataFrame(
            {
                "index_col": pd.Series(np.arange(170, 190), dtype="Int64"),
                "dish": pd.Series(["wrap"] * 20),
                new_column_name: all_null_col,
            }
        )
        append_to_iceberg_table(append_df, append_sql_schema, table_name, spark)

    setup(orig_spark_schema)

    # Form the filter in Spark and Bodo:
    _bodo_read_as_dict = []
    if use_dict_encoding:
        assert dtype == "string"
        _bodo_read_as_dict = [new_column_name]
    spark_sql_query: str | None = None
    bodo_impl = None
    expected_filter_pushdown_logs = []
    if filter == "IS_NULL":
        spark_sql_query = f"select * from hadoop_prod.{db_schema}.{table_name} WHERE {new_column_name} IS NULL"

        if use_dict_encoding:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_read_as_dict=_bodo_read_as_dict,
                    _bodo_detect_dict_cols=False,
                )
                df = df[pd.isna(df[new_column_name])]
                return df

            bodo_impl = impl
        else:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_detect_dict_cols=False,
                )
                df = df[pd.isna(df[new_column_name])]
                return df

            bodo_impl = impl

        expected_filter_pushdown_logs = [
            "[[('some_new_column', 'is', 'NULL')]]",
            "(((ds.field('{some_new_column}').is_null())))",
        ]

    elif filter == "IS_NOT_NULL":
        spark_sql_query = f"select * from hadoop_prod.{db_schema}.{table_name} WHERE {new_column_name} IS NOT NULL"

        if use_dict_encoding:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_read_as_dict=_bodo_read_as_dict,
                    _bodo_detect_dict_cols=False,
                )
                df = df[~pd.isna(df[new_column_name])]
                return df

            bodo_impl = impl
        else:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_detect_dict_cols=False,
                )
                df = df[~pd.isna(df[new_column_name])]
                return df

            bodo_impl = impl

        expected_filter_pushdown_logs = [
            "((~((ds.field('{some_new_column}').is_null()))))",
        ]

    elif filter == "IS_IN":
        isin_str = isin_tuple
        if dtype in ["date", "timestamp"]:
            isin_str = tuple([str(d) for d in isin_tuple])
        spark_sql_query = f"select * from hadoop_prod.{db_schema}.{table_name} WHERE {new_column_name} IN {isin_str}"

        if use_dict_encoding:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_read_as_dict=_bodo_read_as_dict,
                    _bodo_detect_dict_cols=False,
                )
                df = df[df[new_column_name].isin(list(isin_tuple))]
                return df

            bodo_impl = impl
        else:

            def impl(table_name, conn, db_schema):
                df = pd.read_sql_table(
                    table_name,
                    conn,
                    db_schema,
                    _bodo_detect_dict_cols=False,
                )
                df = df[df[new_column_name].isin(list(isin_tuple))]
                return df

            bodo_impl = impl

        expected_filter_pushdown_logs = [
            "[[('some_new_column', 'in', f0)]]",
            "(((ds.field('{some_new_column}').isin(f0))))",
        ]
    else:
        raise ValueError(f"Unrecognized filter: {filter}")

    # Read with the filter applied.
    @run_rank0
    def get_py_out():
        spark = get_spark()
        py_out = spark.sql(spark_sql_query).toPandas()
        return py_out

    py_out = get_py_out()

    if filter in ("IS_NULL", "IS_NOT_NULL") and use_dict_encoding:
        pytest.skip(
            reason="[BSE-2790] Known bug in Arrow that leads to incorrect output."
        )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(bodo_impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(py_out.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")
        for expected_log in expected_filter_pushdown_logs:
            check_logger_msg(stream, expected_log)

    check_func(
        bodo_impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "streaming",
    [pytest.param(False, id="non_streaming"), pytest.param(True, id="streaming")],
)
def test_filter_pushdown_filter_on_pruned_column(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
    streaming,
):
    """
    Test that we can successfully apply filter pushdown with column
    pruning in cases where the filter is on a column
    that is not part of the final output, i.e. the column is pruned.
    This stress-tests some of the ordering semantics and guarantees
    in our code. We test this with both streaming and non-streaming.
    """
    table_name = "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @run_rank0
    def get_py_out():
        spark = get_spark()

        py_out = spark.sql(
            f"""
        select A, G, B, TY from hadoop_prod.{db_schema}.{table_name}
        where C IN (3, 5) and TY IS NOT NULL
        """
        ).toPandas()
        return py_out

    py_out = get_py_out()

    col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "TY", "F", "G"))
    bodo_impl = None
    if streaming:

        def impl(table_name, conn, db_schema):
            is_last_global = False
            reader = pd.read_sql_table(table_name, conn, db_schema, _bodo_chunksize=64)  # type: ignore
            out_dfs = []
            while not is_last_global:
                table, is_last = read_arrow_next(reader, True)
                index_var = bodo.hiframes.pd_index_ext.init_range_index(
                    0, len(table), 1, None
                )
                df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (table,), index_var, col_meta
                )
                df = df[(df.C.isin([3, 5])) & (~pd.isna(df.TY))]
                df = df[["A", "G", "B", "TY"]]
                out_dfs.append(df)

                is_last_global = bodo.libs.distributed_api.dist_reduce(
                    is_last,
                    np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
                )

            arrow_reader_del(reader)
            return pd.concat(out_dfs)

        bodo_impl = impl
    else:

        def impl(table_name, conn, db_schema):
            df = pd.read_sql_table(table_name, conn, db_schema)
            df = df[(df.C.isin([3, 5])) & (~pd.isna(df.TY))]
            return df[["A", "G", "B", "TY"]]

        bodo_impl = impl

    check_func(
        bodo_impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(bodo_impl)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns loaded ['A', 'B', 'TY', 'G']")
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "[[('C', 'in', f0)]]")
        check_logger_msg(
            stream, "(((ds.field('{C}').isin(f0)) & ~((ds.field('{TY}').is_null()))))"
        )


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
@pytest.mark.slow
def test_limit_pushdown(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Tests that Limit Pushdown is successfully enabled even when the
    table has gone through schema evolution.
    """
    table_name = "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    comm = MPI.COMM_WORLD

    def impl():
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df.head(5)  # type: ignore

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    bodo_out = bodo_func()
    _check_for_sql_read_head_only(bodo_func, 5)

    assert comm.allreduce(bodo_out.shape[0], op=MPI.SUM) == 5


@pytest.mark.parametrize(
    "table_name",
    [
        key
        for key, value in PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP.items()
        if (("DT_TSZ" in value) or ("STRING" in value))
    ],
)
def test_read_partition_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn
):
    """
    Test that we can read from tables that have gone through
    both schema and partition evolution, including schema evolution
    on the partition columns and vice-versa. We check this on
    date/timestamp and string types since those are the most
    critical. This is also to limit the number of tests. For
    more exhaustive tests, one can remove the filter in the
    pytest parameter for 'table_name'.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = run_rank0(spark_reader.read_iceberg_table)(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        # When Spark reads TZ aware columns and converts it to
        # Pandas, the datatype is 'datetime64[ns]' whereas when
        # Bodo reads it, it's 'datetime64[ns, UTC]'.
        check_dtype=(
            False if ("TZ_AWARE" in table_name) or ("DT_TSZ" in table_name) else True
        ),
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        key
        for key, value in PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP.items()
        if (("DT_TSZ" in value) or ("DICT_ENCODED" in value))
    ],
)
def test_write_partition_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn
):
    """
    Test that we can write to and then read from tables that
    have gone through both schema and partition evolution,
    including schema evolution on the partition columns and
    vice-versa. We check this on date/timestamp and
    string (dict-encoding) types since those are the most
    critical. This is also to limit the number of tests. For
    more exhaustive tests, one can remove
    the filter in the pytest parameter for 'table_name'.
    """
    db_schema, warehouse_loc = iceberg_database()
    postfix = "_WRITE_TEST"

    run_rank0(create_partition_schema_evolution_tables, bcast_result=False)(
        [table_name], spark=None, postfix=postfix
    )
    table_name = f"{table_name}{postfix}"
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if "DICT_ENCODED" in table_name:
        bodo.hiframes.boxing._use_dict_str_type = True

    @bodo.jit(distributed=["df"])
    def write_then_read_impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")
        return pd.read_sql_table(table_name, conn, db_schema)

    @bodo.jit
    def read_impl(table_name, conn, db_schema):
        return pd.read_sql_table(table_name, conn, db_schema)

    try:
        df_to_append = read_impl(table_name, conn, db_schema)
        bodo_out = write_then_read_impl(
            _get_dist_arg(df_to_append), table_name, conn, db_schema
        )
        bodo_out = _gather_output(bodo_out)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    passed = 1
    if bodo.get_rank() == 0:
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.iceberg_db.{table_name};")
        py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

        # For easier comparison, convert semi-structured columns in
        # both tables to comparable types:
        py_out = convert_non_pandas_columns(py_out)
        bodo_out = convert_non_pandas_columns(bodo_out)

        passed = _test_equal_guard(
            bodo_out,
            py_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size()


@pytest.mark.parametrize(
    "table_name",
    [
        key
        for key, value in PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP.items()
        if "NUMERIC_TABLE" in value
    ],
)
def test_partition_schema_evolved_table_filter_pushdown(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that we can successfully perform filter pushdown when
    reading from tables that have gone through both schema and
    partition evolution, including schema evolution on the partition
    columns and vice-versa.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @run_rank0
    def get_py_out():
        spark = get_spark()
        df_schema = spark.sql(
            f"select * from hadoop_prod.{db_schema}.{table_name} LIMIT 1"
        )
        col_name = [col for col in df_schema.columns if "B" in col][0]

        py_out = spark.sql(
            f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        WHERE {col_name} = 2
        """
        )
        py_out = py_out.toPandas()
        return py_out, col_name

    py_out, col_name = get_py_out()

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df[col_name] == 2]
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(py_out.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_mixed_partition_schema_evolution(iceberg_database, iceberg_table_conn):
    """
    Smoke test that we can read from tables that have gone through
    both schema and partition evolution.
    """
    db_schema, warehouse_loc = iceberg_database()
    postfix = "_MIXED_PARTITION_SCHEMA_EVOLUTION"
    part_table_name = "part_NUMERIC_TABLE_A_bucket_50"
    table_name = f"{part_table_name}{postfix}"

    df, sql_schema = SIMPLE_TABLE_MAP["SIMPLE_NUMERIC_TABLE"]
    df = df.copy()
    sql_schema = sql_schema.copy()

    @run_rank0
    def setup(df, sql_schema):
        spark = get_spark()
        create_partition_tables([part_table_name], spark=spark, postfix=postfix)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} REPLACE PARTITION FIELD Bucket(50, A) WITH Bucket(40, A)"
        )
        append_to_iceberg_table(df, sql_schema, table_name, spark)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} ALTER COLUMN A TYPE bigint"
        )
        sql_schema = [
            (col, "bigint" if col == "A" else type, nullable)
            for (col, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table_name, spark)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} REPLACE PARTITION FIELD Bucket(40, A) WITH Bucket(30, A)"
        )
        append_to_iceberg_table(df, sql_schema, table_name, spark)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} ALTER COLUMN B FIRST"
        )
        sql_schema = [schema for schema in sql_schema if schema[0] == "B"] + [
            schema for schema in sql_schema if schema[0] != "B"
        ]
        col_order = [schema[0] for schema in sql_schema]
        df = df[col_order]
        append_to_iceberg_table(df, sql_schema, table_name, spark)
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name} DROP PARTITION FIELD Bucket(30, A)"
        )
        append_to_iceberg_table(df, sql_schema, table_name, spark)
        py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        return py_out

    py_out = setup(df, sql_schema)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    list(STRUCT_UNSUPPORTED_OPERATIONS_TABLES_MAP.keys()),
)
def test_unsupported_struct_operations(
    table_name, iceberg_database, iceberg_table_conn
):
    """
    Test that we can detect schema evolution within struct
    fields (which is unsupported) and can raise reasonable
    errors when we encounter this.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    expected_error_msg = STRUCT_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR[
        table_name
    ]
    with pytest.raises(BodoError, match=re.escape(expected_error_msg)):
        bodo.jit(impl)(table_name, conn, db_schema)


@pytest.mark.parametrize(
    "table_name",
    list(MAP_UNSUPPORTED_OPERATIONS_TABLES_MAP.keys()),
)
def test_unsupported_map_operations(table_name, iceberg_database, iceberg_table_conn):
    """
    Test that we can detect schema evolution within map
    fields (which is unsupported) and can raise reasonable
    errors when we encounter this.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    expected_error_msg = MAP_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR[
        table_name
    ]

    with pytest.raises(
        BodoError,
        match=re.escape(expected_error_msg),
    ):
        bodo.jit(impl)(table_name, conn, db_schema)


@pytest.mark.parametrize(
    "table_name",
    list(LIST_UNSUPPORTED_OPERATIONS_TABLES_MAP.keys()),
)
def test_unsupported_list_operations(table_name, iceberg_database, iceberg_table_conn):
    """
    Test that we can detect schema evolution within list
    fields (which is unsupported) and can raise reasonable
    errors when we encounter this.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    expected_error_msg = LIST_UNSUPPORTED_OPERATIONS_TABLES_MAP_EXPECTED_ERROR[
        table_name
    ]

    with pytest.raises(
        BodoError,
        match=re.escape(expected_error_msg),
    ):
        bodo.jit(impl)(table_name, conn, db_schema)
