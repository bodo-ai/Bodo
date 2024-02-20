import io

import pandas as pd
import pytest

import bodo
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.schema_evolution_tables import (
    SCHEMA_EVOLUTION_TABLE_NAME_MAP,
    create_schema_evolution_tables,
)
from bodo.tests.iceberg_database_helpers.utils import (
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
from bodo.tests.utils import DistTestPipeline, _test_equal_guard, check_func

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.slow,
    # pytest.mark.skip("Not implemented yet"),
]


@pytest.mark.parametrize(
    "table_name",
    list(SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys()),
)
def test_read_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    # check_func(
    #    impl,
    #    (table_name, conn, db_schema),
    #    py_output=py_out,
    #    sort_output=True,
    #    reset_index=True,
    # )


@pytest.mark.parametrize(
    "table_name",
    list(SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys()),
)
def test_write_schema_evolved_table(
    table_name, iceberg_database, iceberg_table_conn, memory_leak_check
):
    db_schema, warehouse_loc = iceberg_database()
    postfix = "_WRITE_TEST"
    if bodo.get_rank() == 0:
        create_schema_evolution_tables([table_name], spark=None, postfix=postfix)
    bodo.barrier()
    table_name = f"{table_name}{postfix}"
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @bodo.jit
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")
        return pd.read_sql_table(table_name, conn, db_schema)

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    # _test_equal_guard(
    #     impl(df, table_name, conn, db_schema),
    #     py_out,
    #     sort_output=True,
    #     check_dtype=False,
    #     reset_index=True,
    # )


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
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.B == 2]
        return df

    spark = get_spark()

    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    WHERE B = 2
    """
    )
    py_out = py_out.toPandas()
    # check_func(
    #    impl,
    #    (table_name, conn, db_schema),
    #    py_output=py_out,
    #    sort_output=True,
    #    reset_index=True,
    #    check_dtype=False,
    # )
    # stream = io.StringIO()
    # logger = create_string_io_logger(stream)
    # with set_logging_stream(logger, 1):
    #    bodo.jit(impl)(table_name, conn, db_schema)
    #    check_logger_msg(stream, f"Columns loaded {list(df.columns)})"
    #    check_logger_msg(stream, "Filter pushdown successfully performed")
    #


def test_filter_pushdown_adversarial(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    table_name = "filter_pushdown_adversarial"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    spark = get_spark()
    sql_schema: list[tuple[str, str, bool]] = [
        ("A", "bigint", True),
        ("B", "bigint", True),
    ]

    if bodo.get_rank() == 0:
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
    bodo.barrier()

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.B == 2]
        return df

    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    WHERE B = 2
    """
    )
    py_out = py_out.toPandas()
    # check_func(
    #    impl,
    #    (table_name, conn, db_schema),
    #    py_output=py_out,
    #    sort_output=True,
    #    reset_index=True,
    #    check_dtype=False,
    # )
    # stream = io.StringIO()
    # logger = create_string_io_logger(stream)
    # with set_logging_stream(logger, 1):
    #    bodo.jit(impl)(table_name, conn, db_schema)
    #    check_logger_msg(stream, f"Columns loaded {list(df.columns)})"
    #    check_logger_msg(stream, "Filter pushdown successfully performed")
    #


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
@pytest.mark.parametrize(
    "table_name",
    [
        key
        for key, value in SCHEMA_EVOLUTION_TABLE_NAME_MAP.items()
        if value == "NUMERIC_TABLE"
    ],
)
@pytest.mark.slow
def test_limit_pushdown(table_name, iceberg_database, iceberg_table_conn):
    """Test that Limit Pushdown is successfully enabled"""
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl():
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df.head(5)  # type: ignore

    spark = get_spark()
    py_out = spark.sql(f"select * from hadoop_prod.{db_schema}.{table_name} LIMIT 5;")
    py_out = py_out.toPandas()

    # check_func(
    #    impl,
    #    (),
    #    py_output=py_out,
    #    sort_output=True,
    #    reset_index=True,
    # )

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    # bodo_func()
    # _check_for_sql_read_head_only(bodo_func, 5)


# TODO Add a test using the unsupported schema_evolution_tables
