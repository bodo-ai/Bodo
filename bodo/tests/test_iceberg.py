import datetime

import pandas as pd
import pytest

import bodo
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.utils import get_spark
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg


@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "simple_map_table",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "simple_numeric_table",
        "simple_string_table",
        "partitions_dt_table",
        # TODO: The results of Bodo and Spark implemenation are different from original
        # but only in check_func
        pytest.param("simple_dt_tsz_table", marks=pytest.mark.slow),
    ],
)
def test_simple_table_read(iceberg_database, iceberg_table_conn, table_name):
    """
    Test simple read operation on test tables
    """

    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
def test_simple_list_table_read(iceberg_database, iceberg_table_conn):
    """
    Test reading simple_list_table which consists of columns of lists.
    Need to compare Bodo and PySpark results without sorting them.
    """
    table_name = "simple_list_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
        # No sorting because lists are not hashable
    )


def test_simple_bool_binary_table_read(iceberg_database, iceberg_table_conn):
    """
    Test reading simple_bool_binary_table which consists of boolean
    and binary typs (bytes). Needs special handling to compare
    with PySpark.
    """
    table_name = "simple_bool_binary_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    # Bodo outputs binary data as bytes while Spark does bytearray (which Bodo doesn't support),
    # so we convert Spark output.
    # This has been copied from BodoSQL. See `convert_spark_bytearray`
    # in `bodosql/tests/utils.py`.
    py_out[["C"]] = py_out[["C"]].apply(
        lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
        axis=1,
        result_type="expand",
    )
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def test_simple_struct_table_read(iceberg_database, iceberg_table_conn):
    """
    Test reading simple_struct_table which consists of columns of structs.
    Needs special handling since PySpark returns nested structs as tuples.
    """
    table_name = "simple_struct_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    # Convert columns with nested structs from tuples to dictionaries with correct keys
    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out["A"] = py_out["A"].map(lambda x: {"a": x[0], "b": x[1]})
    py_out["B"] = py_out["B"].map(lambda x: {"a": x[0], "b": x[1], "c": x[2]})

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
    )


def test_no_files_after_filter_pushdown(iceberg_database, iceberg_table_conn):
    """
    Test the use case where Iceberg filters out all files
    based on the provided filters. We need to load an empty
    dataframe with the right schema in this case.
    """

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["TY"].isna()]
        return df

    spark = get_spark()
    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    where TY IS NULL;
    """
    )
    py_out = py_out.toPandas()
    assert (
        py_out.shape[0] == 0
    ), f"Expected DataFrame to be empty, found {py_out.shape[0]} rows instead."

    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


def test_filter_pushdown_partitions(iceberg_database, iceberg_table_conn):
    """
    Test that simple date based partitions can be read as expected.
    """

    table_name = "partitions_dt_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["A"] <= datetime.date(2018, 12, 12)]
        return df

    spark = get_spark()
    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    where A <= '2018-12-12';
    """
    )
    py_out = py_out.toPandas()

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def test_schema_evolution_detection(iceberg_database, iceberg_table_conn):
    """
    Test that we throw the right error when dataset has schema evolution,
    which we don't support yet. This test should be removed once
    we add support for it.
    """

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[(df["TY"].notnull()) & (df["B"] > 10)]
        return df

    with pytest.raises(
        BodoError,
        match="Bodo currently doesn't support reading Iceberg tables with schema evolution.",
    ):
        bodo.jit(impl)(table_name, conn, db_schema)


def test_iceberg_invalid_table(iceberg_database, iceberg_table_conn):
    """Tests error raised when a nonexistent Iceberg table is provided."""

    table_name = "no_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


def test_iceberg_invalid_path(iceberg_database, iceberg_table_conn):
    """Tests error raised when invalid path is provided."""

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    db_schema += "not"

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)
