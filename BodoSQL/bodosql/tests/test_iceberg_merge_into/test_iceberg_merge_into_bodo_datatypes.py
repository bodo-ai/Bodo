from datetime import date

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import check_func

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.skip(
        reason="[BSE-4569] MERGE INTO with PyIceberg is not supported yet"
    ),
]

bodo_datatype_cols = {
    "INT_COL": pd.Series([np.int32(i) for i in range(10)], dtype=np.int32),
    "FLOAT_COL": pd.Series([np.float32(i) for i in range(10)], dtype=np.float32),
    "STR_COL": pd.Series([str(i) for i in range(10)], dtype="string[pyarrow]"),
    "BOOL_COL": pd.Series([bool(i % 2) for i in range(10)], dtype=bool),
    # TODO: resolve this issue: https://bodo.atlassian.net/browse/BE-4072
    # "TS_COL": pd.Series(
    #     [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i) for i in range(10)],
    # , dtype="datetime64[ns, UTC]"),
    "NON_ASCII_COL": pd.Series(
        [str(i) + "é" for i in range(10)], dtype="string[pyarrow]"
    ),
    "BYTE_COL": pd.Series([bytes(str(i), "utf8") for i in range(10)], dtype="bytes"),
    "LONG_COL": pd.Series([np.int64(i) for i in range(10)], dtype=np.int64),
    "DOUBLE_COL": pd.Series([np.float64(i) for i in range(10)], dtype=np.float64),
    "DATE_COL": pd.Series(
        [
            date(2018, 11, 12),
            date(2019, 11, 12),
            date(2018, 12, 12),
            date(2017, 11, 16),
            date(2020, 7, 11),
            date(2017, 11, 30),
            date(2016, 2, 3),
            date(2019, 11, 12),
            date(2018, 12, 20),
            date(2017, 12, 12),
        ]
    ),
}
bodo_datatype_expected_sql_types = {
    "INT_COL": "int",
    "FLOAT_COL": "float",
    "STR_COL": "string",
    "BOOL_COL": "boolean",
    # TODO: resolve this issue: https://bodo.atlassian.net/browse/BE-4072
    # "TS_COL": "timestamp",  # Spark writes timestamps with UTC timezone
    "NON_ASCII_COL": "string",
    "BYTE_COL": "binary",
    "LONG_COL": "bigint",
    "DOUBLE_COL": "double",
    "DATE_COL": "date",
}


@pytest.mark.slow
@pytest.mark.timeout(600)
# NOTE (allai5): Arbitrary high timeout number due to inability to replicate
# timeout locally
def test_merge_into_bodo_datatypes_as_values(iceberg_database, iceberg_table_conn):
    """
    Test MERGE INTO with all Bodo datatypes as values.
    """

    # create table data
    target_table = pd.DataFrame(
        {
            "ID": pd.Series(list(range(10)), dtype=np.int32),
        }
        | bodo_datatype_cols
    )

    source = target_table.copy()
    source.ID = source.ID + 10
    expected = pd.concat((target_table, source), ignore_index=True)

    # create query
    query = (
        "MERGE INTO target_table AS t USING source AS s "
        "ON t.id = s.id "
        "WHEN NOT MATCHED THEN "
        f"  INSERT ({', '.join(source.columns)}) "
        f"    values ({', '.join(['s.' + key for key in source.columns])})"
    )

    # create BodoSQL context
    spark = get_spark()
    table_name = "TARGET_TABLE_MERGE_INTO_BODO_DATATYPES_AS_VALUES"
    db_schema, warehouse_loc = iceberg_database()
    sql_schema = [("ID", "int", False)] + [
        (col, bodo_datatype_expected_sql_types[col], False)
        for col in bodo_datatype_cols.keys()
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
            spark,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "TARGET_TABLE": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "SOURCE": source,
        }
    )

    # write target table
    bc.add_or_replace_view(table_name, target_table)

    # execute query
    def impl(bc, query):
        bc.sql(query)

        return bc.sql("SELECT * FROM target_table ORDER BY id")

    check_func(
        impl,
        (bc, query),
        py_output=expected,
        reset_index=True,
        sort_output=True,
        only_1DVar=True,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
@pytest.mark.parametrize("col_name", bodo_datatype_cols.keys())
def test_merge_into_bodo_datatypes_as_expr(
    col_name: str, iceberg_database, iceberg_table_conn
):
    """
    Test MERGE INTO with individual Bodo datatypes as join expression.
    """
    expr = bodo_datatype_cols[col_name]
    sql_type = bodo_datatype_expected_sql_types[col_name]

    # create table data
    target_table = pd.DataFrame(
        {
            "EXPR": expr[:7],
        }
    )

    source = pd.DataFrame(
        {
            "EXPR": expr[3:],
        }
    )

    if col_name == "BOOL_COL":
        expected = target_table.copy()
    else:
        expected = target_table.merge(source, on="EXPR", how="outer")

    # create query
    query = (
        "MERGE INTO target_table AS t USING source AS s "
        "ON t.expr = s.expr "
        "WHEN NOT MATCHED THEN "
        "  INSERT (expr) values (s.expr)"
    )

    # create BodoSQL context
    spark = get_spark()
    db_schema, warehouse_loc = iceberg_database()
    table_name = "TARGET_TABLE_MERGE_INTO_BODO_DATATYPES_AS_EXPRS_" + col_name
    sql_schema = [("EXPR", sql_type, False)]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
            spark,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "TARGET_TABLE": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "SOURCE": source,
        }
    )

    # write target table
    bc.add_or_replace_view(table_name, target_table)

    # execute query
    def impl(bc, query):
        bc.sql(query)

        return bc.sql("SELECT * FROM target_table")

    check_func(
        impl,
        (bc, query),
        py_output=expected,
        reset_index=True,
        sort_output=True,
        only_1DVar=True,
        check_dtype=False,
    )
