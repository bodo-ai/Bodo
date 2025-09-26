from io import StringIO

import pandas as pd
import pyarrow
import pytest

from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    get_snowflake_connection_string,
    pytest_mark_one_rank,
    pytest_snowflake,
)

pytestmark = [pytest.mark.iceberg, pytest.mark.jit_dependency] + pytest_snowflake


@pytest_mark_one_rank
def test_get_iceberg_schema_snowflake(memory_leak_check):
    """Get the Iceberg read schema from a Snowflake-managed table"""
    import bodo.io.iceberg

    conn = "iceberg+" + get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", {"role": "ACCOUNTADMIN", "warehouse": "DEMO_WH"}
    )

    col_names, bodo_col_types, pyarrow_schema = bodo.io.iceberg.get_iceberg_orig_schema(
        conn, "TEST_DB.PUBLIC.TEST_ICEBERG_TABLE"
    )

    assert col_names == ["VAL", "B"]
    assert bodo_col_types == [
        bodo.types.boolean_array_type,
        bodo.types.IntegerArrayType(bodo.types.int32),
    ]
    assert pyarrow_schema == pyarrow.schema(
        [
            pyarrow.field("VAL", pyarrow.bool_(), nullable=False),
            pyarrow.field("B", pyarrow.int32()),
        ]
    )


def test_basic_read(memory_leak_check):
    """
    Test reading a complete Iceberg table from Snowflake
    """

    def impl(table_name, conn, db_schema):
        return pd.read_sql_table(table_name, conn, db_schema)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "bob", "cassie", "david", pd.NA],
            "B": [10.5, -124.0, 11.11, 456.2, -8e2],
            "C": [True, pd.NA, False, pd.NA, pd.NA],
        }
    )

    db_schema = "TEST_DB.PUBLIC"
    conn = "iceberg+" + get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", {"role": "ACCOUNTADMIN"}
    )
    check_func(
        impl,
        ("BODOSQL_ICEBERG_READ_TEST", conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def test_read_implicit_pruning(memory_leak_check):
    """
    Test reading an Iceberg table from Snowflake with Bodo
    compiler column pruning
    """

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df["B"] = df["B"].abs()
        return df[["B", "A"]]

    py_out = pd.DataFrame(
        {
            "B": [10.5, 124.0, 11.11, 456.2, 8e2],
            "A": ["ally", "bob", "cassie", "david", pd.NA],
        }
    )

    db_schema = "TEST_DB.PUBLIC"
    conn = "iceberg+" + get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", {"role": "ACCOUNTADMIN"}
    )
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            ("BODOSQL_ICEBERG_READ_TEST", conn, db_schema),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")
