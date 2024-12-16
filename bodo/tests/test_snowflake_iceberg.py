import os
from io import StringIO

import pandas as pd
import pyarrow
import pytest
from bodo_iceberg_connector.schema import BodoIcebergSchema, get_iceberg_info

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    get_snowflake_connection_string,
    pytest_snowflake,
    snowflake_cred_env_vars_present,
    temp_env_override,
)

pytestmark = [pytest.mark.iceberg] + pytest_snowflake


@temp_env_override({"AWS_REGION": "us-east-1"})
@pytest.mark.skipif(
    not snowflake_cred_env_vars_present(),
    reason="Snowflake environment variables not set",
)
@pytest.mark.skipif(
    bodo.get_size() != 1, reason="get_iceberg_info should only be called on 1 rank"
)
def test_get_iceberg_info_snowflake(memory_leak_check):
    SF_USERNAME = os.environ["SF_USERNAME"]
    SF_PASSWORD = os.environ["SF_PASSWORD"]
    info = get_iceberg_info(
        f"snowflake://bodopartner.us-east-1.snowflakecomputing.com/?warehouse=DEMO_WH&user={SF_USERNAME}&password={SF_PASSWORD}",
        "TEST_DB.PUBLIC",
        "TEST_ICEBERG_TABLE",
    )
    assert info[0] == 0
    assert info[1] == "s3://bodo-snowflake-iceberg-test/test_iceberg_table/data"
    assert info[2] == BodoIcebergSchema(
        ["VAL", "B"],
        [
            pyarrow.field("VAL", pyarrow.bool_(), nullable=False),
            pyarrow.field("B", pyarrow.int32()),
        ],
        [2, 3],
        [True, False],
    )
    assert info[3] == pyarrow.schema(
        [
            pyarrow.field("VAL", pyarrow.bool_(), nullable=False),
            pyarrow.field("B", pyarrow.int32()),
        ]
    )
    assert (
        info[4]
        == '{"type":"struct","schema-id":0,"fields":[{"id":2,"name":"VAL","required":true,"type":"boolean"},{"id":3,"name":"B","required":false,"type":"int"}]}'
    )
    assert info[5] == []
    assert info[6] == []


@temp_env_override({"AWS_REGION": "us-east-1"})
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


@temp_env_override({"AWS_REGION": "us-east-1"})
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
