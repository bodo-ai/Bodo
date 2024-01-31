import os

import pyarrow
import pytest
from bodo_iceberg_connector.schema import BodoIcebergSchema, get_iceberg_info

import bodo
from bodo.tests.utils import (
    pytest_snowflake,
    snowflake_cred_env_vars_present,
    temp_env_override,
)

pytestmark = pytest_snowflake


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
