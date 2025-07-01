import random
import string
from io import StringIO

import boto3
import pandas as pd

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    pytest_s3_tables,
    run_rank0,
    temp_env_override,
)

pytestmark = pytest_s3_tables

# This bucket must exist and have the read_namespace and write_namespace namespaces
# created. Additionally, the bodo_iceberg_read_test table should have contents
# matching the test_basic_read test's py_out.
bucket_arn = "arn:aws:s3tables:us-east-2:427443013497:bucket/unittest-bucket"


@temp_env_override({"AWS_DEFAULT_REGION": "us-east-2"})
def test_basic_read(memory_leak_check):
    """
    Test reading a complete Iceberg table S3 Tables
    """

    def impl(table_name, conn, db_schema):
        return pd.read_sql_table(table_name, conn, db_schema)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "bob", "cassie", "david", None],
            "B": [10.5, -124.0, 11.11, 456.2, -8e2],
            "C": [True, None, False, None, None],
        }
    )

    conn = "iceberg+" + bucket_arn
    check_func(
        impl,
        ("bodo_iceberg_read_test", conn, "read_namespace"),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@temp_env_override({"AWS_DEFAULT_REGION": "us-east-2"})
def test_read_implicit_pruning(memory_leak_check):
    """
    Test reading an Iceberg table from S3 Tables with Bodo
    compiler column pruning
    """

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df["B"] = df["B"].abs()
        return df[["B", "A"]]

    py_out = pd.DataFrame(
        {
            "B": [10.5, 124.0, 11.11, 456.2, 8e2],
            "A": ["ally", "bob", "cassie", "david", None],
        }
    )

    conn = "iceberg+" + bucket_arn
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            ("bodo_iceberg_read_test", conn, "read_namespace"),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")


@temp_env_override({"AWS_DEFAULT_REGION": "us-east-2"})
def test_basic_write(memory_leak_check):
    """
    Test writing a complete Iceberg table to S3 Tables
    """

    @bodo.jit(distributed=["df"])
    def write(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema)

    def read(table_name, conn, db_schema):
        return pd.read_sql_table(table_name, conn, db_schema)

    df = pd.DataFrame(
        {
            "A": ["ally", "bob", "cassie", "david", None] * 5,
            "B": [10.5, -124.0, 11.11, 456.2, -8e2] * 5,
            "C": [True, None, False, None, None] * 5,
        }
    )
    conn = "iceberg+" + bucket_arn
    r = random.Random()
    table_name = run_rank0(
        lambda: (
            f"bodo_iceberg_write_test_{''.join(r.choices(string.ascii_lowercase, k=4))}"
        )
    )()

    try:
        write(_get_dist_arg(df), table_name, conn, "write_namespace")

        check_func(
            read,
            (table_name, conn, "write_namespace"),
            py_output=df,
            sort_output=True,
            reset_index=True,
        )
    finally:

        def cleanup():
            client = boto3.client("s3tables")
            client.delete_table(
                name=table_name,
                namespace="write_namespace",
                tableBucketARN=bucket_arn,
            )
            client.close()

        run_rank0(cleanup)()
