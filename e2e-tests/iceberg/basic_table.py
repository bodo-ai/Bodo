import argparse
import functools
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime

import boto3
import numba
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import snowflake
from pyspark.sql import SparkSession

import bodo
from bodo.mpi4py import MPI
from bodo.tests.utils import _get_dist_arg

BUCKET_NAME = "engine-e2e-tests-iceberg"
# The AWS Java SDK 2.0 library (used by Iceberg-AWS) gets the default region
# from the AWS_REGION environment variable instead of the AWS_DEFAULT_REGION,
# which is used by other libraries.
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_REGION"] = "us-east-1"
comm = MPI.COMM_WORLD

logger = logging.getLogger(__name__)


def get_spark_iceberg(nessie_token):
    spark = (
        SparkSession.builder.appName("Iceberg with Spark")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.5.2,"
            "software.amazon.awssdk:bundle:2.19.13,"
            "software.amazon.awssdk:url-connection-client:2.19.13,"
            "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.4_2.12:0.71.1",
        )
        .config(
            "spark.sql.catalog.nessie_catalog", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config(
            "spark.sql.catalog.nessie_catalog.catalog-impl",
            "org.apache.iceberg.nessie.NessieCatalog",
        )
        .config(
            "spark.sql.catalog.nessie_catalog.uri",
            "https://nessie.dremio.cloud/v1/repositories/e83a2cde-4a47-4522-82c0-4aa5f358e1bf",
        )
        .config(
            "spark.sql.catalog.nessie_catalog.warehouse", f"s3://{BUCKET_NAME}/nessie"
        )
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie_catalog.authentication.type", "BEARER")
        .config("spark.sql.catalog.nessie_catalog.authentication.token", nessie_token)
        .config("spark.sql.catalog.nessie_catalog.cache-enabled", "false")
        .config(
            "spark.sql.catalog.nessie_catalog.io-impl",
            "org.apache.iceberg.aws.s3.S3FileIO",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
            "org.projectnessie.spark.extensions.NessieSparkSessionExtensions",
        )
        .getOrCreate()
    )
    return spark


# ------------------------ Cleanup Code ------------------------
def cleanup_nessie(nessie_token: str, table_name: str):
    spark = get_spark_iceberg(nessie_token)
    spark.sql(f"DROP TABLE nessie_catalog.{table_name} PURGE")


def cleanup_glue(table_name: str):
    client = boto3.client("glue")
    client.batch_delete_table(DatabaseName="glue", TablesToDelete=[table_name])
    fs = pafs.S3FileSystem(region="us-east-1")
    fs.delete_dir_contents(f"{BUCKET_NAME}/glue.db")


def cleanup_hadoop():
    fs = pafs.S3FileSystem(region="us-east-1")
    fs.delete_dir_contents(f"{BUCKET_NAME}/hadoop")


def cleanup_snowflake(table_name: str):
    conn = snowflake.connector.connect(
        user=os.environ["SF_USERNAME"],
        password=os.environ["SF_PASSWORD"],
        account=os.environ["SF_ACCOUNT"],
        warehouse="DEMO_WH",
        database="TEST_DB",
        schema="PUBLIC",
    )
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")


# ------------------------ Snowflake Write Impl ----------------
def snowflake_write_impl(df, table_name):
    SF_USERNAME = os.environ["SF_USERNAME"]
    SF_PASSWORD = os.environ["SF_PASSWORD"]
    SF_ACCOUNT = os.environ["SF_ACCOUNT"]
    bodo.jit(
        lambda df, SF_USERNAME, SF_PASSWORD, SF_ACCOUNT: df.to_sql(
            f"{table_name}_non_iceberg",
            f"snowflake://{SF_USERNAME}:{SF_PASSWORD}@{SF_ACCOUNT}/TEST_DB/PUBLIC?warehouse=DEMO_WH",
        ),
        distributed=["df"],
        cache=True,
    )(df, SF_USERNAME, SF_PASSWORD, SF_ACCOUNT)
    if bodo.get_rank() == 0:
        conn = snowflake.connector.connect(
            user=SF_USERNAME,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,
            warehouse="DEMO_WH",
            database="TEST_DB",
            schema="PUBLIC",
        )
        cur = conn.cursor()
        cur.execute(
            f'CREATE OR REPLACE ICEBERG TABLE {table_name} ("bools" boolean, "bytes" BINARY, "ints" bigint, "floats" float, "strings" string, "lists" array(number(2, 0)), "timestamps" timestamp_ntz) CATALOG=\'SNOWFLAKE\' EXTERNAL_VOLUME=EXVOL BASE_LOCATION={table_name} AS SELECT bools, bytes, ints, floats, strings, lists::ARRAY(NUMBER(2,0)) as lists, timestamps FROM {table_name}_non_iceberg'
        ).fetchall()
        cur.execute(f"DROP TABLE {table_name}_non_iceberg").fetchall()
        cur.close()
    bodo.barrier()


# ------------------------ Test Code ------------------------
def test_builder(
    conn: str,
    db_name: str,
    table_name: str,
    write_impl: Callable[[pd.DataFrame], None] | None = None,
    read_back_impl: Callable[[], pd.DataFrame] | None = None,
) -> tuple[Callable[[pd.DataFrame], None], Callable[[], pd.DataFrame]]:
    if write_impl is None:

        def write_impl_def(df):
            print("starting write...")
            df.to_sql(table_name, conn, schema=db_name, if_exists="fail")

        write_impl = bodo.jit(write_impl_def, distributed=["df"], cache=True)

    if read_back_impl is None:

        def read_back_impl_def():
            print("starting read...")
            out_df = pd.read_sql_table(table_name, conn, schema=db_name)
            return out_df

        read_back_impl = bodo.jit(read_back_impl_def, cache=True)

    return write_impl, read_back_impl


nessie_tests = lambda nessie_token, table_name: test_builder(
    "iceberg+https://nessie.dremio.cloud/v1/repositories/"
    "e83a2cde-4a47-4522-82c0-4aa5f358e1bf"
    "?type=nessie&authentication.type=BEARER"
    f"&authentication.token={nessie_token}"
    f"&warehouse=s3://{BUCKET_NAME}/nessie",
    "",
    table_name,
)

glue_tests = lambda table_name: test_builder(
    f"iceberg+glue?warehouse=s3://{BUCKET_NAME}", "glue", table_name
)

s3_tests = lambda table_name: test_builder(
    f"iceberg+s3://{BUCKET_NAME}", "hadoop", table_name
)

snowflake_tests = lambda table_name: test_builder(
    f"iceberg+snowflake://{os.environ['SF_ACCOUNT']}/?warehouse=DEMO_WH&user={os.environ['SF_USERNAME']}&password={os.environ['SF_PASSWORD']}",
    "TEST_DB.PUBLIC",
    table_name,
    write_impl=functools.partial(snowflake_write_impl, table_name=table_name),
)

# ------------------------ Test Case ------------------------
df = pd.DataFrame(
    {
        "bools": np.array([True, False, True, True, False] * 10, dtype=np.bool_),
        "bytes": np.array([1, 1, 0, 1, 0] * 10).tobytes(),
        "ints": np.arange(50, dtype=np.int32),
        "floats": np.array([1, 2, 3, 4, 5] * 10, dtype=np.float32),
        "strings": np.array(["A", "B", "C", "D", "E"] * 10),
        "lists": pd.Series(
            [[0, 1, 2], [3, 4], [5], [6, 7], [8, 9, 10]] * 10, dtype=object
        ),
        "timestamps": pd.Series(
            [
                datetime.strptime("12/11/2018", "%d/%m/%Y"),
                datetime.strptime("12/11/2019", "%d/%m/%Y"),
                datetime.strptime("12/12/2018", "%d/%m/%Y"),
                datetime.strptime("13/11/2018", "%d/%m/%Y"),
                datetime.strptime("14/10/2019", "%d/%m/%Y"),
            ]
            * 10
        ),
    }
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_name", type=str)
    parser.add_argument("--require_cache", action="store_true", default=False)
    args = parser.parse_args()
    print("starting test...")

    table_name = args.table_name

    # Get Nessie Authentication Token
    # TODO[BSE-1408]: enable Nessie tests after fixing issues
    # nessie_token = os.environ.get("NESSIE_AUTH_TOKEN")
    # assert nessie_token is not None, "Missing Environment Variable: NESSIE_AUTH_TOKEN"

    # Build Test Cases
    tests = {
        # TODO[BSE-1408]: enable Nessie tests after fixing issues
        # "Nessie": nessie_tests(nessie_token, table_name),
        "Glue": glue_tests(table_name),
        "S3": s3_tests(table_name),
        "Snowflake": snowflake_tests(table_name),
    }

    passed = 0
    for key, (write_test_func, read_test_func) in tests.items():
        print(f"Running {key}")
        try:
            write_test_func(_get_dist_arg(df))
            out_df = read_test_func()
            out_df = bodo.allgatherv(out_df)
            out_df_to_cmp = out_df.sort_values("ints", ignore_index=True)
            df_to_cmp = df.sort_values("ints", ignore_index=True)

            pd.testing.assert_frame_equal(out_df_to_cmp, df_to_cmp, check_dtype=False)
            if args.require_cache:
                if isinstance(write_test_func, numba.core.dispatcher.Dispatcher):
                    assert (
                        write_test_func._cache_hits[write_test_func.signatures[0]] == 1
                    ), "ERROR: Bodo did not load write function from cache"
                if isinstance(read_test_func, numba.core.dispatcher.Dispatcher):
                    assert (
                        read_test_func._cache_hits[read_test_func.signatures[0]] == 1
                    ), "ERROR: Bodo did not load read function from cache"

            passed += 1
            print(f"Finished {key} Test successfully...")
        except Exception as e:
            print(f"Error During {key} Test")
            logging.exception(e)
    # Cleanup of Test Cases
    bodo.barrier()
    cleanup_failed = False
    if bodo.get_rank() == 0:
        try:
            print("Starting Cleanup...")
            print("Cleaning up Hadoop...")
            cleanup_hadoop()
            print("Cleaning up Glue...")
            cleanup_glue(table_name)
            print("Cleaning up Snowflake...")
            cleanup_snowflake(table_name)
            # TODO[BSE-1408]: enable Nessie tests after fixing issues
            # print("Cleaning up Nessie...")
            # cleanup_nessie(nessie_token, table_name)
        except Exception as e:
            cleanup_failed = True
            print(f"Failed During Cleanup...\n{e}", file=sys.stderr)
        else:
            print("Successfully Finished Cleanup...")

    cleanup_failed = comm.bcast(cleanup_failed)
    if cleanup_failed:
        raise Exception("Iceberg E2E Cleanup Failed. See Rank 0.")

    assert passed == len(tests), f"{len(tests) - passed} tests failed."
