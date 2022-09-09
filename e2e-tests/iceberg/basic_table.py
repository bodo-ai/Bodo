import argparse
import os
import sys
from datetime import datetime
from typing import Callable, Tuple

import bodo_iceberg_connector  # noqa
import boto3
import numba
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
from mpi4py import MPI
from pyspark.sql import SparkSession

import bodo
from bodo.tests.utils import _get_dist_arg

BUCKET_NAME = "engine-e2e-tests-iceberg"
# The AWS Java SDK 2.0 library (used by Iceberg-AWS) gets the default region
# from the AWS_REGION environment variable instead of the AWS_DEFAULT_REGION,
# which is used by other libraries.
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_REGION"] = "us-east-1"
comm = MPI.COMM_WORLD


def get_spark_iceberg(nessie_token):
    spark = (
        SparkSession.builder.appName("Iceberg with Spark")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.2_2.12:0.13.1,software.amazon.awssdk:bundle:2.15.40,software.amazon.awssdk:url-connection-client:2.15.40,org.apache.hadoop:hadoop-aws:3.2.0,org.projectnessie:nessie-spark-3.2-extensions:0.30.0",
        )
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.nessie.catalog-impl",
            "org.apache.iceberg.nessie.NessieCatalog",
        )
        .config(
            "spark.sql.catalog.nessie.uri",
            "https://nessie.dremio.cloud/v1/projects/50824e14-fd95-434c-a0cb-cc988e57969f",
        )
        .config(
            "spark.sql.catalog.nessie.warehouse", "s3a://bodo-iceberg-test2/arctic_test"
        )
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie.authentication.type", "BEARER")
        .config(
            "spark.sql.catalog.nessie.authentication.token",
            nessie_token,
        )
        .config("spark.sql.catalog.nessie.cache-enabled", "false")
        .config(
            "spark.sql.catalog.nessie.io-impl", "org.apache.iceberg.aws.s3.S3FileIO"
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,org.projectnessie.spark.extensions.NessieSpark32SessionExtensions",
        )
        .getOrCreate()
    )
    return spark


# ------------------------ Cleanup Code ------------------------
def cleanup_nessie(nessie_token: str, table_name: str):
    spark = get_spark_iceberg(nessie_token)
    spark.sql(f"DROP TABLE nessie.e2e.{table_name}")
    fs = pafs.S3FileSystem(region="us-east-1")
    fs.delete_dir_contents(f"bodo-iceberg-test2/arctic_test/e2e")


def cleanup_glue(table_name: str):
    client = boto3.client("glue")
    client.batch_delete_table(DatabaseName="glue", TablesToDelete=[table_name])
    fs = pafs.S3FileSystem(region="us-east-1")
    fs.delete_dir_contents(f"{BUCKET_NAME}/glue.db")


def cleanup_hadoop():
    fs = pafs.S3FileSystem(region="us-east-1")
    fs.delete_dir_contents(f"{BUCKET_NAME}/hadoop")


CLEANUP = {
    "Nessie": cleanup_nessie,
    "Glue": cleanup_glue,
    "S3": cleanup_hadoop,
}


# ------------------------ Test Code ------------------------
def test_builder(
    conn: str, db_name: str, table_name: str
) -> Tuple[Callable[[pd.DataFrame], None], Callable[[], pd.DataFrame]]:
    @bodo.jit(distributed=["df"], cache=True)
    def write_impl(df):
        df.to_sql(table_name, conn, schema=db_name, if_exists="fail")

    @bodo.jit(cache=True)
    def read_back_impl():
        out_df = pd.read_sql_table(table_name, conn, schema=db_name)
        return out_df

    return write_impl, read_back_impl


nessie_tests = lambda nessie_token, table_name: test_builder(
    "iceberg+https://nessie.dremio.cloud/v1/projects/"
    "50824e14-fd95-434c-a0cb-cc988e57969f"
    "?type=nessie&authentication.type=BEARER"
    f"&authentication.token={nessie_token}"
    f"&warehouse=s3://bodo-iceberg-test2/arctic_test",
    "e2e",
    table_name,
)

glue_tests = lambda table_name: test_builder(
    f"iceberg+glue?warehouse=s3://{BUCKET_NAME}", "glue", table_name
)

s3_tests = lambda table_name: test_builder(
    f"iceberg+s3://{BUCKET_NAME}", "hadoop", table_name
)


# ------------------------ Test Case ------------------------
df = pd.DataFrame(
    {
        "bools": np.array([True, False, True, True, False] * 10, dtype=np.bool_),
        "bytes": np.array([1, 1, 0, 1, 0] * 10).tobytes(),
        "ints": np.array([1, 2, 3, 4, 5] * 10, dtype=np.int32),
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

    table_name = args.table_name

    # Get Nessie Authentication Token
    nessie_token = os.environ.get("NESSIE_AUTH_TOKEN")
    assert nessie_token is not None, "Missing Environment Variable: NESSIE_AUTH_TOKEN"

    # Build Test Cases
    tests = {
        "Nessie": nessie_tests(nessie_token, table_name),
        "Glue": glue_tests(table_name),
        "S3": s3_tests(table_name),
    }

    passed = 0
    for key, (write_test_func, read_test_func) in tests.items():
        try:
            write_test_func(_get_dist_arg(df))
            out_df = read_test_func()
            out_df = bodo.allgatherv(out_df)
            pd.testing.assert_frame_equal(out_df, df, check_dtype=False)

            if args.require_cache:
                if isinstance(write_test_func, numba.core.dispatcher.Dispatcher):
                    assert (
                        write_test_func._cache_hits[write_test_func.signatures[0]] == 1
                    ), f"ERROR: Bodo did not load write function from cache"
                if isinstance(read_test_func, numba.core.dispatcher.Dispatcher):
                    assert (
                        read_test_func._cache_hits[read_test_func.signatures[0]] == 1
                    ), f"ERROR: Bodo did not load read function from cache"

            passed += 1
            if bodo.get_rank() == 0:
                print(f"Finished {key} Test successfully...")
        except Exception as e:
            if bodo.get_rank() == 0:
                print(f"Error During {key} Test\n{e}")

    # Cleanup of Test Cases
    bodo.barrier()
    cleanup_failed = False
    if bodo.get_rank() == 0:
        try:
            print("Starting cleanup...")
            print("Cleaning up hadoop...")
            cleanup_hadoop()
            print("Cleaning up glue...")
            cleanup_glue(table_name)
            print("Cleaning up nessie...")
            cleanup_nessie(nessie_token, table_name)
            print("Successfully finished cleanup...")
        except Exception as e:
            cleanup_failed = True
            print(f"Failed during cleanup ...\n{e}", file=sys.stderr)

    cleanup_failed = comm.bcast(cleanup_failed)
    if cleanup_failed:
        raise Exception("Iceberg E2E Cleanup Failed. See Rank 0.")

    assert passed == len(tests), f"{len(tests) - passed} tests failed."
