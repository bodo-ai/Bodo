"""
Benchmark Daft on a Ray cluster, writing output to an S3 bucket.
"""

import os
import time

import boto3
import daft
import ray
from pyiceberg.catalog import load_catalog

session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()

# Set credential-related environment variables
# If we don't do this all workers will try to connect to IMDS
# and potentially get throttled.
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
os.environ["AWS_SESSION_TOKEN"] = credentials.token
os.environ["AWS_REGION"] = "us-east-2"


def read_write_iceberg():
    s3tables_arn = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
    tpch_dataset = "sf1000.orders"
    rest_catalog = load_catalog(
        "tpch",
        **{
            "type": "rest",
            "warehouse": s3tables_arn,
            "uri": "https://s3tables.us-east-2.amazonaws.com/iceberg",
            "rest.sigv4-enabled": "true",
            "rest.signing-name": "s3tables",
            "rest.signing-region": "us-east-2",
        },
    )
    if rest_catalog.table_exists(f"{tpch_dataset}_copy_daft"):
        rest_catalog.purge_table(f"{tpch_dataset}_copy_daft")

    orders_table = rest_catalog.load_table(tpch_dataset)

    start = time.time()
    try:
        orders_copy = rest_catalog.create_table(
            f"{tpch_dataset}_copy_daft", orders_table.schema()
        )
        dataset = daft.read_iceberg(orders_table)
        # Limit the dataset to <10000000 rows for testing
        # dataset = dataset.limit(10000)
        # TODO: Resolve OOM issues with larger datasets
        dataset.write_iceberg(orders_copy, mode="overwrite")
        end = time.time()
        print("Total read-write time:", (end - start))
    finally:
        rest_catalog.purge_table(f"{tpch_dataset}_copy_daft")


def main():
    # start ray cluster, configure Daft
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                "AWS_SESSION_TOKEN": os.environ["AWS_SESSION_TOKEN"],
                "AWS_REGION": os.environ["AWS_REGION"],
            },
        },
    )
    daft.context.set_runner_ray()

    read_write_iceberg()


if __name__ == "__main__":
    main()
