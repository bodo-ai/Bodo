"""
Benchmark Daft on a Ray cluster, writing output to an S3 bucket.
"""

import time

import daft
import ray
from pyiceberg.catalog import load_catalog


def read_write_iceberg():
    s3tables_arn = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
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
    if rest_catalog.table_exists("sf1000.orders_copy_daft"):
        rest_catalog.purge_table("sf1000.orders_copy_daft")

    orders_table = rest_catalog.load_table("sf1000.orders")

    start = time.time()
    try:
        orders_copy = rest_catalog.create_table(
            "sf1000.orders_copy_daft", orders_table.schema()
        )
        dataset = daft.read_iceberg(orders_table)
        dataset.write_iceberg(orders_copy, mode="overwrite")

        end = time.time()
        print("Total read-write time:", (end - start))
    finally:
        rest_catalog.purge_table("sf1000.orders_copy_daft")


def main():
    # start ray cluster, configure Daft
    ray.init(address="auto")
    daft.context.set_runner_ray()

    read_write_iceberg()


if __name__ == "__main__":
    main()
