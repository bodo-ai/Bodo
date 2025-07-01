import time

from pyiceberg.catalog import load_catalog

import bodo.pandas as pd

s3tables_arn = "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
catalog_properties = {
    "type": "rest",
    "warehouse": s3tables_arn,
    "uri": "https://s3tables.us-east-2.amazonaws.com/iceberg",
    "rest.sigv4-enabled": "true",
    "rest.signing-name": "s3tables",
    "rest.signing-region": "us-east-2",
}

rest_catalog = load_catalog(
    "tpch",
    **catalog_properties,
)
if rest_catalog.table_exists("sf1000.orders_copy_bodo"):
    rest_catalog.purge_table("sf1000.orders_copy_bodo")

start = time.time()
try:
    orders_table = pd.read_iceberg(
        "sf1000.orders",
        location=s3tables_arn,
    )
    orders_table.to_iceberg(
        "sf1000.orders_copy_bodo",
        location=s3tables_arn,
    )
    end = time.time()
    print("Time taken to copy orders: ", end - start)
finally:
    rest_catalog.purge_table("sf1000.orders_copy_pbodo")
