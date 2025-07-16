import os
import time

import boto3
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

# Get credentials from IMDS
session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()

# Set credential-related environment variables
# If we don't do this all workers will try to connect to IMDS
# and potentially get throttled. The downside is they expire after six hours
# and won't be refetched automatically
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
os.environ["AWS_SESSION_TOKEN"] = credentials.token

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
    rest_catalog.purge_table("sf1000.orders_copy_bodo")
