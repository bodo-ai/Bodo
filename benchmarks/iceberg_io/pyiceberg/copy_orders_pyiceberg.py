import time

import pyarrow as pa
from pyiceberg.catalog import load_catalog

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
orders_table = rest_catalog.load_table("sf1000.orders")
if rest_catalog.table_exists("sf1000.orders_copy_pyiceberg"):
    rest_catalog.purge_table("sf1000.orders_copy_pyiceberg")

start = time.time()
orders_copy = rest_catalog.create_table(
    "sf1000.orders_copy_pyiceberg", orders_table.schema()
)
try:
    tx = orders_copy.transaction()
    reader = orders_table.scan().to_arrow_batch_reader()
    for batch in reader:
        tx.append(pa.Table.from_batches([batch]))
    tx.commit_transaction()
    end = time.time()
    print("Time taken to copy orders: ", end - start)
finally:
    rest_catalog.purge_table("sf1000.orders_copy_pyiceberg")
