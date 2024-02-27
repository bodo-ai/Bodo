# Taken from: https://github.com/Bodo-inc/Bodo/blob/master/e2e-tests/snowflake/snowflake-read/snowflake_read.py
import os
import bodo
import time
import bodo
import pandas as pd
from bodo_platform_utils import catalog
import sys

conn_secrets = catalog.get_data("integration-test")
username = conn_secrets["username"]
password = conn_secrets["password"]
account = conn_secrets["host"]
warehouse = "DEMO_WH"

schema_read = "TPCH_SF1"
database_read = "SNOWFLAKE_SAMPLE_DATA"

schema_write = "PUBLIC"
database_write = "E2E_TESTS_DB"


@bodo.jit(cache=True)
def main(sf_read_conn, sf_write_conn, table_name):
    """
    Snowflake E2E read test. We read in three string columns from TPCH-SF1 on Snowflake,
    perform some groupby and string functions to the dataframe, and write the transformed
    data frame back to snowflake.
    Args:
        sf_conn (str): snowflake connection string
    """
    query = "SELECT l_shipmode, l_shipinstruct, l_comment FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY"
    # Load in the data
    t0 = time.time()
    print("Starting read...")
    df = pd.read_sql(
        query, sf_read_conn, _bodo_read_as_dict=["l_shipmode", "l_shipinstruct"]
    )
    bodo.barrier()
    print("Length of input dataframe: ", len(df))
    print("Read time: ", time.time() - t0)

    # Perform simple series.str operation on string columns
    t1 = time.time()
    df["a_count"] = df["l_shipinstruct"].str.count("A")
    df["paded_mode"] = df["l_shipmode"].str.ljust(10)
    bodo.barrier()
    print("string operations(count & ljust) time: ", time.time() - t1)

    # Perform simple transformations
    # the transformations bear little meaning, and
    # they serve only as a means to demonstrate our ability
    # to carry out operations on dictionary-encoded arrays.
    t2 = time.time()
    print("Starting groupby transformation...")
    gb = df.groupby(["l_shipmode"])
    bodo.barrier()
    print("Groupby time: ", time.time() - t2)
    print("Number of groups: ", gb.size())

    t3 = time.time()
    agg_strs = gb["l_shipinstruct"].agg(["min", "nunique"])
    bodo.barrier()
    print("Agg time: ", time.time() - t3)
    print("Number of unique l_shipinstruct for each group, sorted from high to low:")
    print("nunique:\n", agg_strs["nunique"])

    # The summation is for testing correctness only and is
    # not related to dictionary encoding
    agg_counts = gb["a_count"].sum()

    t4 = time.time()
    print("Starting Snowflake write...")
    df.to_sql(table_name, sf_write_conn, if_exists="replace", index=False)
    bodo.barrier()
    print("Snowflake Write time: ", time.time() - t4)

    return agg_counts


if __name__ == "__main__":
    read_conn = f"snowflake://{username}:{password}@{account}/{database_read}/{schema_read}?warehouse={warehouse}"
    write_conn = f"snowflake://{username}:{password}@{account}/{database_write}/{schema_write}?warehouse={warehouse}"
    main(read_conn, write_conn, "integration_test_write")
