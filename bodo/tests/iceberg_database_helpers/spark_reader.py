import sys

import numpy as np

from bodo.tests.iceberg_database_helpers.utils import get_spark


def read_iceberg_table(table_name, database_name, spark=None):
    if not spark:
        spark = get_spark()

    # This is so when reading timestamp columns, the output will
    # match that of Bodo.
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = spark.table(f"hadoop_prod.{database_name}.{table_name}")
    count = df.count()
    spark_schema = df.schema
    pd_df = df.toPandas()

    # Convert datetime64 to tz-aware UTC to match Bodo output
    for i in range(len(pd_df.columns)):
        if pd_df.dtypes.iloc[i] == np.dtype("datetime64[ns]"):
            pd_df[pd_df.columns[i]] = pd_df[pd_df.columns[i]].dt.tz_localize("UTC")

    return pd_df, count, spark_schema


def read_iceberg_table_single_rank(table_name, database_name, spark=None):
    from mpi4py import MPI

    import bodo

    if bodo.get_rank() == 0:
        py_out, _, _ = read_iceberg_table(table_name, database_name)
    else:
        py_out = None

    comm = MPI.COMM_WORLD
    py_out = comm.bcast(py_out, root=0)
    return py_out


if __name__ == "__main__":
    table_name = sys.argv[1]
    database_name = sys.argv[2]
    pd_df, count, spark_schema = read_iceberg_table(table_name, database_name)
    print(f"Count: {count}")
    print(f"Schema:\n{spark_schema}")
    print(f"Dataframe:\n{pd_df}")
