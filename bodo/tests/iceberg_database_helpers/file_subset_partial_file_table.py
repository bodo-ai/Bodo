import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="file_subset_partial_file_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(["A", "B", "C", "D"] * 500000),
            "B": np.arange(0, 2000000, dtype=np.int64),
        }
    )
    sql_schema = [
        ("A", "string", True),
        ("B", "long", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)

    # Delete some rows
    print("Deleting some rows...")
    spark.sql(
        f""" 
        DELETE FROM hadoop_prod.{DATABASE_NAME}.{table_name}
        WHERE B = 16;
    """
    )

    ## Seems like the way at least Spark does it is that it creates new files for the
    ## rows don't get filtered out, rather than storing any information about
    ## rows to skip.


if __name__ == "__main__":
    create_table()
