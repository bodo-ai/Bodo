import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="file_subset_deleted_rows_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(["A", "B", "C", "D"] * 5),
            "B": np.array(["lorem", "ipsum"] * 10),
            "C": np.array((["A"] * 2) + (["b"] * 18)),
            "D": np.array([1, 2] * 10, np.int32),
            "E": np.array([1, 2] * 10, np.float32),
            "K": np.array([54] * 20, np.int64),
        }
    )
    sql_schema = [
        ("A", "string", True),
        ("B", "string", True),
        ("C", "string", True),
        ("D", "int", True),
        ("E", "float", True),
        ("K", "long", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('QWERTY', 'dolor', 'C', 5, 5.34, 32),
        ('ASDFGH', 'sit', 'D', 56, 9.87, 12);
    """
    )

    # Delete all rows except those from last insert
    print("Deleting rows...")
    spark.sql(
        f"""
        DELETE FROM hadoop_prod.{DATABASE_NAME}.{table_name}
        WHERE K = 54;
    """
    )


if __name__ == "__main__":
    create_table()
