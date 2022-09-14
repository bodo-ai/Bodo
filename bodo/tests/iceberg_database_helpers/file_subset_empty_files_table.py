import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="file_subset_empty_files_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(["A", "B", "C", "D"] * 5),
            "B": np.array([1, 2] * 10, np.float32),
        }
    )
    sql_schema = [
        ("A", "string", True),
        ("B", "float", True),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)

    # Add more columns
    print("Adding columns...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD COLUMN E varchar(60000),
        C varchar(255),
        D int;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('QWERTY', 5.34, 'dolor', 'C', 5),
        ('ASDFGH', 9.87, 'sit', 'D', 56);
    """
    )

    # Remove original columns
    print("Removing original columns...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP COLUMN A, B;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('lorem', 'ER', 98),
        ('ipsum', 'RT', 234);
    """
    )


if __name__ == "__main__":
    create_table()
