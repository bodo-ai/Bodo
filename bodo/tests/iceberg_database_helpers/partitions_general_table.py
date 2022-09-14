import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="partitions_general_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array([0.0, 1.4, 5.6, 7.3] * 5, np.float64),
            "B": np.arange(0, 20, dtype=np.int64),
            "C": pd.Series(["ABC", "DEF"] * 10),
        }
    )
    sql_schema = [
        ("A", "double", True),
        ("B", "long", True),
        ("C", "string", False),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)

    # Add partition field
    print("Adding partition field (year)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD A
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (56.98, 34, 'ERT'),
        (432.986, 23, 'TRY');
    """
    )

    # Add partition field
    print("Adding partition field (B)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD B
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (872.98, 45, 'WER'),
        (2180.45, 23, 'FGH');
    """
    )

    # Add partition field (day)
    print("Adding partition field (day)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD C
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (87.76, 87, 'POR'),
        (9.345, 65, 'NJU');
    """
    )

    # Remove partition field (month)
    print("Removing partition field (month)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP PARTITION FIELD B
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (64.267, 67, 'LKJ'),
        (98.652, 32, 'YTR');
    """
    )

    # Remove a partition column (B)
    print("Removing partition field B...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP PARTITION FIELD A
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (87.09, 987, 'KMO'),
        (8003.56, 382, 'VGY');
    """
    )


if __name__ == "__main__":
    create_table()
