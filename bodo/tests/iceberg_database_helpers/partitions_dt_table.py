from datetime import datetime

import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="partitions_dt_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(
                [
                    datetime.strptime("12/11/2018", "%d/%m/%Y").date(),
                    datetime.strptime("12/11/2019", "%d/%m/%Y").date(),
                    datetime.strptime("12/12/2018", "%d/%m/%Y").date(),
                    datetime.strptime("13/11/2018", "%d/%m/%Y").date(),
                ]
                * 5
            ),
            "B": np.arange(0, 20, dtype=np.int64),
            "C": pd.Series(["ABC", "DEF"] * 10),
        }
    )
    sql_schema = [
        ("A", "date", True),
        ("B", "long", True),
        ("C", "string", False),
    ]
    create_iceberg_table(df, sql_schema, table_name, spark)

    # Add partition field
    print("Adding partition field (year)...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD years(A)
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2017-10-15', 34, 'ERT'),
        (DATE'2017-03-29', 23, 'TRY');
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
        (DATE'2021-06-19', 45, 'WER'),
        (DATE'2012-06-14', 23, 'FGH');
    """
    )

    # Modify partition field (year to month)
    print("Modifying partition field (year --> month)...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD months(A)
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2021-06-22', 45, 'WER'),
        (DATE'2014-06-15', 23, 'FGH');
    """
    )

    # Add partition field (day)
    print("Adding partition field (day)...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD days(A)
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2021-06-30', 87, 'POR'),
        (DATE'2019-02-18', 65, 'NJU');
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'1997-08-15', 67, 'LKJ'),
        (DATE'1989-12-31', 32, 'YTR');
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2002-10-19', 987, 'KMO'),
        (DATE'2016-11-14', 382, 'VGY');
    """
    )


if __name__ == "__main__":
    create_table()
