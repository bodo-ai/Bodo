from datetime import datetime

import numpy as np
import pandas as pd

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="filter_pushdown_test_table", spark=None):

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

    # Change to partition on year
    print("Adding partition field (year)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD years(A)
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2017-10-15', 34, 'ERT'),
        (DATE'2017-03-29', 23, 'TRY');
    """
    )

    # Rename column
    print("Renaming column...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        RENAME COLUMN C TO TY;
    """
    )

    # Change to partition on month
    print("Modifying partition field (year --> month)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP PARTITION FIELD years(A)
    """
    )
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
        (DATE'2021-06-19', 45, 'WER'),
        (DATE'2012-06-14', 23, 'FGH');
    """
    )

    # Change to partition on day
    print("Modifying partition field (month --> day)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP PARTITION FIELD months(A)
    """
    )
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD PARTITION FIELD days(A)
    """
    )

    # Move a columm position
    print("Moving a column position...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ALTER COLUMN TY AFTER A;
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2021-06-22', 'WER', 45),
        (DATE'2014-06-15', 'FGH', 23);
    """
    )

    # Remove partition (but not the column, assuming this is possible)
    print("Removing partition field days(A)...")
    spark.sql(
        f""" 
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP PARTITION FIELD days(A)
    """
    )

    # Write more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        (DATE'2021-06-30', 'POR', 87),
        (DATE'2019-02-18', 'NJU', 65);
    """
    )


if __name__ == "__main__":
    create_table()
