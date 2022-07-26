import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)


def create_table(table_name="schema_evolution_eg_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(["A", "B", "C", "D"] * 25),
            "B": np.array(["lorem", "ipsum"] * 50),
            "C": np.array((["A"] * 10) + (["b"] * 90)),
            "D": np.array([1, 2] * 50, np.int32),
            "E": np.array([1, 2] * 50, np.float32),
        }
    )
    sql_schema = [
        ("A", "string"),
        ("B", "string"),
        ("C", "string"),
        ("D", "int"),
        ("E", "float"),
    ]
    spark_schema = spark_types.StructType(
        [
            spark_types.StructField("A", spark_types.StringType(), True),
            spark_types.StructField("B", spark_types.StringType(), True),
            spark_types.StructField("C", spark_types.StringType(), True),
            spark_types.StructField("D", spark_types.IntegerType(), True),
            spark_types.StructField("E", spark_types.FloatType(), True),
        ]
    )
    create_iceberg_table(df, sql_schema, spark_schema, table_name, spark)

    # Add column
    print("Adding column...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ADD COLUMN F int AFTER E;
    """
    )

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

    # Rename column
    print("Renaming column...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        RENAME COLUMN C TO TY;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        (A, F, TY, D, B, E)
        VALUES
        ('ZXCVBN', 667, 'POW', 2894, 'return', 32.32),
        ('DFGHJK', 954, 'QWRM', 523, 'catch', 12.12);
    """
    )

    # Remove column
    print("Removing column...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        DROP COLUMN B;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('YUIOP', 'INTJ', 5289, 89.21, 90);
    """
    )

    # Change dtype of column
    print("Changing dtype of a column...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ALTER COLUMN D TYPE bigint;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('VBNMLK', 'ESPQ', 1222435, 90.45, 1234);
    """
    )

    # Move a columm position
    print("Moving a column position...")
    spark.sql(
        f"""
        ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table_name}
        ALTER COLUMN D AFTER A;
    """
    )

    # Add more data
    print("Adding some data...")
    spark.sql(
        f"""
        INSERT INTO hadoop_prod.{DATABASE_NAME}.{table_name}
        VALUES
        ('FGHJKL', 34789, 'MNWE', 75.23, 8723);
    """
    )


if __name__ == "__main__":
    create_table()
