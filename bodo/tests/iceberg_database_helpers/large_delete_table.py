import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    create_iceberg_table,
    get_spark,
)

"""
Test a large delete the should use delete files instead of
rewriting data
"""


def create_table(table_name="large_delete_table", spark=None):

    if spark is None:
        spark = get_spark()

    # Write a simple dataset
    print("Writing a simple dataset...")
    df = pd.DataFrame(
        {
            "A": np.array(["A", "B", "C", "D"] * 5 * 1000),
            "B": np.array(["lorem", "ipsum"] * 10 * 1000),
            "C": np.array((["A"] * 2 * 1000) + (["b"] * 18 * 1000)),
            "D": np.array([1, 2] * 10 * 1000, np.int32),
            "E": np.array([1, 2] * 10 * 1000, np.float32),
            "K": np.array([54] * 20 * 1000, np.int64),
        }
    )
    sql_schema = [
        ("A", "string"),
        ("B", "string"),
        ("C", "string"),
        ("D", "int"),
        ("E", "float"),
        ("K", "long"),
    ]
    spark_schema = spark_types.StructType(
        [
            spark_types.StructField("A", spark_types.StringType(), True),
            spark_types.StructField("B", spark_types.StringType(), True),
            spark_types.StructField("C", spark_types.StringType(), True),
            spark_types.StructField("D", spark_types.IntegerType(), True),
            spark_types.StructField("E", spark_types.FloatType(), True),
            spark_types.StructField("K", spark_types.LongType(), True),
        ]
    )
    create_iceberg_table(df, sql_schema, spark_schema, table_name, spark)

    # Delete all some rows
    print("Deleting rows...")
    spark.sql(
        f"""
        DELETE FROM hadoop_prod.{DATABASE_NAME}.{table_name}
        WHERE A = 'B';
    """
    )


if __name__ == "__main__":
    create_table()
