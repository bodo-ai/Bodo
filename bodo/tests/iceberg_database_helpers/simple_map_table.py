from decimal import Decimal

import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark


def create_table(table_name="simple_map_table", spark=None):

    if spark is None:
        spark = get_spark()

    df = pd.DataFrame(
        {
            "A": pd.Series([{"a": 10}, {"c": 13}] * 25, dtype=object),
            "B": pd.Series(
                [{10: Decimal(5.6)}, {65: Decimal(34.6)}] * 25, dtype=object
            ),
            "C": pd.Series([{"ERT": 10.0}, {"ASD": 23.87}] * 25, dtype=object),
            "D": pd.Series(
                [{Decimal(54.67): 54}, {Decimal(32.90): 32}] * 25, dtype=object
            ),
        }
    )
    schema = spark_types.StructType(
        [
            spark_types.StructField(
                "A",
                spark_types.MapType(
                    spark_types.StringType(), spark_types.LongType(), True
                ),
            ),
            spark_types.StructField(
                "B",
                spark_types.MapType(
                    spark_types.IntegerType(), spark_types.DecimalType(5, 2), True
                ),
            ),
            spark_types.StructField(
                "C",
                spark_types.MapType(
                    spark_types.StringType(), spark_types.DoubleType(), True
                ),
            ),
            spark_types.StructField(
                "D",
                spark_types.MapType(
                    spark_types.DecimalType(5, 2), spark_types.IntegerType(), True
                ),
            ),
        ]
    )

    create_iceberg_table(df, schema, table_name, spark)


if __name__ == "__main__":
    create_table()
