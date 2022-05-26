from decimal import Decimal

import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark


def create_table(table_name="simple_list_table", spark=None):
    if spark is None:
        spark = get_spark()

    df = pd.DataFrame(
        {
            "A": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
            "B": pd.Series([["abc", "rtf"], ["def", "xyz", "typ"]] * 25, dtype=object),
            "C": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
            "D": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
            "E": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
            "F": pd.Series(
                [
                    [Decimal(0.3), Decimal(1.5), Decimal(2.9)],
                    [Decimal(3.4), Decimal(4.8)],
                ]
                * 25,
                dtype=object,
            ),
        }
    )
    schema = spark_types.StructType(
        [
            spark_types.StructField(
                "A", spark_types.ArrayType(spark_types.LongType(), True)
            ),
            spark_types.StructField(
                "B", spark_types.ArrayType(spark_types.StringType(), True)
            ),
            spark_types.StructField(
                "C", spark_types.ArrayType(spark_types.IntegerType(), True)
            ),
            spark_types.StructField(
                "D", spark_types.ArrayType(spark_types.FloatType(), True)
            ),
            spark_types.StructField(
                "E", spark_types.ArrayType(spark_types.DoubleType(), True)
            ),
            spark_types.StructField(
                "F", spark_types.ArrayType(spark_types.DecimalType(5, 2), True)
            ),
        ]
    )

    create_iceberg_table(df, schema, table_name, spark)


if __name__ == "__main__":
    create_table()
