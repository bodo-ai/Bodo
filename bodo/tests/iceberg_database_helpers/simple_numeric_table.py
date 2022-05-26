from decimal import Decimal

import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark


def create_table(table_name="simple_numeric_table", spark=None):

    if spark is None:
        spark = get_spark()

    df = pd.DataFrame(
        {
            "A": np.array([1, 2] * 25, np.int32),
            "B": np.array([1, 2] * 25, np.int64),
            "C": np.array([1, 2] * 25, np.float32),
            "D": np.array([1, 2] * 25, np.float64),
            "E": np.array([Decimal(1.0), Decimal(2.0)] * 25),
        }
    )
    schema = spark_types.StructType(
        [
            spark_types.StructField("A", spark_types.IntegerType(), True),
            spark_types.StructField("B", spark_types.LongType(), True),
            spark_types.StructField("C", spark_types.FloatType(), True),
            spark_types.StructField("D", spark_types.DoubleType(), False),
            spark_types.StructField("E", spark_types.DecimalType(10, 5), True),
        ]
    )

    create_iceberg_table(df, schema, table_name, spark)


if __name__ == "__main__":
    create_table()
