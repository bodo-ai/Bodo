import pandas as pd
import pyspark.sql.types as spark_types

from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark


def create_table(table_name="simple_struct_table", spark=None):

    if spark is None:
        spark = get_spark()

    df = pd.DataFrame(
        {
            "A": pd.Series([["f3e", 3], ["e3", 666]] * 25, dtype=object),
            "B": pd.Series([[2.0, 5, 78.23], [1.98, 45, 12.90]] * 25, dtype=object),
            # TODO Add timestamp, datetime, etc. (might not be possible through Spark)
        }
    )
    schema = spark_types.StructType(
        [
            spark_types.StructField(
                "A",
                spark_types.StructType(
                    [
                        spark_types.StructField("a", spark_types.StringType()),
                        spark_types.StructField("b", spark_types.LongType()),
                    ]
                ),
                True,
            ),
            spark_types.StructField(
                "B",
                spark_types.StructType(
                    [
                        spark_types.StructField(
                            "a", spark_types.FloatType(), nullable=True
                        ),
                        spark_types.StructField("b", spark_types.IntegerType(), True),
                        spark_types.StructField("c", spark_types.DoubleType()),
                    ]
                ),
                True,
            ),
        ]
    )

    create_iceberg_table(df, schema, table_name, spark)


if __name__ == "__main__":
    create_table()
