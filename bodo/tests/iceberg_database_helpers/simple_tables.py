from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
import pytz

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

# Map table name to a tuple of pandas dataframe and pyspark schema
# Spark data types: https://spark.apache.org/docs/latest/sql-ref-datatypes.html
TABLE_MAP = {
    "simple_bool_binary_table": (
        pd.DataFrame(
            {
                "A": np.array([True, False, True, True] * 25, dtype=np.bool_),
                "B": np.array([False, None] * 50, dtype=np.bool_),
                "C": np.array([1, 1, 0, 1, 0] * 20).tobytes(),
            }
        ),
        [
            ("A", "boolean", "not null"),
            ("B", "boolean"),
            ("C", "binary"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.BooleanType(), False),
                spark_types.StructField("B", spark_types.BooleanType(), True),
                spark_types.StructField("C", spark_types.BinaryType(), True),
            ]
        ),
    ),
    "simple_dt_tsz_table": (
        pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        datetime.strptime("12/11/2018", "%d/%m/%Y").date(),
                        datetime.strptime("12/11/2019", "%d/%m/%Y").date(),
                        datetime.strptime("12/12/2018", "%d/%m/%Y").date(),
                        datetime.strptime("13/11/2018", "%d/%m/%Y").date(),
                    ]
                    * 5
                ),
                "B": pd.Series(
                    [
                        datetime.strptime("12/11/2018", "%d/%m/%Y").date(),
                        datetime.strptime("12/11/2019", "%d/%m/%Y").date(),
                        datetime.strptime("12/12/2018", "%d/%m/%Y").date(),
                        datetime.strptime("13/11/2018", "%d/%m/%Y").date(),
                    ]
                    * 5
                ),
                "C": pd.Series(
                    # For some reason, we need to wrap datetime arrays in np.array for pandas to catch the correct dtype
                    np.array(
                        [
                            datetime(
                                2019, 8, 21, 15, 23, 45, 0, pytz.timezone("US/Eastern")
                            ),
                            datetime(
                                2019,
                                8,
                                21,
                                15,
                                23,
                                45,
                                0,
                                pytz.timezone("Asia/Calcutta"),
                            ),
                        ]
                        * 10,
                        dtype="datetime64[ns]",
                    )
                ),
            }
        ),
        [
            ("A", "date"),
            ("B", "date"),
            ("C", "timestamp"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.DateType(), True),
                spark_types.StructField("B", spark_types.DateType(), True),
                spark_types.StructField("C", spark_types.TimestampType(), True),
            ]
        ),
    ),
    "simple_dtype_list_table": (
        pd.DataFrame(
            {
                "A": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
                "B": pd.Series(
                    [["abc", "rtf"], ["def", "xyz", "typ"]] * 25, dtype=object
                ),
                "C": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
            }
        ),
        [
            ("A", "ARRAY<long>"),
            ("B", "ARRAY<string>"),
            ("C", "ARRAY<double>"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField(
                    "A", spark_types.ArrayType(spark_types.LongType(), True)
                ),
                spark_types.StructField(
                    "B", spark_types.ArrayType(spark_types.StringType(), True)
                ),
                spark_types.StructField(
                    "C", spark_types.ArrayType(spark_types.DoubleType(), True)
                ),
            ]
        ),
    ),
    "simple_list_table": (
        pd.DataFrame(
            {
                "A": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
                "B": pd.Series(
                    [["abc", "rtf"], ["def", "xyz", "typ"]] * 25, dtype=object
                ),
                "C": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
                "D": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
                "E": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
            }
        ),
        [
            ("A", "ARRAY<long>"),
            ("B", "ARRAY<string>"),
            ("C", "ARRAY<int>"),
            ("D", "ARRAY<float>"),
            ("E", "ARRAY<double>"),
        ],
        spark_types.StructType(
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
            ]
        ),
    ),
    "simple_map_table": (
        pd.DataFrame(
            {
                "A": pd.Series([{"a": 10}, {"c": 13}] * 25, dtype=object),
                "B": pd.Series([{"ERT": 10.0}, {"ASD": 23.87}] * 25, dtype=object),
                "C": pd.Series(
                    [{10: Decimal(5.6)}, {65: Decimal(34.6)}] * 25, dtype=object
                ),
                "D": pd.Series(
                    [{Decimal(54.67): 54}, {Decimal(32.90): 32}] * 25, dtype=object
                ),
            }
        ),
        [
            ("A", "MAP<string, long>"),
            ("B", "MAP<string, double>"),
            ("C", "MAP<int, decimal(5,2)>"),
            ("D", "MAP<decimal(5,2), int>"),
        ],
        spark_types.StructType(
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
                        spark_types.StringType(), spark_types.DoubleType(), True
                    ),
                ),
                spark_types.StructField(
                    "C",
                    spark_types.MapType(
                        spark_types.IntegerType(), spark_types.DecimalType(5, 2), True
                    ),
                ),
                spark_types.StructField(
                    "D",
                    spark_types.MapType(
                        spark_types.DecimalType(5, 2), spark_types.IntegerType(), True
                    ),
                ),
            ]
        ),
    ),
    "simple_numeric_table": (
        pd.DataFrame(
            {
                "A": np.array([1, 2] * 25, np.int32),
                "B": np.array([1, 2] * 25, np.int64),
                "C": np.array([1, 2] * 25, np.float32),
                "D": np.array([1, 2] * 25, np.float64),
            }
        ),
        [
            ("A", "int"),
            ("B", "long"),
            ("C", "float"),
            ("D", "double", "not null"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.IntegerType(), True),
                spark_types.StructField("B", spark_types.LongType(), True),
                spark_types.StructField("C", spark_types.FloatType(), True),
                spark_types.StructField("D", spark_types.DoubleType(), False),
            ]
        ),
    ),
    "simple_string_table": (
        pd.DataFrame(
            {
                "A": np.array(["A", "B", "C", "D"] * 25),
                "B": np.array(["lorem", "ipsum"] * 50),
                "C": np.array((["A"] * 10) + (["b"] * 90)),
            }
        ),
        [
            ("A", "string"),
            ("B", "string"),
            ("C", "string"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.StringType(), True),
                spark_types.StructField("B", spark_types.StringType(), True),
                spark_types.StructField("C", spark_types.StringType(), True),
            ]
        ),
    ),
    # TODO Create simple_dtype_struct_table,
    "simple_struct_table": (
        pd.DataFrame(
            {
                "A": pd.Series(
                    [{"a": 1, "b": 3}, {"a": 2, "b": 666}] * 25, dtype=object
                ),
                "B": pd.Series(
                    [{"a": 2.0, "b": 5, "c": 78.23}, {"a": 1.98, "b": 45, "c": 12.90}]
                    * 25,
                    dtype=object,
                ),
                # TODO Add timestamp, datetime, etc. (might not be possible through Spark)
            }
        ),
        [
            ("A", "STRUCT<a: long, b: long>"),
            ("B", "STRUCT<a: double, b: long, c: double>"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField(
                    "A",
                    spark_types.StructType(
                        [
                            spark_types.StructField("a", spark_types.LongType(), True),
                            spark_types.StructField("b", spark_types.LongType(), True),
                        ]
                    ),
                    True,
                ),
                spark_types.StructField(
                    "B",
                    spark_types.StructType(
                        [
                            spark_types.StructField(
                                "a", spark_types.DoubleType(), nullable=True
                            ),
                            spark_types.StructField("b", spark_types.LongType(), True),
                            spark_types.StructField(
                                "c", spark_types.DoubleType(), True
                            ),
                        ]
                    ),
                    True,
                ),
            ]
        ),
    ),
    # We currently don't have support for custom precisions and scale in Bodo,
    # hence we have separate tables for decimals. We can still use it for reads, but not
    # for writes, so we skip it for that.
    "simple_decimals_table": (
        pd.DataFrame(
            {
                # Scalar
                "A": np.array([Decimal(1.0), Decimal(2.0)] * 25),
            }
        ),
        [
            ("A", "decimal(10,5)"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.DecimalType(10, 5), True),
            ]
        ),
    ),
    "simple_decimals_list_table": (
        pd.DataFrame(
            {
                # List
                "A": pd.Series(
                    [
                        [Decimal(0.3), Decimal(1.5), Decimal(2.9)],
                        [Decimal(3.4), Decimal(4.8)],
                    ]
                    * 25,
                    dtype=object,
                ),
            }
        ),
        [
            ("A", "ARRAY<decimal(5,2)>"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField(
                    "A", spark_types.ArrayType(spark_types.DecimalType(5, 2), True)
                ),
            ]
        ),
    ),
    "simple_tz_aware_table": (
        pd.DataFrame(
            {
                "A": pd.arrays.DatetimeArray(
                    pd.Series(
                        [
                            datetime(
                                2019, 8, 21, 15, 23, 45, 0, pytz.timezone("US/Eastern")
                            )
                        ]
                        * 10
                    )
                ),
                "B": pd.arrays.DatetimeArray(
                    pd.Series(
                        [
                            datetime(
                                2019,
                                8,
                                21,
                                15,
                                23,
                                45,
                                0,
                                pytz.timezone("Asia/Calcutta"),
                            )
                        ]
                        * 10
                    )
                ),
            }
        ),
        [
            ("A", "timestamp"),
            ("B", "timestamp"),
        ],
        spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.TimestampType(), True),
                spark_types.StructField("B", spark_types.TimestampType(), True),
            ]
        ),
    ),
}


def create_table(table_name, spark=None):
    if spark is None:
        spark = get_spark()

    assert table_name in TABLE_MAP, f"Didn't find table definition for {table_name}."
    df, sql_schema, spark_schema = TABLE_MAP[table_name]
    create_iceberg_table(df, sql_schema, spark_schema, table_name, spark)


def create_all_simple_tables():
    for table_name in TABLE_MAP:
        create_table(table_name)


if __name__ == "__main__":
    create_all_simple_tables()
