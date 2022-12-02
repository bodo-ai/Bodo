from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytz

from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)

# Map table name to a tuple of pandas dataframe, SQL schema, and pyspark schema
# Spark data types: https://spark.apache.org/docs/latest/sql-ref-datatypes.html
# TODO: Missing Types / Tests:
# - Timezone aware timestamp column with Nulls: Pandas converts to NaTs and Spark converts to 0s
# - Boolean Not Null: Bodo always treats boolean arrays as nullable
#   whether or not Pandas treats it a nullable or not
# - Decimal: Bodo / Python doesn't support custom precisions and scale. It works
#   reads, but not for writes, which is why we have separate tables.
BASE_MAP: Dict[str, Tuple[Dict, List]] = {
    "bool_binary_table": (
        {
            "A": np.array([True, False, True, True] * 25, dtype=np.bool_),
            "B": pd.Series([False, None, True, False, None] * 20, dtype="boolean"),
            "C": np.array([1, 1, 0, 1, 0] * 20).tobytes(),
        },
        [
            ("A", "boolean", True),
            ("B", "boolean", True),
            ("C", "binary", True),
        ],
    ),
    "dt_tsz_table": (
        {
            "A": pd.Series(
                [
                    date(2018, 11, 12),
                    date(2019, 11, 12),
                    date(2018, 12, 12),
                    date(2017, 11, 16),
                    None,
                    date(2017, 11, 30),
                    date(2016, 2, 3),
                    date(2019, 11, 12),
                    date(2018, 12, 20),
                    date(2017, 12, 12),
                ]
                * 5
            ),
            "B": pd.Series(
                [
                    datetime.strptime("12/11/2018", "%d/%m/%Y"),
                    datetime.strptime("11/11/2020", "%d/%m/%Y"),
                    datetime.strptime("12/11/2019", "%d/%m/%Y"),
                    None,
                    datetime.strptime("13/11/2018", "%d/%m/%Y"),
                ]
                * 10
            ),
            "C": np.arange(50, dtype=np.int32),
        },
        [
            ("A", "date", True),
            ("B", "timestamp", True),
            ("C", "int", False),
        ],
    ),
    "dtype_list_table": (
        {
            "A": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
            "B": pd.Series([["abc", "rtf"], ["def", "xyz", "typ"]] * 25, dtype=object),
            "C": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
        },
        [
            ("A", "ARRAY<long>", True),
            ("B", "ARRAY<string>", True),
            ("C", "ARRAY<double>", True),
        ],
    ),
    "list_table": (
        {
            "A": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
            "B": pd.Series([["abc", "rtf"], ["def", "xyz", "typ"]] * 25, dtype=object),
            "C": pd.Series([[0, 1, 2], [3, 4]] * 25, dtype=object),
            "D": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
            "E": pd.Series([[0.0, 1.0, 2.0], [3.0, 4.0]] * 25, dtype=object),
        },
        [
            ("A", "ARRAY<long>", True),
            ("B", "ARRAY<string>", True),
            ("C", "ARRAY<int>", True),
            ("D", "ARRAY<float>", True),
            ("E", "ARRAY<double>", True),
        ],
    ),
    "map_table": (
        {
            "A": pd.Series([{"a": 10}, {"c": 13}] * 25, dtype=object),
            "B": pd.Series([{"ERT": 10.0}, {"ASD": 23.87}] * 25, dtype=object),
            "C": pd.Series(
                [{10: Decimal(5.6)}, {65: Decimal(34.6)}] * 25, dtype=object
            ),
            "D": pd.Series(
                [{Decimal(54.67): 54}, {Decimal(32.90): 32}] * 25, dtype=object
            ),
        },
        [
            ("A", "MAP<string, long>", True),
            ("B", "MAP<string, double>", True),
            ("C", "MAP<int, decimal(5,2)>", True),
            ("D", "MAP<decimal(5,2), int>", True),
        ],
    ),
    "numeric_table": (
        {
            "A": pd.Series([1, 2, 3, 4, 5] * 10, dtype="int32"),
            "B": pd.Series([1, 2, 3, 4, 5] * 10, dtype="int64"),
            "C": np.array([1, 2, 3, 4, 5] * 10, np.float32),
            "D": np.array([1, 2, 3, 4, 5] * 10, np.float64),
            "E": pd.Series([6, 7, 8, 9, None] * 10, dtype="Int32"),
            "F": pd.Series([6, 7, 8, 9, None] * 10, dtype="Int64"),
        },
        [
            ("A", "int", False),
            ("B", "long", False),
            ("C", "float", False),
            ("D", "double", False),
            ("E", "int", True),
            ("F", "long", True),
        ],
    ),
    "string_table": (
        {
            "A": np.array(["A", "B", "C", "D"] * 25),
            "B": np.array(["lorem", "ipsum", "loden", "ion"] * 25),
            "C": np.array((["A"] * 10) + (["b"] * 90)),
            "D": np.array(
                (
                    ["four hundred"] * 10
                    + ["five"] * 20
                    + [None] * 10
                    + ["forty-five"] * 10
                    + ["four"] * 20
                    + ["fifeteen"] * 20
                    + ["f"] * 10
                )
            ),
        },
        [
            ("A", "string", True),
            ("B", "string", True),
            ("C", "string", True),
            ("D", "string", True),
        ],
    ),
    "dict_encoded_string_table": (
        {
            "A": pa.array(
                ["abc", "b", "c", "abc", "peach", "b", "cde"] * 20,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            "B": pa.array(
                ["abc", "b", None, "abc", None, "b", "cde"] * 20,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
        },
        [
            ("A", "string", True),
            ("B", "string", True),
        ],
    ),
    "struct_table": (
        {
            "A": pd.Series([{"a": 1, "b": 3}, {"a": 2, "b": 666}] * 25, dtype=object),
            "B": pd.Series(
                [{"a": 2.0, "b": 5, "c": 78.23}, {"a": 1.98, "b": 45, "c": 12.90}] * 25,
                dtype=object,
            ),
            # TODO Add timestamp, datetime, etc. (might not be possible through Spark)
        },
        [
            ("A", "STRUCT<a: long, b: long>", True),
            ("B", "STRUCT<a: double, b: long, c: double>", True),
        ],
    ),
    "struct_dtype_table": (
        {
            "A": pd.Series(
                [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}] * 25, dtype=object
            ),
            "B": pd.Series(
                [
                    {"a": False, "b": date(2019, 5, 5), "c": 78.23},
                    {"a": True, "b": date(2021, 10, 10), "c": 12.90},
                ]
                * 25,
                dtype=object,
            ),
        },
        [
            ("A", "STRUCT<a: long, b: string>", True),
            ("B", "STRUCT<a: boolean, b: date, c: double>", True),
        ],
    ),
    "decimals_table": (
        {"A": np.array([Decimal(1.0), Decimal(2.0)] * 25)},
        [("A", "decimal(10,5)", True)],
    ),
    "decimals_list_table": (
        {
            "A": pd.Series(
                [
                    [Decimal(0.3), Decimal(1.5), Decimal(2.9)],
                    [Decimal(3.4), Decimal(4.8)],
                ]
                * 25,
                dtype=object,
            ),
        },
        [("A", "ARRAY<decimal(5,2)>", True)],
    ),
    "tz_aware_table": (
        {
            "A": pd.arrays.DatetimeArray(
                pd.Series(
                    [datetime(2019, 8, 21, 15, 23, 45, 0, pytz.timezone("US/Eastern"))]
                    * 10
                )
            ),
            "B": pd.arrays.DatetimeArray(
                pd.Series(
                    [
                        datetime(
                            2019, 8, 21, 15, 23, 45, 0, pytz.timezone("Asia/Calcutta")
                        )
                    ]
                    * 10
                )
            ),
        },
        [
            ("A", "timestamp", True),
            ("B", "timestamp", True),
        ],
    ),
    "primitives_table": (
        {
            "A": pd.date_range(
                start="1/1/2019", periods=200, freq="10D", tz="US/Eastern"
            ),
            "B": pd.Series(
                (
                    list(range(10))
                    + [None, None]
                    + list(range(10, 20))
                    + [None, None, None]
                )
                * 8,
                dtype="Int64",
            ),
            "C": pd.Series(
                [True, False, None, None, False, True, True, True, False, False] * 20
            ),
            "D": pd.Series(
                ["one"] * 20
                + ["two", "ten"] * 40
                + [None] * 10
                + ["four", "seven", "five"] * 30
            ),
        },
        [
            ("A", "timestamp", True),
            ("B", "long", True),
            ("C", "boolean", True),
            ("D", "string", True),
        ],
    ),
    "optional_table": (
        {
            "A": np.array([1, 2] * 25, np.int32),
            "B": np.array(["a", "b"] * 25),
        },
        [
            ("A", "int", False),
            ("B", "string", True),
        ],
    ),
}


def build_map(base_map):
    table_map = {}

    for key, (a, b) in base_map.items():
        df = pd.DataFrame(a)
        table_map[key] = (df, b)

    return table_map


TABLE_MAP: Dict[str, Tuple[pd.DataFrame, List]] = build_map(BASE_MAP)


def create_table(base_name: str, spark=None):
    if spark is None:
        spark = get_spark()

    assert base_name in TABLE_MAP, f"Didn't find table definition for {base_name}."
    df, sql_schema = TABLE_MAP[base_name]
    create_iceberg_table(df, sql_schema, f"simple_{base_name}", spark)


def create_all_simple_tables(spark=None):
    if spark is None:
        spark = get_spark()

    for base_table_name in TABLE_MAP:
        create_table(base_table_name, spark)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_all_simple_tables()

    elif len(sys.argv) == 2:
        create_table(sys.argv[1])

    else:
        print("Invalid Number of Arguments")
        exit(1)
