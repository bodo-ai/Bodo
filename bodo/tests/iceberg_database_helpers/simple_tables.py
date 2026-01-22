from datetime import date, datetime
from decimal import Decimal

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
BASE_MAP: dict[str, tuple[dict, list]] = {
    "BOOL_BINARY_TABLE": (
        {
            "A": np.array([True, False, True, True] * 25, dtype=np.bool_),
            "B": pd.Series([False, None, True, False, None] * 20, dtype="boolean"),
            "C": np.array([b"1", b"1", b"0", b"1", b"0"] * 20, dtype=object),
        },
        [
            ("A", "boolean", True),
            ("B", "boolean", True),
            ("C", "binary", True),
        ],
    ),
    "DT_TSZ_TABLE": (
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
            ).dt.tz_localize("UTC"),
            "C": np.arange(50, dtype=np.int32),
        },
        [
            ("A", "date", True),
            ("B", "timestamp", True),
            ("C", "int", False),
        ],
    ),
    # TODO figure out why pyspark won't accept series with a pyarrow list type
    "DTYPE_LIST_TABLE": (
        {
            "A": pd.Series(
                [[0, 1, 2], [3, 4]] * 25, dtype=pd.ArrowDtype(pa.large_list(pa.int64()))
            ),
            "B": pd.Series(
                [["abc", "rtf"], ["def", "xyz", "typ"]] * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            "C": pd.Series(
                [[0.0, 1.0, 2.0], [3.0, 4.0]] * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
            ),
        },
        [
            ("A", "ARRAY<long>", True),
            ("B", "ARRAY<string>", True),
            ("C", "ARRAY<double>", True),
        ],
    ),
    "LIST_TABLE": (
        {
            "A": pd.Series(
                [[0, 1, 2], [3, 4]] * 25, dtype=pd.ArrowDtype(pa.large_list(pa.int64()))
            ),
            "B": pd.Series(
                [["abc", "rtf"], ["def", "xyz", "typ"]] * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            "C": pd.Series(
                [[0, 1, 2], [3, 4]] * 25, dtype=pd.ArrowDtype(pa.large_list(pa.int32()))
            ),
            "D": pd.Series(
                [[0.0, 1.0, 2.0], [3.0, 4.0]] * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.float32())),
            ),
            "E": pd.Series(
                [[0.0, 1.0, 2.0], [3.0, 4.0]] * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
            ),
        },
        [
            ("A", "ARRAY<long>", True),
            ("B", "ARRAY<string>", True),
            ("C", "ARRAY<int>", True),
            ("D", "ARRAY<float>", True),
            ("E", "ARRAY<double>", True),
        ],
    ),
    # https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#supported-sql-types
    # MapArray types not supported
    # Arrow Maps only support string or byte keys
    "MAP_TABLE": (
        {
            "A": pd.Series([{"a": 10}, {"c": 13}] * 25, dtype=object),
            # dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            "B": pd.Series([{"ERT": 10.0}, {"ASD": 23.87}] * 25, dtype=object),
            # dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.float64())),
            "C": pd.Series(
                [{"10": Decimal("005.60")}, {"65": Decimal("034.60")}] * 25,
                dtype=object,
            ),
            # kdtype=pd.ArrowDtype(pa.map_(pa.string(), pa.decimal128(5, 2))),
            "D": pd.Series([{"54.67": 54}, {"32.90": 32}] * 25, dtype=object),
            # dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
        },
        [
            ("A", "MAP<string, int>", True),
            ("B", "MAP<string, double>", True),
            ("C", "MAP<string, decimal(5,2)>", True),
            ("D", "MAP<string, int>", True),
        ],
    ),
    "NUMERIC_TABLE": (
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
    "STRING_TABLE": (
        {
            "A": np.array(["A", "B", "C", "D"] * 25),
            "B": np.array(["lorem", "ipsum", "loden", "ion"] * 25),
            "C": np.array((["A"] * 10) + (["b"] * 90)),
            "D": np.array(
                ["four hundred"] * 10
                + ["five"] * 20
                + [None] * 10
                + ["forty-five"] * 10
                + ["four"] * 20
                + ["fifeteen"] * 20
                + ["f"] * 10
            ),
        },
        [
            ("A", "string", True),
            ("B", "string", True),
            ("C", "string", True),
            ("D", "string", True),
        ],
    ),
    "DICT_ENCODED_STRING_TABLE": (
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
    "STRUCT_TABLE": (
        {
            "A": pd.Series(
                [{"a": 1, "b": 3}, {"a": 2, "b": 666}] * 25,
                dtype=pd.ArrowDtype(pa.struct([("a", pa.int32()), ("b", pa.int64())])),
            ),
            "B": pd.Series(
                [{"a": 2.0, "b": 5, "c": 78.23}, {"a": 1.98, "b": 45, "c": 12.90}] * 25,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [("a", pa.float64()), ("b", pa.int64()), ("c", pa.float64())]
                    )
                ),
            ),
            # TODO Add timestamp, datetime, etc. (might not be possible through Spark)
        },
        [
            ("A", "STRUCT<a: int, b: long>", True),
            ("B", "STRUCT<a: double, b: long, c: double>", True),
        ],
    ),
    "STRUCT_DTYPE_TABLE": (
        {
            "A": pd.Series(
                [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}] * 25,
                dtype=pd.ArrowDtype(pa.struct([("a", pa.int64()), ("b", pa.string())])),
            ),
            "B": pd.Series(
                [
                    {"a": False, "b": date(2019, 5, 5), "c": 78.23},
                    {"a": True, "b": date(2021, 10, 10), "c": 12.90},
                ]
                * 25,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [("a", pa.bool_()), ("b", pa.date32()), ("c", pa.float64())]
                    )
                ),
            ),
        },
        [
            ("A", "STRUCT<a: long, b: string>", True),
            ("B", "STRUCT<a: boolean, b: date, c: double>", True),
        ],
    ),
    "DECIMALS_TABLE": (
        {
            "A": np.array([Decimal(1.0), Decimal(2.0)] * 25),
            "B": np.array([Decimal(5.0), Decimal(10.0)] * 25),
        },
        [("A", "decimal(10,5)", True), ("B", "decimal(38,18)", True)],
    ),
    # TODO figure out why pyspark won't accept series with a pyarrow list type
    "DECIMALS_LIST_TABLE": (
        {
            "A": pd.Series(
                [
                    [Decimal("000.30"), Decimal("001.50"), Decimal("002.90")],
                    [Decimal("003.40"), Decimal("004.80")],
                ]
                * 25,
                dtype=pd.ArrowDtype(pa.large_list(pa.decimal128(5, 2))),
            ),
            "B": pd.Series(
                [
                    [Decimal("000.34"), Decimal("021.50"), Decimal("202.90")],
                    [Decimal("013.40"), Decimal("004.90")],
                ]
                * 25,
                dtype=object,  # pd.ArrowDtype(pa.large_list(pa.decimal128(5, 2))),
            ),
        },
        [("A", "ARRAY<decimal(5,2)>", True), ("B", "ARRAY<decimal(5,2)>", True)],
    ),
    "TZ_AWARE_TABLE": (
        {
            "A": pd.array(
                pd.Series(
                    [datetime(2019, 8, 21, 15, 23, 45, 0, pytz.timezone("US/Eastern"))]
                    * 10
                )
            ),
            "B": pd.array(
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
    "PRIMITIVES_TABLE": (
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
    "OPTIONAL_TABLE": (
        {
            "A": np.array([1, 2] * 25, np.int32),
            "B": np.array(["a", "b"] * 25),
        },
        [
            ("A", "int", False),
            ("B", "string", True),
        ],
    ),
    "OPTIONAL_TABLE_MIDDLE": (
        {
            "A": np.array([1, 2] * 25, np.int32),
            "B": np.array(["a", "b"] * 25),
            "C": pd.Series(
                [{"f1": 1.0, "f2": "A", "f3": 3.0}, {"f1": 4.0, "f2": "b", "f3": 6.0}]
                * 25,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            ("f1", pa.float64()),
                            ("f2", pa.string()),
                            ("f3", pa.float64()),
                        ]
                    )
                ),
            ),
            "D": np.array([1, 2] * 25, np.int64),
        },
        [
            ("A", "int", False),
            ("B", "string", True),
            ("C", "STRUCT<f1: double, f2: string, f3: double>", True),
            ("D", "long", False),
        ],
    ),
}


def build_map(base_map):
    table_map = {}

    for key, (a, b) in base_map.items():
        df = pd.DataFrame(a)
        table_map[f"SIMPLE_{key}"] = (df, b)

    return table_map


TABLE_MAP: dict[str, tuple[pd.DataFrame, list]] = build_map(BASE_MAP)


def create_table(base_name: str, spark=None):
    if spark is None:
        spark = get_spark()

    assert base_name in TABLE_MAP, f"Didn't find table definition for {base_name}."
    df, sql_schema = TABLE_MAP[base_name]
    create_iceberg_table(df, sql_schema, base_name, spark)


def create_simple_tables(tables: list[str], spark=None):
    if spark is None:
        spark = get_spark()

    for table in tables:
        if table in TABLE_MAP:
            create_table(table, spark)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_simple_tables(list(TABLE_MAP.keys()))

    elif len(sys.argv) == 2:
        create_table(sys.argv[1])

    else:
        print("Invalid Number of Arguments")
        exit(1)
