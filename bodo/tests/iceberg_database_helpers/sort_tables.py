from typing import List, Tuple

from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    SortField,
    create_iceberg_table,
    get_spark,
)

# TODO: Open issue in Iceberg GitHub Repo?
# Can not test binary column with partitioning or sorting applied to it.
# Seems to be a bug in Java Spark to deserialize some typing related info
# communicated from the Python side. Throws the error:
# java.lang.ClassCastException: class [B cannot be cast to class java.nio.ByteBuffer
#   ([B and java.nio.ByteBuffer are in module java.base of loader 'bootstrap')
# TODO: [BE-3596] Include void transformation test in Spark
SORT_MAP: List[Tuple[str, List[SortField]]] = [
    # Bool Table
    ("BOOL_BINARY_TABLE", [SortField("A", "identity", -1, False, True)]),  # bool
    ("BOOL_BINARY_TABLE", [SortField("B", "identity", -1, False, True)]),  # bool
    # Numeric Table
    ("NUMERIC_TABLE", [SortField("A", "identity", -1, False, True)]),  # int
    ("NUMERIC_TABLE", [SortField("B", "identity", -1, False, True)]),  # long
    ("NUMERIC_TABLE", [SortField("A", "truncate", 3, True, True)]),  # int
    ("NUMERIC_TABLE", [SortField("B", "truncate", 3, False, False)]),  # long
    ("NUMERIC_TABLE", [SortField("E", "truncate", 3, True, True)]),  # int null
    ("NUMERIC_TABLE", [SortField("F", "truncate", 3, False, False)]),  # long null
    ("NUMERIC_TABLE", [SortField("A", "bucket", 4, True, False)]),  # int
    ("NUMERIC_TABLE", [SortField("B", "bucket", 4, True, False)]),  # long
    ("NUMERIC_TABLE", [SortField("E", "bucket", 4, True, False)]),  # int
    ("NUMERIC_TABLE", [SortField("F", "bucket", 4, True, False)]),  # long
    ("NUMERIC_TABLE", [SortField("A", "bucket", 50, False, False)]),  # int
    ("NUMERIC_TABLE", [SortField("B", "bucket", 50, False, True)]),  # long
    ("NUMERIC_TABLE", [SortField("E", "bucket", 50, True, False)]),  # int
    ("NUMERIC_TABLE", [SortField("F", "bucket", 50, True, True)]),  # long
    # String Table
    ("STRING_TABLE", [SortField("B", "identity", -1, False, True)]),
    ("STRING_TABLE", [SortField("B", "truncate", 1, True, True)]),
    ("STRING_TABLE", [SortField("D", "truncate", 2, False, False)]),  # nulls
    ("STRING_TABLE", [SortField("A", "bucket", 4, False, True)]),
    ("STRING_TABLE", [SortField("A", "bucket", 50, False, True)]),
    # Dict-encoded string table
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("A", "identity", -1, False, True)],
    ),  # w/o nulls
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("B", "identity", -1, True, False)],
    ),  # w/ nulls
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("A", "truncate", 1, True, True)],
    ),  # w/o nulls
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("B", "truncate", 2, False, False)],
    ),  # w/ nulls
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("A", "bucket", 4, False, True)],
    ),  # w/o nulls
    (
        "DICT_ENCODED_STRING_TABLE",
        [SortField("B", "bucket", 50, False, True)],
    ),  # w/ nulls
    # Date Table
    ("DT_TSZ_TABLE", [SortField("A", "identity", -1, False, False)]),
    ("DT_TSZ_TABLE", [SortField("A", "bucket", 4, True, True)]),
    ("DT_TSZ_TABLE", [SortField("A", "bucket", 50, True, True)]),
    ("DT_TSZ_TABLE", [SortField("A", "years", -1, False, True)]),
    ("DT_TSZ_TABLE", [SortField("A", "months", -1, True, False)]),
    ("DT_TSZ_TABLE", [SortField("A", "days", -1, True, True)]),
    # Datetime Table (w/ NaTs)
    ("DT_TSZ_TABLE", [SortField("B", "identity", -1, False, False)]),
    ("DT_TSZ_TABLE", [SortField("B", "bucket", 4, True, True)]),
    ("DT_TSZ_TABLE", [SortField("B", "bucket", 50, True, True)]),
    ("DT_TSZ_TABLE", [SortField("B", "years", -1, False, True)]),
    ("DT_TSZ_TABLE", [SortField("B", "months", -1, True, False)]),
    ("DT_TSZ_TABLE", [SortField("B", "days", -1, True, True)]),
    ("DT_TSZ_TABLE", [SortField("B", "hours", -1, True, True)]),
    # Timestamps Table
    ("TZ_AWARE_TABLE", [SortField("A", "identity", -1, True, True)]),
    ("TZ_AWARE_TABLE", [SortField("A", "bucket", 4, False, False)]),
    ("TZ_AWARE_TABLE", [SortField("A", "bucket", 50, False, False)]),
    ("TZ_AWARE_TABLE", [SortField("A", "years", -1, True, True)]),
    ("TZ_AWARE_TABLE", [SortField("A", "months", -1, False, False)]),
    ("TZ_AWARE_TABLE", [SortField("A", "days", -1, True, False)]),
    ("TZ_AWARE_TABLE", [SortField("A", "hours", -1, False, True)]),
    (
        "PRIMITIVES_TABLE",
        [
            SortField("A", "months", -1, True, False),
            SortField("B", "truncate", 10, False, True),
        ],
    ),
    (
        "PRIMITIVES_TABLE",
        [
            SortField("C", "identity", -1, False, False),
            SortField("A", "years", -1, True, True),
        ],
    ),
    (
        "PRIMITIVES_TABLE",
        [
            SortField("B", "identity", -1, False, True),
            SortField("D", "truncate", 1, False, True),
        ],
    ),
    (
        "PRIMITIVES_TABLE",
        [
            SortField("D", "bucket", 2, True, False),
            SortField("C", "identity", -1, True, False),
        ],
    ),
    (
        "PRIMITIVES_TABLE",
        [
            SortField("B", "bucket", 4, True, True),
            SortField("A", "bucket", 3, False, False),
        ],
    ),
]


def sort_table_name(base_name, sort_order):
    col, trans, val, asc, nulls_first = sort_order[0]
    val_str = f"_{val}" if trans in ["bucket", "truncate"] else ""
    asc_str = "asc" if asc else "desc"
    nulls_str = "first" if nulls_first else "last"
    return f"sort_{base_name}_{col}_{trans}{val_str}_{asc_str}_{nulls_str}"


def create_table(base_name, sort_order, spark=None):
    if spark is None:
        spark = get_spark()

    assert base_name in TABLE_MAP, f"Didn't find table definition for {base_name}."
    df, sql_schema = TABLE_MAP[base_name]

    create_iceberg_table(
        df,
        sql_schema,
        sort_table_name(base_name, sort_order),
        spark,
        par_spec=None,
        sort_order=sort_order,
    )


def create_all_sort_tables(spark=None):
    if spark is None:
        spark = get_spark()

    for base_name, sort_order in SORT_MAP:
        create_table(base_name, sort_order, spark)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_all_sort_tables()
    else:
        print("Invalid Number of Arguments")
        exit(1)
