from typing import List, Tuple

from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    SortField,
    create_iceberg_table,
    get_spark,
)

# TODO: Open issue in Iceberg GitHub Repo?
# Can not test binary column with partitioning or sorting applied to it.
# Seems to be a bug in Java Spark to deserialze some typing related info
# communicated from the Python side. Throws the error:
# java.lang.ClassCastException: class [B cannot be cast to class java.nio.ByteBuffer
#   ([B and java.nio.ByteBuffer are in module java.base of loader 'bootstrap')
# TODO: [BE-3596] Include void transformation test in Spark
SORT_MAP: List[Tuple[str, List[SortField]]] = [
    # Bool Table
    ("bool_binary_table", [SortField("A", "identity", -1, False, True)]),  # bool
    ("bool_binary_table", [SortField("B", "identity", -1, False, True)]),  # bool
    # Numeric Table
    ("numeric_table", [SortField("A", "identity", -1, False, True)]),  # int
    ("numeric_table", [SortField("B", "identity", -1, False, True)]),  # long
    ("numeric_table", [SortField("A", "truncate", 3, True, True)]),  # int
    ("numeric_table", [SortField("B", "truncate", 3, False, False)]),  # long
    ("numeric_table", [SortField("E", "truncate", 3, True, True)]),  # int null
    ("numeric_table", [SortField("F", "truncate", 3, False, False)]),  # long null
    ("numeric_table", [SortField("A", "bucket", 4, True, False)]),  # int
    ("numeric_table", [SortField("B", "bucket", 4, True, False)]),  # long
    ("numeric_table", [SortField("E", "bucket", 4, True, False)]),  # int
    ("numeric_table", [SortField("F", "bucket", 4, True, False)]),  # long
    ("numeric_table", [SortField("A", "bucket", 50, False, False)]),  # int
    ("numeric_table", [SortField("B", "bucket", 50, False, True)]),  # long
    ("numeric_table", [SortField("E", "bucket", 50, True, False)]),  # int
    ("numeric_table", [SortField("F", "bucket", 50, True, True)]),  # long
    # String Table
    ("string_table", [SortField("B", "identity", -1, False, True)]),
    ("string_table", [SortField("B", "truncate", 1, True, True)]),
    ("string_table", [SortField("D", "truncate", 2, False, False)]),  # nulls
    ("string_table", [SortField("A", "bucket", 4, False, True)]),
    ("string_table", [SortField("A", "bucket", 50, False, True)]),
    # Dict-encoded string table
    (
        "dict_encoded_string_table",
        [SortField("A", "identity", -1, False, True)],
    ),  # w/o nulls
    (
        "dict_encoded_string_table",
        [SortField("B", "identity", -1, True, False)],
    ),  # w/ nulls
    (
        "dict_encoded_string_table",
        [SortField("A", "truncate", 1, True, True)],
    ),  # w/o nulls
    (
        "dict_encoded_string_table",
        [SortField("B", "truncate", 2, False, False)],
    ),  # w/ nulls
    (
        "dict_encoded_string_table",
        [SortField("A", "bucket", 4, False, True)],
    ),  # w/o nulls
    (
        "dict_encoded_string_table",
        [SortField("B", "bucket", 50, False, True)],
    ),  # w/ nulls
    # Date Table
    ("dt_tsz_table", [SortField("A", "identity", -1, False, False)]),
    ("dt_tsz_table", [SortField("A", "bucket", 4, True, True)]),
    ("dt_tsz_table", [SortField("A", "bucket", 50, True, True)]),
    ("dt_tsz_table", [SortField("A", "years", -1, False, True)]),
    ("dt_tsz_table", [SortField("A", "months", -1, True, False)]),
    ("dt_tsz_table", [SortField("A", "days", -1, True, True)]),
    # Datetime Table (w/ NaTs)
    ("dt_tsz_table", [SortField("B", "identity", -1, False, False)]),
    ("dt_tsz_table", [SortField("B", "bucket", 4, True, True)]),
    ("dt_tsz_table", [SortField("B", "bucket", 50, True, True)]),
    ("dt_tsz_table", [SortField("B", "years", -1, False, True)]),
    ("dt_tsz_table", [SortField("B", "months", -1, True, False)]),
    ("dt_tsz_table", [SortField("B", "days", -1, True, True)]),
    ("dt_tsz_table", [SortField("B", "hours", -1, True, True)]),
    # Timestampz Table
    ("tz_aware_table", [SortField("A", "identity", -1, True, True)]),
    ("tz_aware_table", [SortField("A", "bucket", 4, False, False)]),
    ("tz_aware_table", [SortField("A", "bucket", 50, False, False)]),
    ("tz_aware_table", [SortField("A", "years", -1, True, True)]),
    ("tz_aware_table", [SortField("A", "months", -1, False, False)]),
    ("tz_aware_table", [SortField("A", "days", -1, True, False)]),
    ("tz_aware_table", [SortField("A", "hours", -1, False, True)]),
    (
        "primitives_table",
        [
            SortField("A", "months", -1, True, False),
            SortField("B", "truncate", 10, False, True),
        ],
    ),
    (
        "primitives_table",
        [
            SortField("C", "identity", -1, False, False),
            SortField("A", "years", -1, True, True),
        ],
    ),
    (
        "primitives_table",
        [
            SortField("B", "identity", -1, False, True),
            SortField("D", "truncate", 1, False, True),
        ],
    ),
    (
        "primitives_table",
        [
            SortField("D", "bucket", 2, True, False),
            SortField("C", "identity", -1, True, False),
        ],
    ),
    (
        "primitives_table",
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
