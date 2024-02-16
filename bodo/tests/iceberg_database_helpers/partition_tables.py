from typing import Dict, List, Tuple

from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
    get_spark,
)

PARTITION_MAP: List[Tuple[str, List[PartitionField]]] = [
    # Identity for Bools
    ("BOOL_BINARY_TABLE", [PartitionField("A", "identity", -1)]),  # bool not null
    ("BOOL_BINARY_TABLE", [PartitionField("B", "identity", -1)]),  # bool null
    # Identity
    ("NUMERIC_TABLE", [PartitionField("A", "identity", -1)]),  # int32 no nulls
    ("NUMERIC_TABLE", [PartitionField("B", "identity", -1)]),  # int64 no nulls
    ("NUMERIC_TABLE", [PartitionField("E", "identity", -1)]),  # int32
    ("NUMERIC_TABLE", [PartitionField("F", "identity", -1)]),  # int64
    ("STRING_TABLE", [PartitionField("B", "identity", -1)]),  # string no nulls
    ("DT_TSZ_TABLE", [PartitionField("A", "identity", -1)]),  # date with nulls
    ("DT_TSZ_TABLE", [PartitionField("B", "identity", -1)]),  # datetime with NaTs
    ("TZ_AWARE_TABLE", [PartitionField("A", "identity", -1)]),  # datetime no nulls
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("A", "identity", -1)]),  # w/o nulls
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("B", "identity", -1)]),  # w/ nulls
    # Date / Time Transformations
    ("DT_TSZ_TABLE", [PartitionField("A", "years", -1)]),
    ("DT_TSZ_TABLE", [PartitionField("A", "months", -1)]),
    ("DT_TSZ_TABLE", [PartitionField("A", "days", -1)]),
    ("DT_TSZ_TABLE", [PartitionField("B", "years", -1)]),  # datetime with NaTs
    ("DT_TSZ_TABLE", [PartitionField("B", "months", -1)]),  # datetime with NaTs
    ("DT_TSZ_TABLE", [PartitionField("B", "days", -1)]),  # datetime with NaTs
    ("DT_TSZ_TABLE", [PartitionField("B", "hours", -1)]),  # datetime with NaTs
    ("TZ_AWARE_TABLE", [PartitionField("A", "years", -1)]),  # datetime w/o NaTs
    ("TZ_AWARE_TABLE", [PartitionField("A", "months", -1)]),  # datetime w/o NaTs
    ("TZ_AWARE_TABLE", [PartitionField("A", "days", -1)]),  # datetime w/o NaTs
    ("TZ_AWARE_TABLE", [PartitionField("A", "hours", -1)]),  # datetime w/o NaTs
    # TODO: Include timestamps?
    # Truncate Transformation
    ("NUMERIC_TABLE", [PartitionField("A", "truncate", 3)]),  # int
    ("NUMERIC_TABLE", [PartitionField("B", "truncate", 3)]),  # long
    ("NUMERIC_TABLE", [PartitionField("E", "truncate", 3)]),  # int nulls
    ("NUMERIC_TABLE", [PartitionField("F", "truncate", 3)]),  # long nulls
    ("STRING_TABLE", [PartitionField("B", "truncate", 1)]),  # string
    ("STRING_TABLE", [PartitionField("D", "truncate", 2)]),  # string nulls
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("A", "truncate", 1)]),  # w/o nulls
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("B", "truncate", 2)]),  # w/ nulls
    # Bucket Transformation
    ("NUMERIC_TABLE", [PartitionField("A", "bucket", 4)]),  # int
    ("NUMERIC_TABLE", [PartitionField("B", "bucket", 4)]),  # long
    ("NUMERIC_TABLE", [PartitionField("E", "bucket", 4)]),  # int
    ("NUMERIC_TABLE", [PartitionField("F", "bucket", 4)]),  # long
    ("STRING_TABLE", [PartitionField("A", "bucket", 4)]),  # string
    ("DT_TSZ_TABLE", [PartitionField("A", "bucket", 4)]),  # date
    ("DT_TSZ_TABLE", [PartitionField("B", "bucket", 4)]),  # datetime (w/ NaTs)
    ("TZ_AWARE_TABLE", [PartitionField("A", "bucket", 4)]),  # timestamp
    ("NUMERIC_TABLE", [PartitionField("A", "bucket", 50)]),  # int
    ("NUMERIC_TABLE", [PartitionField("B", "bucket", 50)]),  # long
    ("NUMERIC_TABLE", [PartitionField("E", "bucket", 50)]),  # int
    ("NUMERIC_TABLE", [PartitionField("F", "bucket", 50)]),  # long
    ("STRING_TABLE", [PartitionField("A", "bucket", 50)]),  # string
    ("DT_TSZ_TABLE", [PartitionField("A", "bucket", 50)]),  # date
    ("DT_TSZ_TABLE", [PartitionField("B", "bucket", 50)]),  # datetime (w/ NaTs)
    ("TZ_AWARE_TABLE", [PartitionField("A", "bucket", 50)]),  # timestamp
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("A", "bucket", 4)]),  # w/o nulls
    ("DICT_ENCODED_STRING_TABLE", [PartitionField("A", "bucket", 50)]),  # w/ nulls
    # TODO: Try with another bucket modulus as well?
    (
        "PRIMITIVES_TABLE",
        [PartitionField("A", "months", -1), PartitionField("B", "truncate", 10)],
    ),
    (
        "PRIMITIVES_TABLE",
        [PartitionField("C", "identity", -1), PartitionField("A", "years", -1)],
    ),
    (
        "PRIMITIVES_TABLE",
        [PartitionField("B", "identity", -1), PartitionField("D", "truncate", 1)],
    ),
    (
        "PRIMITIVES_TABLE",
        [PartitionField("D", "bucket", 2), PartitionField("C", "identity", -1)],
    ),
    (
        "PRIMITIVES_TABLE",
        [PartitionField("B", "bucket", 4), PartitionField("A", "bucket", 3)],
    ),
    (
        "PRIMITIVES_TABLE",
        [
            PartitionField("A", "years", -1),
            PartitionField("A", "bucket", 3),
            PartitionField("A", "identity", -1),
        ],
    ),
]


def part_table_name(base_name, part_fields):
    ans = f"part_{base_name}"

    for field in part_fields:
        (
            col,
            trans,
            val,
        ) = field
        val_str = f"_{val}" if trans in ["bucket", "truncate"] else ""
        ans += f"_{col}_{trans}{val_str}"

    return ans


PARTITION_TABLE_NAME_MAP: Dict[str, Tuple[str, List[PartitionField]]] = {
    part_table_name(starter_table_name, part_fields): (starter_table_name, part_fields)
    for starter_table_name, part_fields in PARTITION_MAP
}


def create_table(base_name, part_fields, spark=None):
    if spark is None:
        spark = get_spark()

    assert (
        f"SIMPLE_{base_name}" in TABLE_MAP
    ), f"Didn't find table definition for {base_name}."
    df, sql_schema = TABLE_MAP[f"SIMPLE_{base_name}"]

    create_iceberg_table(
        df,
        sql_schema,
        part_table_name(base_name, part_fields),
        spark,
        part_fields,
    )


def create_partition_tables(tables: List[str], spark=None):
    if spark is None:
        spark = get_spark()

    for table in tables:
        if table in PARTITION_TABLE_NAME_MAP:
            starter_table_name, part_fields = PARTITION_TABLE_NAME_MAP[table]
            create_table(starter_table_name, part_fields, spark)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_partition_tables(list(PARTITION_TABLE_NAME_MAP.keys()))
    else:
        print("Invalid Number of Arguments")
        exit(1)
