from typing import List, Tuple

from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
    get_spark,
)

PARTITION_MAP: List[Tuple[str, List[PartitionField]]] = [
    # Identity for Bools
    ("bool_binary_table", [PartitionField("A", "identity", -1)]),  # bool not null
    ("bool_binary_table", [PartitionField("B", "identity", -1)]),  # bool null
    # Identity
    ("numeric_table", [PartitionField("A", "identity", -1)]),  # int32 no nulls
    ("numeric_table", [PartitionField("B", "identity", -1)]),  # int64 no nulls
    ("numeric_table", [PartitionField("E", "identity", -1)]),  # int32
    ("numeric_table", [PartitionField("F", "identity", -1)]),  # int64
    ("string_table", [PartitionField("B", "identity", -1)]),  # string no nulls
    ("dt_tsz_table", [PartitionField("A", "identity", -1)]),  # date with nulls
    ("dt_tsz_table", [PartitionField("B", "identity", -1)]),  # datetime with NaTs
    ("tz_aware_table", [PartitionField("A", "identity", -1)]),  # datetime no nulls
    ("dict_encoded_string_table", [PartitionField("A", "identity", -1)]),  # w/o nulls
    ("dict_encoded_string_table", [PartitionField("B", "identity", -1)]),  # w/ nulls
    # Date / Time Transformations
    ("dt_tsz_table", [PartitionField("A", "years", -1)]),
    ("dt_tsz_table", [PartitionField("A", "months", -1)]),
    ("dt_tsz_table", [PartitionField("A", "days", -1)]),
    ("dt_tsz_table", [PartitionField("B", "years", -1)]),  # datetime with NaTs
    ("dt_tsz_table", [PartitionField("B", "months", -1)]),  # datetime with NaTs
    ("dt_tsz_table", [PartitionField("B", "days", -1)]),  # datetime with NaTs
    ("dt_tsz_table", [PartitionField("B", "hours", -1)]),  # datetime with NaTs
    ("tz_aware_table", [PartitionField("A", "years", -1)]),  # datetime w/o NaTs
    ("tz_aware_table", [PartitionField("A", "months", -1)]),  # datetime w/o NaTs
    ("tz_aware_table", [PartitionField("A", "days", -1)]),  # datetime w/o NaTs
    ("tz_aware_table", [PartitionField("A", "hours", -1)]),  # datetime w/o NaTs
    # TODO: Include timestampz?
    # Truncate Transformation
    ("numeric_table", [PartitionField("A", "truncate", 3)]),  # int
    ("numeric_table", [PartitionField("B", "truncate", 3)]),  # long
    ("numeric_table", [PartitionField("E", "truncate", 3)]),  # int nulls
    ("numeric_table", [PartitionField("F", "truncate", 3)]),  # long nulls
    ("string_table", [PartitionField("B", "truncate", 1)]),  # string
    ("string_table", [PartitionField("D", "truncate", 2)]),  # string nulls
    ("dict_encoded_string_table", [PartitionField("A", "truncate", 1)]),  # w/o nulls
    ("dict_encoded_string_table", [PartitionField("B", "truncate", 2)]),  # w/ nulls
    # Bucket Transformation
    ("numeric_table", [PartitionField("A", "bucket", 4)]),  # int
    ("numeric_table", [PartitionField("B", "bucket", 4)]),  # long
    ("numeric_table", [PartitionField("E", "bucket", 4)]),  # int
    ("numeric_table", [PartitionField("F", "bucket", 4)]),  # long
    ("string_table", [PartitionField("A", "bucket", 4)]),  # string
    ("dt_tsz_table", [PartitionField("A", "bucket", 4)]),  # date
    ("dt_tsz_table", [PartitionField("B", "bucket", 4)]),  # datetime (w/ NaTs)
    ("tz_aware_table", [PartitionField("A", "bucket", 4)]),  # timestamp
    ("numeric_table", [PartitionField("A", "bucket", 50)]),  # int
    ("numeric_table", [PartitionField("B", "bucket", 50)]),  # long
    ("numeric_table", [PartitionField("E", "bucket", 50)]),  # int
    ("numeric_table", [PartitionField("F", "bucket", 50)]),  # long
    ("string_table", [PartitionField("A", "bucket", 50)]),  # string
    ("dt_tsz_table", [PartitionField("A", "bucket", 50)]),  # date
    ("dt_tsz_table", [PartitionField("B", "bucket", 50)]),  # datetime (w/ NaTs)
    ("tz_aware_table", [PartitionField("A", "bucket", 50)]),  # timestamp
    ("dict_encoded_string_table", [PartitionField("A", "bucket", 4)]),  # w/o nulls
    ("dict_encoded_string_table", [PartitionField("A", "bucket", 50)]),  # w/ nulls
    # TODO: Try with another bucket modulus as well?
    (
        "primitives_table",
        [PartitionField("A", "months", -1), PartitionField("B", "truncate", 10)],
    ),
    (
        "primitives_table",
        [PartitionField("C", "identity", -1), PartitionField("A", "years", -1)],
    ),
    (
        "primitives_table",
        [PartitionField("B", "identity", -1), PartitionField("D", "truncate", 1)],
    ),
    (
        "primitives_table",
        [PartitionField("D", "bucket", 2), PartitionField("C", "identity", -1)],
    ),
    (
        "primitives_table",
        [PartitionField("B", "bucket", 4), PartitionField("A", "bucket", 3)],
    ),
    (
        "primitives_table",
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


def create_table(base_name, part_fields, spark=None):
    if spark is None:
        spark = get_spark()

    assert base_name in TABLE_MAP, f"Didn't find table definition for {base_name}."
    df, sql_schema = TABLE_MAP[base_name]

    create_iceberg_table(
        df,
        sql_schema,
        part_table_name(base_name, part_fields),
        spark,
        part_fields,
    )


def create_all_partition_tables(spark=None):
    if spark is None:
        spark = get_spark()

    for starter_table_name, part_fields in PARTITION_MAP:
        create_table(starter_table_name, part_fields, spark)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_all_partition_tables()
    else:
        print("Invalid Number of Arguments")
        exit(1)
