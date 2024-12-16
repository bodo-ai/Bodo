from bodo.tests.iceberg_database_helpers.simple_tables import TABLE_MAP
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    SortField,
    create_iceberg_table,
    get_spark,
)

BASE_NAME = "PRIMITIVES_TABLE"
PARTITION_SPEC = [
    PartitionField("A", "years", -1),
    PartitionField("C", "identity", -1),
]

SORT_ORDER = [
    SortField("B", "bucket", 5, True, True),
    SortField("D", "truncate", 1, False, True),
    SortField("A", "identity", -1, False, False),
]

TABLE_NAME = f"PARTSORT_{BASE_NAME}"


def create_table(table_name=TABLE_NAME, spark=None):
    if spark is None:
        spark = get_spark()

    assert (
        f"SIMPLE_{BASE_NAME}" in TABLE_MAP
    ), f"Didn't find table definition for {BASE_NAME}."
    df, sql_schema = TABLE_MAP[f"SIMPLE_{BASE_NAME}"]
    create_iceberg_table(df, sql_schema, table_name, spark, PARTITION_SPEC, SORT_ORDER)


if __name__ == "__main__":
    create_table()
