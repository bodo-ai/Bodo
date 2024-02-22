from bodo.tests.iceberg_database_helpers.partition_tables import (
    PARTITION_TABLE_NAME_MAP,
    create_partition_tables,
)
from bodo.tests.iceberg_database_helpers.schema_evolution_tables import (
    COLUMN_DROP_TABLES_MAP,
    gen_column_reorder_tables,
    gen_nullable_tables,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLE_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    append_to_iceberg_table,
    get_spark,
    transform_str,
)

NULLABLE_TABLES_MAP = gen_nullable_tables(list(PARTITION_TABLE_NAME_MAP.keys()))


def gen_promotion_tables(input_tables: list[str]):
    tables = {}
    for table in input_tables:
        tables[f"{table}_PROMOTION"] = table
    return tables


PROMOTION_TABLES_MAP: dict[str, str] = gen_promotion_tables(PARTITION_TABLE_NAME_MAP)

# It seems we can't drop a column that was ever a partition field in spark, not sure what the spec says
# COLUMN_DROP_TABLES_MAP

# It seems like we can't rename a column that was ever a partition in spark, not sure what the spec says
# COLUMN_RENAME_TABLES_MAP

COLUMN_REORDER_TABLES_MAP: dict[str, str] = gen_column_reorder_tables(
    PARTITION_TABLE_NAME_MAP
)
STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP: dict[str, str] = {
    "part_STRUCT_TABLE_A_a_identity_STRUCT_FIELD_TYPE_PROMOTION": "part_STRUCT_TABLE_A_a_identity"
}

STRUCT_FIELD_NULLABLE_TABLES_MAP: dict[str, str] = {
    "part_STRUCT_TABLE_A_a_identity_STRUCT_FIELD_NULLABLE": "part_STRUCT_TABLE_A_a_identity"
}


def gen_change_part_column_tables(input_tables: list[str]):
    tables = {}
    for table in input_tables:
        # Skip tables that partition by year when the base table is PRIMITIVES since their is no other timestamp column to partition by
        if "PRIMITIVES" in table and any(
            [
                part.col_name == "A"
                and part.transform in ("months", "years", "days", "hours")
                for part in PARTITION_TABLE_NAME_MAP[table][1]
            ]
        ):
            continue
        b_part = [
            part for part in PARTITION_TABLE_NAME_MAP[table][1] if part.col_name == "B"
        ]
        if (
            any([part.transform == "hours" for part in b_part])
            and SIMPLE_TABLE_MAP[f"SIMPLE_{PARTITION_TABLE_NAME_MAP[table][0]}"][1][0][
                1
            ]
            == "date"
        ):
            continue
        tables[f"{table}_CHANGE_PART_COLUMN"] = table
    return tables


CHANGE_PART_COLUMN_TABLES_MAP: dict[str, str] = gen_change_part_column_tables(
    list(PARTITION_TABLE_NAME_MAP.keys())
)

PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP: dict[str, str] = (
    NULLABLE_TABLES_MAP
    | PROMOTION_TABLES_MAP
    | COLUMN_DROP_TABLES_MAP
    | COLUMN_REORDER_TABLES_MAP
    | STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP
    | STRUCT_FIELD_NULLABLE_TABLES_MAP
    | CHANGE_PART_COLUMN_TABLES_MAP
)


def create_nullable_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table
    but with all columns that are partition columns set to nullable.
    """
    if spark is None:
        spark = get_spark()
    base_name = NULLABLE_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    partition_columns = [
        part.col_name for part in PARTITION_TABLE_NAME_MAP[base_name][1]
    ]
    postfix = f"_NULLABLE{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        for column in partition_columns:
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} DROP NOT NULL"
            )
        sql_schema = [
            (name, type, True if name in partition_columns else nullable)
            for (name, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)


def create_promotion_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table but
    with a partition column that has been promoted to a larger type.
    """
    if spark is None:
        spark = get_spark()
    base_name = PROMOTION_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    partition_columns = [
        part.col_name for part in PARTITION_TABLE_NAME_MAP[base_name][1]
    ]
    postfix = f"_PROMOTION{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        for column in partition_columns:
            if column == "A" and "NUMERIC" in table:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} TYPE BIGINT"
                )
                sql_schema = [
                    (name, "bigint" if name.lower() == "a" else type, nullable)
                    for (name, type, nullable) in sql_schema
                ]
                append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)
            elif column == "A" and "DECIMAL" in table:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} TYPE DECIMAL(38, 5)"
                )
                sql_schema = [
                    (name, "decimal(38,5)" if name.lower() == "a" else type, nullable)
                    for (name, type, nullable) in sql_schema
                ]
                append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)
            elif column == "C" and "NUMERIC" in table:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} TYPE DOUBLE"
                )
                sql_schema = [
                    (name, "double" if name.lower() == "c" else type, nullable)
                    for (name, type, nullable) in sql_schema
                ]
                append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)


def create_column_reorder_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table but
    with the columns in reverse order.
    """
    if spark is None:
        spark = get_spark()
    base_name = COLUMN_REORDER_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    postfix = f"_REORDER_COLUMN{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        prev_column = None
        for i, column in enumerate(reversed(df.columns)):
            if i == 0:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} FIRST"
                )
            else:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN {column} AFTER {prev_column}"
                )
            prev_column = column
        df = df[(reversed(df.columns))]
        sql_schema = list(reversed(sql_schema))
        append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)


def create_struct_field_type_promotion_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table which is partitioned by a struct's field but
    with a field in a struct column that has been promoted to a larger type.
    """
    if spark is None:
        spark = get_spark()
    base_name = STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    postfix = f"_STRUCT_FIELD_TYPE_PROMOTION{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN A.a TYPE BIGINT"
        )
        sql_schema = [
            (
                name,
                type.replace("int", "bigint") if name.lower() == "a" else type,
                nullable,
            )
            for (name, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)


def create_struct_field_nullable_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table which is partitioned by a struct's field but
    with a field in a struct column field that has been set to nullable.
    """
    if spark is None:
        spark = get_spark()
    base_name = STRUCT_FIELD_NULLABLE_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    postfix = f"_STRUCT_FIELD_NULLABLE{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} ALTER COLUMN A.a DROP NOT NULL"
        )
        append_to_iceberg_table(df, sql_schema, f"{base_name}{postfix}", spark)


def create_change_part_column_table(table: str, spark=None, postfix=""):
    """
    Create a table with the same schema and data as the base partition table but with a different column as a partition column.
    """
    if spark is None:
        spark = get_spark()
    base_name = CHANGE_PART_COLUMN_TABLES_MAP[table]
    assert (
        base_name in PARTITION_TABLE_NAME_MAP
    ), f"Didn't find table definition for {base_name}."
    postfix = f"_CHANGE_PART_COLUMN{postfix}"
    df, sql_schema = SIMPLE_TABLE_MAP[
        f"SIMPLE_{PARTITION_TABLE_NAME_MAP[base_name][0]}"
    ]
    df = df.copy()
    sql_schema = list(sql_schema)
    if base_name in create_partition_tables([base_name], spark, postfix):
        if len(df.columns) > 1:
            partition = PARTITION_TABLE_NAME_MAP[base_name][1][0]
            if partition.col_name == "A.a":
                new_part_col = "A.b"
            else:
                new_part_col = "A" if partition.col_name == "B" else "B"
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{base_name}{postfix} REPLACE PARTITION FIELD {transform_str(partition.col_name, partition.transform, partition.transform_val)} WITH {transform_str(new_part_col, partition.transform, partition.transform_val)}"
            )


def create_partition_schema_evolution_tables(
    tables: list[str], spark=None, postfix: str = ""
):
    if spark is None:
        spark = get_spark()
    for table in tables:
        if table in NULLABLE_TABLES_MAP:
            create_nullable_table(table, spark, postfix=postfix)
        elif table in PROMOTION_TABLES_MAP:
            create_promotion_table(table, spark, postfix=postfix)
        elif table in COLUMN_REORDER_TABLES_MAP:
            create_column_reorder_table(table, spark, postfix=postfix)
        elif table in STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP:
            create_struct_field_type_promotion_table(table, spark, postfix=postfix)
        elif table in STRUCT_FIELD_NULLABLE_TABLES_MAP:
            create_struct_field_nullable_table(table, spark, postfix=postfix)
        elif table in CHANGE_PART_COLUMN_TABLES_MAP:
            create_change_part_column_table(table, spark, postfix=postfix)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_partition_schema_evolution_tables(
            list(PARTITION_SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys())
        )
    else:
        print("Invalid Number of Arguments")
        exit(1)
