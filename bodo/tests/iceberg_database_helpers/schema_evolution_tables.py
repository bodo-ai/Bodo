import re
from copy import deepcopy
from itertools import chain, combinations

import pandas as pd
import pyarrow as pa

from bodo.tests.iceberg_database_helpers.simple_tables import BASE_MAP
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLE_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    append_to_iceberg_table,
    create_iceberg_table,
    get_spark,
)


def gen_nullable_tables(input_tables: list[str]) -> dict[str, str]:
    tables = {}
    for table in input_tables:
        tables[f"{table}_NULLABLE"] = table
    return tables


NULLABLE_TABLES_MAP: dict[str, str] = gen_nullable_tables(BASE_MAP)

PROMOTION_TABLES_MAP: dict[str, str] = {
    "INT_TO_BIGINT_PROMOTION_TABLE": "NUMERIC_TABLE",
    "FLOAT_TO_DOUBLE_PROMOTION_TABLE": "NUMERIC_TABLE",
    "DECIMALS_PRECISION_PROMOTION_TABLE": "DECIMALS_TABLE",
}


def gen_column_add_tables(input_tables: list[str]) -> dict[str, str]:
    tables = {}
    for table in input_tables:
        tables[f"{table}_ADD_COLUMN"] = table
    return tables


COLUMN_ADD_TABLES_MAP: dict[str, str] = gen_column_add_tables(BASE_MAP)


def gen_column_drop_tables(input_tables: list[str]) -> dict[str, str]:
    tables = {}
    for table in input_tables:
        tables[f"{table}_DROP_COLUMN"] = table
    return tables


COLUMN_DROP_TABLES_MAP: dict[str, str] = gen_column_drop_tables(BASE_MAP)


def gen_column_rename_tables(input_tables: list[str]) -> dict[str, str]:
    tables = {}
    for table in input_tables:
        tables[f"{table}_RENAME_COLUMN"] = table
    return tables


COLUMN_RENAME_TABLES_MAP: dict[str, str] = gen_column_rename_tables(BASE_MAP)


def gen_column_reorder_tables(input_tables: list[str]) -> dict[str, str]:
    tables = {}
    for table in input_tables:
        tables[f"{table}_REORDER_COLUMN"] = table
    return tables


COLUMN_REORDER_TABLES_MAP: dict[str, str] = gen_column_reorder_tables(BASE_MAP)

STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP: dict[str, str] = {
    "STRUCT_FIELD_TYPE_PROMOTION": "STRUCT_TABLE"
}

STRUCT_FIELD_NULLABLE_TABLES_MAP: dict[str, str] = {
    "STRUCT_FIELD_NULLABLE": "STRUCT_TABLE"
}

STRUCT_FIELD_EVOLUTIONS_TABLES_MAP: dict[str, str] = {
    "STRUCT_FIELD_DROP": "STRUCT_TABLE",
    "STRUCT_FIELD_RENAME": "STRUCT_TABLE",
    "STRUCT_FIELD_REORDER": "STRUCT_TABLE",
    "STRUCT_FIELD_ADD": "STRUCT_TABLE",
    "STRUCT_ADD_STRUCT_LIST_MAP_FIELDS": "STRUCT_TABLE",
}

MAP_FIELDS_EVOLUTION_TABLES_MAP: dict[str, str] = {
    "MAP_VALUE_PROMOTE": "MAP_TABLE",
    # Spark doesn't seem to allow updating the key type!
    # In Spark there's no way to create a non-nullable
    # value type, so there's no way to go from non-nullable
    # to nullable.
}

LIST_ELEMENT_EVOLUTION_TABLES_MAP: dict[str, str] = {
    "LIST_VALUES_PROMOTE": "LIST_TABLE",
    # In Spark, there's no way to create a non-nullable
    # element type, so there's no way to go from non-nullable
    # to nullable.
}


def gen_combo_tables(input_table: str) -> dict[str, str]:
    funcs = ["PROMOTE", "RENAME", "NULLABLE", "REORDER", "ADD", "DROP"]
    # Possible combinations of functions where drop isn't an operation or is the last operation
    combos = list(
        chain.from_iterable(combinations(funcs, r) for r in range(1, len(funcs) + 1))
    )
    tables = {
        f"{'_'.join(combo)}_TABLE": input_table
        for combo in combos
        if "DROP" not in combo[:-1]
    }
    return tables


COMBO_TABLES_MAP: dict[str, str] = gen_combo_tables("NUMERIC_TABLE")

ADVERSARIAL_TABLES_MAP: dict[str, str] = {
    "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE": "NUMERIC_TABLE"
}

SCHEMA_EVOLUTION_TABLE_NAME_MAP: dict[str, str] = (
    NULLABLE_TABLES_MAP
    | PROMOTION_TABLES_MAP
    | COLUMN_ADD_TABLES_MAP
    | COLUMN_DROP_TABLES_MAP
    | COLUMN_RENAME_TABLES_MAP
    | COLUMN_REORDER_TABLES_MAP
    | STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP
    | STRUCT_FIELD_NULLABLE_TABLES_MAP
    | STRUCT_FIELD_EVOLUTIONS_TABLES_MAP
    | MAP_FIELDS_EVOLUTION_TABLES_MAP
    | LIST_ELEMENT_EVOLUTION_TABLES_MAP
    | COMBO_TABLES_MAP
    | ADVERSARIAL_TABLES_MAP
)


def create_nullable_table(table: str, spark=None, postfix: str = ""):
    """
    Create a nullable table by removing the NOT NULL constraint from the columns. Postfix is added to the table name.
    """
    if spark is None:
        spark = get_spark()
    base_name = NULLABLE_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        for column in df.columns:
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} DROP NOT NULL"
            )

        sql_schema = [(column, type, True) for (column, type, _) in sql_schema]
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_promotion_table(table: str, spark=None, postfix: str = ""):
    """
    Create a table promoting the column types to a higher type e.g. INT to BIGINT. Postfix is added to the table name.
    """
    if spark is None:
        spark = get_spark()
    base_name = PROMOTION_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        for i, (column, type, _) in enumerate(sql_schema):
            if "INT" == type.upper() and "INT" in table:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} TYPE BIGINT"
                )
                sql_schema[i] = (column, "BIGINT", sql_schema[i][2])
            if "FLOAT" == type.upper() and "FLOAT" in table:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} TYPE DOUBLE"
                )
                sql_schema[i] = (column, "DOUBLE", sql_schema[i][2])
            if "DECIMAL" in type.upper() and "DECIMAL" in table:
                precision = int(re.search(r"\((\d+),", type).group(1))
                scale = int(re.search(r",(\d+)\)", type).group(1))
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} TYPE DECIMAL({min(38, precision + 1)}, {scale})"
                )
                sql_schema[i] = (
                    column,
                    f"DECIMAL({min(38, precision + 1)}, {scale})",
                    sql_schema[i][2],
                )
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_column_add_table(table: str, spark=None, postfix: str = ""):
    """Create a table with added new columns. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = COLUMN_ADD_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMNS (new_column_one INT, new_column_two STRING)"
        )
        df["new_column_one"] = pd.Series([1] * len(df), dtype=pd.ArrowDtype(pa.int32()))
        df["new_column_two"] = pd.Series(
            ["new_column_two" * len(df)], dtype=pd.ArrowDtype(pa.string())
        )
        sql_schema = sql_schema + [
            ("new_column_one", "INT", True),
            ("new_column_two", "STRING", True),
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_column_drop_table(table: str, spark=None, postfix: str = ""):
    """Create a table with a column dropped. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = COLUMN_DROP_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        drop_column = df.columns[0]
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} DROP COLUMN {drop_column}"
        )
        df = df.drop(columns=[drop_column])
        sql_schema = [x for x in sql_schema if x[0] != drop_column]
        if len(sql_schema) != 0:
            append_to_iceberg_table(df, sql_schema, table, spark)


def create_column_rename_table(table: str, spark=None, postfix: str = ""):
    """Create a table with all columns renamed. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = COLUMN_RENAME_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        for column in df.columns:
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN {column} TO {column}_renamed"
            )
        df = df.rename(columns={column: f"{column}_renamed" for column in df.columns})
        sql_schema = [
            (f"{column}_renamed", type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_column_reorder_table(table: str, spark=None, postfix: str = ""):
    """Create a table with columns reordered in reverse. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = COLUMN_REORDER_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        prev_column = None
        for i, column in enumerate(reversed(df.columns)):
            if i == 0:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} FIRST"
                )
            else:
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {column} AFTER {prev_column}"
                )
            prev_column = column
        df = df[reversed(df.columns)]
        sql_schema = list(reversed(sql_schema))
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_struct_field_type_promotion_table(table: str, spark=None, postfix: str = ""):
    """Create a table with a struct field type promoted to a higher type. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A.a TYPE BIGINT"
        )
        sql_schema[0] = ("A", "STRUCT<a: bigint, b: long>", True)
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_struct_field_nullable_table(table: str, spark=None, postfix: str = ""):
    """Create a table with struct fields made nullable. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = STRUCT_FIELD_NULLABLE_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A.a DROP NOT NULL"
        )
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A.b DROP NOT NULL"
        )
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN B.b DROP NOT NULL"
        )
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_struct_field_evolution_table(table: str, spark=None, postfix: str = ""):
    """
    Create a table with evolution operations performed on
    the fields of a struct (field drop, rename, addition,
    reorder, etc.).
    Postfix is added to the table name.
    """
    if spark is None:
        spark = get_spark()
    base_name = STRUCT_FIELD_EVOLUTIONS_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        if table == "STRUCT_FIELD_DROP":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} DROP COLUMN A.a"
            )
            df["A"] = df["A"].apply(lambda x: {"b": x["b"]})
            sql_schema[0] = ("A", "STRUCT<b: long>", True)
            append_to_iceberg_table(df, sql_schema, table, spark)

        elif table == "STRUCT_FIELD_RENAME":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN A.b TO b_renamed"
            )

            def rename_field(x):
                x["b_renamed"] = x["b"]
                del x["b"]
                return x

            df["A"] = df["A"].apply(rename_field)
            sql_schema[0] = ("A", "STRUCT<a: int, b_renamed: long>", True)
            append_to_iceberg_table(df, sql_schema, table, spark)

        elif table == "STRUCT_FIELD_ADD":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMNS (A.new_column_one INT, A.new_column_two STRING)"
            )
            df["A"] = df["A"].apply(
                lambda x: {**x, "new_column_one": 1, "new_column_two": "new_column_two"}
            )
            sql_schema[0] = (
                "A",
                "STRUCT<a: int, b: long, new_column_one: int, new_column_two: string>",
                True,
            )
            append_to_iceberg_table(df, sql_schema, table, spark)

        elif table == "STRUCT_FIELD_REORDER":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A.b FIRST"
            )
            df["A"] = df["A"].apply(
                lambda x: {
                    **{"b": x["b"]},
                    **{key: value for key, value in x.items() if key != "b"},
                }
            )
            sql_schema[0] = ("A", "STRUCT<b: long, a: int>", True)
            append_to_iceberg_table(df, sql_schema, table, spark)
        elif table == "STRUCT_ADD_STRUCT_LIST_MAP_FIELDS":
            spark.sql(
                f"""
                      ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMNS 
                      (
                        B.d struct<t: array<string>, up: map<int, string>>,
                        A.pld array<string>,
                        A.iwi map<string, int>
                      )
            """
            )
            sql_schema[0] = (
                "A",
                "STRUCT<a: int, b: long, pld: array<string>, iwi: map<string, int>>",
                True,
            )
            sql_schema[1] = (
                "B",
                "STRUCT<a: double, b: long, c: double, d: struct<t: array<string>, up: map<int, string>>>",
                True,
            )
            df = pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            {
                                "a": 100,
                                "b": 300,
                                "pld": ["abc", "def", "ghi"],
                                "iwi": {"abc": 300, "pqu": 100},
                            },
                            {
                                "a": 2,
                                "b": 666,
                                "pld": ["def", "oip", "abc"],
                                "iwi": {"def": 666, "oip": 2},
                            },
                        ]
                        * 10,
                        dtype=object,
                        # Using 'object' since PySpark can have issues with
                        # converting some of these types. Including the exact
                        # type in comments for reference.
                        # dtype=pd.ArrowDtype(
                        #     pa.struct(
                        #         [
                        #             ("a", pa.int32()),
                        #             ("b", pa.int64()),
                        #             ("pld", pa.large_list(pa.string())),
                        #             ("iwi", pa.map_(pa.string(), pa.int32())),
                        #         ]
                        #     )
                        # ),
                    ),
                    "B": pd.Series(
                        [
                            {
                                "a": 2.0,
                                "b": 5,
                                "c": 78.23,
                                "d": {
                                    "t": ["pizza", "pie"],
                                    "up": {23: "aty", 90: "234 hi9"},
                                },
                            },
                            {
                                "a": 1.98,
                                "b": 45,
                                "c": 12.90,
                                "d": {
                                    "t": ["pizza", "crust"],
                                    "up": {203: "aty", 980: "234 hi9"},
                                },
                            },
                        ]
                        * 10,
                        dtype=object,
                        # dtype=pd.ArrowDtype(
                        #     pa.struct(
                        #         [
                        #             ("a", pa.float64()),
                        #             ("b", pa.int64()),
                        #             ("c", pa.float64()),
                        #             (
                        #                 "d",
                        #                 pa.struct(
                        #                     [
                        #                         ("t", pa.large_list(pa.string())),
                        #                         (
                        #                             "up",
                        #                             pa.map_(pa.string(), pa.string()),
                        #                         ),
                        #                     ]
                        #                 ),
                        #             ),
                        #         ]
                        #     )
                        # ),
                    ),
                }
            )
            append_to_iceberg_table(df, sql_schema, table, spark)


def create_combo_table(table: str, spark=None, postfix: str = ""):
    """Create a table with a combination of operations (PROMOTE, RENAME, NULLABLE, REORDER, ADD, DROP). The combination to use is parsed from the table name Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = COMBO_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        ops = table.split("_")[:-1]
        col_name = "A"
        for op in ops:
            if op == "PROMOTE":
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {col_name} TYPE BIGINT"
                )
                append_to_iceberg_table(df, sql_schema, table, spark)
            elif op == "RENAME":
                prev_col_name = col_name
                col_name = f"{prev_col_name}_renamed"
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN {prev_col_name} TO {col_name}"
                )
                df = df.rename(columns={prev_col_name: col_name})
                sql_schema = [
                    (column.replace(prev_col_name, col_name), type, nullable)
                    for (column, type, nullable) in sql_schema
                ]
                append_to_iceberg_table(df, sql_schema, table, spark)
            elif op == "NULLABLE":
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {col_name} DROP NOT NULL"
                )
                sql_schema = [
                    (column, type, True if column == col_name else nullable)
                    for (column, type, nullable) in sql_schema
                ]
                append_to_iceberg_table(df, sql_schema, table, spark)
            elif op == "REORDER":
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN {col_name} AFTER B"
                )
                df = df[["B"] + [col for col in df.columns if col != "B"]]
                sql_schema = [
                    (column, type, nullable)
                    for (column, type, nullable) in sql_schema
                    if column == "B"
                ] + [
                    (column, type, nullable)
                    for (column, type, nullable) in sql_schema
                    if column != "B"
                ]
                append_to_iceberg_table(df, sql_schema, table, spark)

            elif op == "ADD":
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMNS (new_column_one INT, new_column_two STRING)"
                )
                df["new_column_one"] = 1
                df["new_column_two"] = "new_column_two"
                sql_schema = sql_schema + [
                    ("new_column_one", "INT", True),
                    ("new_column_two", "STRING", True),
                ]
                append_to_iceberg_table(df, sql_schema, table, spark)
            elif op == "DROP":
                spark.sql(
                    f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} DROP COLUMN {col_name}"
                )
                df = df.drop(columns=[col_name])
                sql_schema = [
                    (column, type, nullable)
                    for (column, type, nullable) in sql_schema
                    if column != col_name
                ]
                append_to_iceberg_table(df, sql_schema, table, spark)


def create_adversarial_table(table: str, spark=None, postfix: str = ""):
    """Create a table with an adversarial schema evolution. Postfix is added to the table name."""
    if spark is None:
        spark = get_spark()
    base_name = ADVERSARIAL_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        spark.sql(f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMN G INT")
        df["G"] = 1
        sql_schema.append(("G", "INT", True))
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Rename column C to TY
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN C TO TY"
        )
        df = df.rename(columns={"C": "TY"})
        sql_schema = [
            (column.replace("C", "TY"), type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Rename B to C
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN B TO C"
        )
        df = df.rename(columns={"B": "C"})
        sql_schema = [
            (column.replace("B", "C"), type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Drop D
        spark.sql(f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} DROP COLUMN D")
        df = df.drop(columns=["D"])
        sql_schema = [
            (column, type, nullable)
            for (column, type, nullable) in sql_schema
            if column != "D"
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Promote E to BIGINT
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN E TYPE BIGINT"
        )
        df["E"] = df["E"].astype("Int64")
        sql_schema = [
            (column, "BIGINT" if column == "E" else type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Put ty after A
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN TY AFTER A"
        )
        df = df[["A", "TY"] + [col for col in df.columns if col not in ["A", "TY"]]]
        sql_schema = [
            (column, type, nullable)
            for (column, type, nullable) in sql_schema
            if column in ["A", "TY"]
        ] + [
            (column, type, nullable)
            for (column, type, nullable) in sql_schema
            if column not in ["A", "TY"]
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # rename ty to b
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN TY TO B"
        )
        df = df.rename(columns={"TY": "B"})
        sql_schema = [
            (column.replace("TY", "B"), type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)

        # Rename E to TY
        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} RENAME COLUMN E TO TY"
        )
        df = df.rename(columns={"E": "TY"})
        sql_schema = [
            (column.replace("E", "TY"), type, nullable)
            for (column, type, nullable) in sql_schema
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)
        # Drop A
        spark.sql(f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} DROP COLUMN A")
        col_a = df["A"]
        df = df.drop(columns=["A"])
        sql_schema = [
            (column, type, nullable)
            for (column, type, nullable) in sql_schema
            if column != "A"
        ]
        append_to_iceberg_table(df, sql_schema, table, spark)
        # Add A and make it first
        spark.sql(f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ADD COLUMN A INT")
        df["A"] = col_a
        sql_schema.append(("A", "INT", True))

        spark.sql(
            f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A FIRST"
        )
        append_to_iceberg_table(df, sql_schema, table, spark)


def create_map_fields_evolution_table(table: str, spark=None, postfix: str = ""):
    """
    Create a table with evolution operations performed on the value
    field of a map. Only promoting the value type is supported at this time.
    Postfix is added to the table name.
    """
    if spark is None:
        spark = get_spark()
    base_name = MAP_FIELDS_EVOLUTION_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        if table == "MAP_VALUE_PROMOTE":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN A.value TYPE LONG"
            )
            sql_schema[0] = ("A", "MAP<STRING, LONG>", True)
            append_to_iceberg_table(df, sql_schema, table, spark)


def create_list_element_evolution_table(table: str, spark=None, postfix: str = ""):
    """
    Create a table with evolution operations performed on the element
    field of a list. Only promoting the element type is supported at this time.
    Postfix is added to the table name.
    """
    if spark is None:
        spark = get_spark()
    base_name = LIST_ELEMENT_EVOLUTION_TABLES_MAP[table]
    assert f"SIMPLE_{base_name}" in SIMPLE_TABLE_MAP, (
        f"Didn't find table definition for {base_name}."
    )
    df, sql_schema = SIMPLE_TABLE_MAP[f"SIMPLE_{base_name}"]
    df = deepcopy(df)
    sql_schema = sql_schema.copy()
    table = f"{table}{postfix}"
    if create_iceberg_table(df, sql_schema, table, spark) is not None:
        if table == "LIST_VALUES_PROMOTE":
            spark.sql(
                f"ALTER TABLE hadoop_prod.{DATABASE_NAME}.{table} ALTER COLUMN C.element TYPE BIGINT"
            )
            sql_schema[2] = ("C", "ARRAY<BIGINT>", True)
            append_to_iceberg_table(df, sql_schema, table, spark)


def create_schema_evolution_tables(tables: list[str], spark=None, postfix: str = ""):
    if spark is None:
        spark = get_spark()
    for table in tables:
        if table in NULLABLE_TABLES_MAP:
            create_nullable_table(table, spark, postfix=postfix)
        elif table in PROMOTION_TABLES_MAP:
            create_promotion_table(table, spark, postfix=postfix)
        elif table in COLUMN_ADD_TABLES_MAP:
            create_column_add_table(table, spark, postfix=postfix)
        elif table in COLUMN_DROP_TABLES_MAP:
            create_column_drop_table(table, spark, postfix=postfix)
        elif table in COLUMN_RENAME_TABLES_MAP:
            create_column_rename_table(table, spark, postfix=postfix)
        elif table in COLUMN_REORDER_TABLES_MAP:
            create_column_reorder_table(table, spark, postfix=postfix)
        elif table in STRUCT_FIELD_TYPE_PROMOTION_TABLES_MAP:
            create_struct_field_type_promotion_table(table, spark, postfix=postfix)
        elif table in STRUCT_FIELD_NULLABLE_TABLES_MAP:
            create_struct_field_nullable_table(table, spark, postfix=postfix)
        elif table in STRUCT_FIELD_EVOLUTIONS_TABLES_MAP:
            create_struct_field_evolution_table(table, spark, postfix=postfix)
        elif table in COMBO_TABLES_MAP:
            create_combo_table(table, spark, postfix=postfix)
        elif table in ADVERSARIAL_TABLES_MAP:
            create_adversarial_table(table, spark, postfix=postfix)
        elif table in MAP_FIELDS_EVOLUTION_TABLES_MAP:
            create_map_fields_evolution_table(table, spark, postfix=postfix)
        elif table in LIST_ELEMENT_EVOLUTION_TABLES_MAP:
            create_list_element_evolution_table(table, spark, postfix=postfix)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        create_schema_evolution_tables(list(SCHEMA_EVOLUTION_TABLE_NAME_MAP.keys()))
    else:
        print("Invalid Number of Arguments")
        exit(1)
