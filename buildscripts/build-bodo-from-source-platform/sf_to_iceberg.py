# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Helper script to enable converting a snowflake native table to a Snowflake managed
iceberg table, while retaining the original table in Snowflake. This script can
optionally create a zero-copy native clone of the original table to enable better
comparison.
python -u sf_to_iceberg.py -c <CREDS_FILE> -d SNOWFLAKE_SAMPLE_DATA -s TPCDS_SF10TCL -t <TABLE_NAME>
E3 example:
python sf_to_iceberg.py -d E3_PROD -s CLIENT -do E3_PROD -so BODO -t TIER_FOUR -c e3_cred.json -e ICEBERG_STORAGE -r E3_PROD_ADMIN -tc CLIENT_ID WEEKDATE
"""

import argparse
import json
import warnings
from dataclasses import dataclass

import snowflake.connector


@dataclass
class ColumnInfo:
    """
    Dataclass to store the information of a column.
    """

    column_name: str
    type_string: str
    nullable: bool


def default_iceberg_table_name(table_name: str) -> str:
    """
    Generate the default name for the newly created iceberg
    table.
    """
    return f"{table_name}"


def default_native_clone_table_name(table_name: str) -> str:
    """
    Generate the default name for the cloned snowflake table.
    """
    return f"{table_name}"


def get_snowflake_connector(
    username: str,
    password: str,
    account: str,
    warehouse: str,
    database: str,
    role: str,
) -> snowflake.connector.connection:
    """
    Get the snowflake connector.
    """
    return snowflake.connector.connect(
        user=username,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        role=role,
    )


def is_transient_table(
    snowflake_connection: snowflake.connector.connection,
    database: str,
    schema: str,
    table_name: str,
) -> bool:
    """
    Check if a table is transient.
    """
    query = f"SHOW TABLES LIKE '{table_name}' IN SCHEMA {database}.{schema}"
    cursor = snowflake_connection.cursor()
    results = cursor.execute(query).fetchall()
    assert len(results) == 1
    result = results[0]
    kind = result[4].upper()
    return kind == "TRANSIENT"


def clone_native_table(
    snowflake_connection: snowflake.connector.connection,
    database: str,
    schema: str,
    table_name: str,
    output_database: str,
    output_schema: str,
    clone_table_name: str,
):
    """
    Clone the native table.
    """
    if is_transient_table(snowflake_connection, database, schema, table_name):
        # Can't clone transient tables. Need to do a CTAS copy instead.
        query = f"CREATE TABLE {output_database}.{output_schema}.{clone_table_name} AS SELECT * FROM {database}.{schema}.{table_name}"
    else:
        query = f"CREATE TABLE {output_database}.{output_schema}.{clone_table_name} CLONE {database}.{schema}.{table_name}"
    snowflake_connection.cursor().execute(query)


def get_table_column_information(
    snowflake_connection: snowflake.connector.connection,
    database: str,
    schema: str,
    table_name: str,
    columns: list[str] | None,
) -> list[ColumnInfo]:
    """
    Fetch the describe information for a table and format it into a list of
    ColumnInfo objects that can be used to create the iceberg table's types.
    """
    query = f"DESCRIBE TABLE {database}.{schema}.{table_name}"
    cursor = snowflake_connection.cursor()
    cursor.execute(query)
    return [
        ColumnInfo(
            column_name=row[0],
            type_string=row[1].upper(),
            nullable=row[3].upper() == "Y",
        )
        for row in cursor.fetchall()
        if columns is None or row[0] in columns
    ]


def determine_iceberg_type(
    snowflake_type: str, error_on_unsupported_type: bool = True
) -> str | None:
    """
    Convert a snowflake column type to an iceberg column type.
    If we encounter a type that we can't convert then we will either error or
    provide a warning depending on `error_on_unsupported_type`.
    """

    def handle_unsupported_type():
        if error_on_unsupported_type:
            raise ValueError(f"Unsupported snowflake type: {snowflake_type}")
        else:
            msg = f"Dropping unsupported snowflake type: {snowflake_type}"
            warnings.warn(msg)
            return None

    if snowflake_type == "BOOLEAN":
        return "BOOLEAN"
    elif snowflake_type.startswith("NUMBER"):
        parts = snowflake_type.split(",")
        precision = parts[0]
        precision = int(precision[precision.index("(") + 1 :])
        scale = parts[1]
        scale = int(scale[: scale.index(")")])
        if scale != 0:
            msg = f"Converting {snowflake_type} to DOUBLE"
            warnings.warn(msg)
            return "DOUBLE"
        elif precision <= 10:
            return "INTEGER"
        else:
            return "LONG"
    elif snowflake_type == "FLOAT":
        return "DOUBLE"
    elif snowflake_type == "DATE":
        return "DATE"
    elif snowflake_type.startswith("TIME("):
        return "TIME"
    elif snowflake_type.startswith("TIMESTAMP_NTZ"):
        return "TIMESTAMP"
    elif snowflake_type.startswith("TIMESTAMP_LTZ"):
        return "TIMESTAMPTZ"
    elif snowflake_type.startswith("VARCHAR"):
        return "STRING"
    else:
        return handle_unsupported_type()


def convert_column_info_to_iceberg_type(
    column_info: list[ColumnInfo], error_on_unsupported_type: bool = True
) -> list[ColumnInfo]:
    """
    Convert a ColumnInfo object with Snowflake native types
    to a ColumnInfo object with Iceberg types.
    """
    new_columns = []
    for column in column_info:
        iceberg_type = determine_iceberg_type(
            column.type_string, error_on_unsupported_type
        )
        if iceberg_type is not None:
            new_columns.append(
                ColumnInfo(column.column_name, iceberg_type, column.nullable)
            )
    return new_columns


def create_iceberg_table(
    snowflake_connection: snowflake.connector.connection,
    database: str,
    schema: str,
    iceberg_table_name: str,
    column_info: list[ColumnInfo],
    external_volume: str,
):
    """
    Create an iceberg table with the given column information.
    """
    column_strings = [
        f"{column.column_name} {column.type_string} {'NOT NULL' if not column.nullable else ''}"
        for column in column_info
    ]
    column_lines = ",\n".join(column_strings)
    create_table_query = f"""
    CREATE ICEBERG TABLE {database}.{schema}.{iceberg_table_name} (
        {column_lines}
    )
    EXTERNAL_VOLUME = '{external_volume}'
    CATALOG = 'SNOWFLAKE'
    BASE_LOCATION = '{iceberg_table_name}'
    STORAGE_SERIALIZATION_POLICY = 'COMPATIBLE'
    """
    snowflake_connection.cursor().execute(create_table_query)


def populate_iceberg_table(
    snowflake_connection: snowflake.connector.connection,
    output_database: str,
    output_schema: str,
    iceberg_table_name: str,
    input_database: str,
    input_schema: str,
    source_table_name: str,
    column_info: list[ColumnInfo],
):
    """
    Populate the iceberg table with the data from the native table.
    """
    columns = ", ".join(column.column_name for column in column_info)
    query = f"INSERT INTO {output_database}.{output_schema}.{iceberg_table_name} SELECT {columns} FROM {input_database}.{input_schema}.{source_table_name}"
    snowflake_connection.cursor().execute(query)


def main(
    username: str,
    password: str,
    account: str,
    warehouse: str,
    external_volume: str,
    database: str,
    schema: str,
    table_name: str,
    columns: list[str] | None,
    output_database: str,
    output_schema: str,
    iceberg_table_name: str,
    error_on_unsupported_type: bool,
    clone_table_name: str,
    should_clone_native_table: bool,
    role: str,
):
    """
    Create and populate an Iceberg table from the configuration information.
    """
    # Connect to Snowflake
    conn = get_snowflake_connector(
        username, password, account, warehouse, database, role
    )
    # Clone the table if necessary
    if should_clone_native_table:
        clone_native_table(
            conn,
            database,
            schema,
            table_name,
            output_database,
            output_schema,
            clone_table_name,
        )
        read_database = output_database
        read_schema = output_schema
        read_table_name = clone_table_name
    else:
        read_database = database
        read_schema = schema
        read_table_name = table_name
    # Generate the types and convert to iceberg.
    column_info = get_table_column_information(
        conn, read_database, read_schema, read_table_name, columns
    )
    iceberg_column_info = convert_column_info_to_iceberg_type(
        column_info, error_on_unsupported_type
    )
    # Create the iceberg table
    create_iceberg_table(
        conn,
        output_database,
        output_schema,
        iceberg_table_name,
        iceberg_column_info,
        external_volume,
    )
    # Populate the iceberg table
    populate_iceberg_table(
        conn,
        output_database,
        output_schema,
        iceberg_table_name,
        read_database,
        read_schema,
        read_table_name,
        iceberg_column_info,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--credentials-file",
        type=str,
        required=True,
        help="Path to the JSON file containing the Snowflake credentials. We expect this to contain USERNAME, PASSWORD, ACCOUNT, WAREHOUSE",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        required=True,
        help="The database of the source table.",
    )
    parser.add_argument(
        "-s",
        "--schema",
        type=str,
        required=True,
        help="The schema of the source table.",
    )
    parser.add_argument(
        "-do",
        "--output_database",
        type=str,
        required=True,
        help="The database of the output table.",
    )
    parser.add_argument(
        "-so",
        "--output_schema",
        type=str,
        required=True,
        help="The schema of the output table.",
    )
    parser.add_argument(
        "-t",
        "--table-name",
        type=str,
        required=True,
        help="The name of the source table to convert to an iceberg table.",
    )
    parser.add_argument(
        "-e",
        "--external-volume",
        type=str,
        required=True,
        help="The external volume to use for the iceberg table.",
    )
    parser.add_argument(
        "-r",
        "--role",
        type=str,
        required=False,
        default="ACCOUNTADMIN",
        help="The Snowflake role to use for connection (default: ACCOUNTADMIN).",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        default=False,
        help="Whether to clone the source table before converting it to an iceberg table.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Whether to error on unsupported types or just warn.",
    )
    parser.add_argument(
        "-tc",
        "--columns",
        required=False,
        default=None,
        nargs="+",
        help="The columns to write in Iceberg table (default: all columns in input)",
    )
    args = parser.parse_args()
    with open(args.credentials_file) as f:
        creds = json.load(f)
    username = creds["USERNAME"]
    password = creds["PASSWORD"]
    account = creds["ACCOUNT"]
    warehouse = creds["WAREHOUSE"]
    table_name = args.table_name
    iceberg_table_name = default_iceberg_table_name(table_name)
    clone_table_name = default_native_clone_table_name(table_name)
    main(
        username,
        password,
        account,
        warehouse,
        args.external_volume,
        args.database,
        args.schema,
        table_name,
        args.columns,
        args.output_database,
        args.output_schema,
        iceberg_table_name,
        args.strict,
        clone_table_name,
        args.clone,
        args.role,
    )
