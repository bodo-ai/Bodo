# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Utility script for fetching the Bodo type information for a table and writing
the results to the appropriate file. The result will be written to a json file per
schema within a directory for the database. For example, if fetch
MY_DATABASE.MY_SCHEMA.MY_TABLE1

then we would have a file structure of
MY_DATABASE/
    MY_SCHEMA.json -- contains MY_TABLE1
"""

import argparse
import json
import os

import bodo.io.snowflake


def make_conn_str(creds_filename: str) -> str:
    """Create a snowflake connection string from the provided credentials.

    Args:
        creds_filename (str): The filename used to load credentials.

    Returns:
        str: A valid Snowflake connection string.
    """
    with open(creds_filename) as f:
        catalog = json.load(f)

    username = catalog["SF_USERNAME"]
    password = catalog["SF_PASSWORD"]
    account = catalog["SF_ACCOUNT"]
    warehouse = catalog["SF_WAREHOUSE"]
    return f"snowflake://{username}:{password}@{account}?warehouse={warehouse}"


def main(database_name: str, schema_name: str, table_name: str, creds_filename: str):
    base_dir_path = os.path.dirname(__file__)
    # First check if we already have database.
    database_path = f"{base_dir_path}/{database_name}"
    schema_file_path = f"{database_path}/{schema_name}.json"
    # Create the directory if it doesn't exist.
    if not os.path.isdir(database_path):
        os.mkdir(database_path)
    # Check if the file exists.
    if not os.path.isfile(schema_file_path):
        # Create the file.
        with open(schema_file_path, "w") as f:
            f.write(json.dumps({}))
    # Load the file as a json to check for an existing entry. If the
    # entry already exists we will skip the table.
    with open(schema_file_path) as f:
        existing_data = json.load(f)

    if table_name not in existing_data:
        # The table is new so we need to generate the data.
        conn_str = make_conn_str(creds_filename)
        conn = bodo.io.snowflake.snowflake_connect(conn_str)
        (
            df_type,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = bodo.io.snowflake.get_schema(
            conn,
            f"{database_name}.{schema_name}.{table_name}",
            True,  # Query is a select statement
            True,  # Query refers to a single table
            None,  # Do not force any additional dictionary encoding.
            False,  # Do not force Decimal to become Float
        )
        # Convert the DataFrame type to a json.
        new_data_dict = dict(zip(df_type.columns, [str(x) for x in df_type.data]))
        # Add the new table to the existing data and rewrite the file.
        existing_data[table_name] = new_data_dict
        with open(schema_file_path, "w") as f:
            f.write(json.dumps(existing_data))


if __name__ == "__main__":
    # Extract the schema & table names from command line arguments
    parser = argparse.ArgumentParser(
        prog="fetch_bodo_table_types",
        description="Fetches the expected Bodo schema for a given Snowflake table",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        help="which database to add a table schema to (e.g. tpch or etl)",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--schema",
        type=str,
        help="Name of the schema",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--table",
        type=str,
        help="Name of the table",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--catalog_creds",
        required=True,
        help="Path to Snowflake credentials file. The following keys must be present: SF_USERNAME, SF_PASSWORD, SF_ACCOUNT, and SF_WAREHOUSE.",
    )
    args = parser.parse_args()

    main(
        args.database.upper(),
        args.schema.upper(),
        args.table.upper(),
        args.catalog_creds,
    )
