"""
Wrapper around fetch_bodo_table_types.py
that accepts a file of fully qualified table paths:
DATABASE_NAME.SCHEMA_NAME.TABLE_NAME
with one path on each line, and repeatedly calls
fetch_bodo_schemas.py to update the schema information.
"""

import argparse
import os
import subprocess
import textwrap


def main(input_file: str, catalog_creds: str):
    with open(input_file) as f:
        for line in f.readlines():
            database_name, schema_name, table_name = line.strip().split(".")
            # Call request_schema.py
            file_path = os.path.dirname(__file__) + "/fetch_bodo_schemas.py"
            cmd = [
                "python",
                file_path,
                "-d",
                database_name,
                "-s",
                schema_name,
                "-t",
                table_name,
                "-c",
                catalog_creds,
            ]
            try:
                subprocess.run(cmd)
            except Exception as e:
                # Print errors but don't halt the process.
                print(
                    f"Error with collecting schema for table: {line}. Error encountered: {str(e)}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fetch_bodo_table_types_wrapper",
        description=textwrap.dedent(
            """
            Wrapper around fetch_bodo_table_types.py to fetch Bodo schema information
            for a file of fully qualified table names, one table per line.
            """
        ),
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file.",
    )
    parser.add_argument(
        "-c",
        "--catalog_creds",
        required=True,
        help="Path to Snowflake credentials file. The following keys must be present: SF_USERNAME, SF_PASSWORD, SF_ACCOUNT, and SF_WAREHOUSE.",
    )

    args = parser.parse_args()
    main(args.input_file, args.catalog_creds)
