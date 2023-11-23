# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Wrapper script for BodoSQLFetchTables.py. This script is used to run BodoSQLFetchTables.py
and then parse the logs to get additional information about the tables used by BodoSQL.
"""

import argparse
import os
import re
import subprocess
from typing import List, Set


def main(
    dir_name: str,
    cred_filename: str,
    tables_file: str,
    success_file: str,
    fail_file: str,
    starting_offset: int,
    batch_size: int,
    append: bool,
):
    is_first_batch = True
    # Fetch the directory list to determine the maximum number of files.
    filenames = os.listdir(dir_name)
    max_num_files = len(filenames)
    batch_size = max_num_files if batch_size < 0 else batch_size
    for i in range(starting_offset, max_num_files, batch_size):
        print(f"Starting to process queries at offset {i}")
        # Call a subprocess to BodoSQLFetchTables.py and forward the
        # existing arguments.
        file_path = os.path.dirname(__file__) + "/BodoSQLFetchTables.py"
        cmd = [
            "python",
            file_path,
            "-d",
            dir_name,
            "-c",
            cred_filename,
            "-o",
            str(i),
            "-b",
            str(batch_size),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        # Define relevant outputs
        used_tables: List[str] = []
        successful_files: List[str] = []
        failed_files: List[str] = []
        # Define our matchers
        table_matcher = re.compile(".* Validating table: (.*)")
        success_matcher = re.compile("Successfully validated query file: (.*)")
        failed_matcher = re.compile("Failed to validate query file: (.*)")
        # Capture the outputs
        out_messages = ret.stdout
        err_messages = ret.stderr
        outputs = [out_messages, err_messages]
        for output in outputs:
            # Parse each line in the output
            for line in output.split("\n"):
                if table_matcher.match(line):
                    # Extract the table name from the line
                    table_name = table_matcher.match(line).group(1)
                    # Add the table name to the set of used tables
                    used_tables.append(table_name)
                elif success_matcher.match(line):
                    # Extract the filename from the line
                    filename = success_matcher.match(line).group(1)
                    # Add the filename to the list of successful files
                    successful_files.append(filename)
                elif failed_matcher.match(line):
                    # Extract the filename from the line
                    filename = failed_matcher.match(line).group(1)
                    # Add the filename to the list of failed files
                    failed_files.append(filename)

        # Sort the results
        used_tables = sorted(used_tables)
        successful_files = sorted(successful_files)
        failed_files = sorted(failed_files)
        file_permission = "a" if append or not is_first_batch else "w"
        # Write the results
        with open(tables_file, file_permission) as f:
            for table in used_tables:
                f.write(f"{table}\n")
        with open(success_file, file_permission) as f:
            for success in successful_files:
                f.write(f"{success}\n")
        with open(fail_file, file_permission) as f:
            for fail in failed_files:
                f.write(f"{fail}\n")
        print(f"Finished processing queries at offset {i}")
        is_first_batch = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BodoSQLFetchTablesWrapper",
        description="Wrapper script around BodoSQLFetchTables.py to process log outputs.",
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="Path to parquet file containing data.",
    )
    parser.add_argument(
        "-c",
        "--catalog_creds",
        required=True,
        help="Path to Snowflake credentials file. The following keys must be present: SF_USERNAME, SF_PASSWORD and SF_ACCOUNT. The following keys are optional: SF_WAREHOUSE, SF_DATABASE",
    )
    parser.add_argument(
        "-t",
        "--tables_file",
        required=True,
        help="Path to write the list of tables. 1 entry per line.",
    )
    parser.add_argument(
        "-success",
        "--success_file",
        required=True,
        help="Path to write the successful files list. 1 entry per line.",
    )
    parser.add_argument(
        "-fail",
        "--fail_file",
        required=True,
        help="Path to write the failed files list. 1 entry per line.",
    )
    parser.add_argument(
        "-o",
        "--offset",
        required=False,
        type=int,
        default=0,
        help="Offset into the sorted directory file list to start fetching queries. This is to enable mini-batching in case a query hangs.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=-1,
        help="Maximum number of queries to process in 1 batch.",
    )
    parser.add_argument(
        "-a",
        "--append",
        required=False,
        default=False,
        action="store_true",
        help="For all of the given files should we append to the file or replace the contents. This can be used to restart batches.",
    )

    args = parser.parse_args()
    main(
        args.dir,
        args.catalog_creds,
        args.tables_file,
        args.success_file,
        args.fail_file,
        args.offset,
        args.batch_size,
        args.append,
    )
