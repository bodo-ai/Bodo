# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Script that takes the table file output of BodoSQLFetchTablesWrapper.py
and reduces the file to the set of unique table names.
"""

import argparse


def main(input_file: str, output_file: str):
    unique_tables = set()
    # Read the input and remove any inputs
    with open(input_file, "r") as f:
        for line in f.readlines():
            unique_tables.add(line.strip())
    # Sort the output
    sorted_tables = sorted(unique_tables)
    with open(output_file, "w") as f:
        for table in sorted_tables:
            f.write(f"{table}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PruneFetchedTables",
        description="Reduce a table output file from BodoSQLFetchTablesWrapper.py to the unique tables",
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file.",
    )

    args = parser.parse_args()
    main(args.input_file, args.output_file)
