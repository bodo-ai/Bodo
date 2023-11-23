# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Script to convert the parquet file containing the queries with their relevant costs
into a directory of SQL files.
"""
import argparse
import os

import pandas as pd


def main(parquet_file: str, output_directory: str, max_queries: int):
    df = pd.read_parquet(parquet_file)
    # Filter to the relevant columns
    df = df[["dag_id", "task_id", "credits_per_month", "query_text"]]
    # Order by credits consumed
    df = df.sort_values(by="credits_per_month", ascending=False)
    # Prune the credits
    df = df[["dag_id", "task_id", "query_text"]]
    # Select the top N queries
    df = df.head(max_queries)
    # Create the directory
    os.mkdir(output_directory)
    os.chdir(output_directory)
    # Iterate over the row, generating a file
    for _, row in df.iterrows():
        filename = f'{row["dag_id"]}-{row["task_id"]}.sql'
        with open(filename, "w") as f:
            f.write(row["query_text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GreenlightParquetToQueryFolder",
        description="Convert a parquet file to a directory of SQL files.",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        help="Path to parquet file containing data.",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Path for the output directory.",
    )
    parser.add_argument(
        "-n",
        "--num_queries",
        type=int,
        required=False,
        default=500,
        help="Maximum number of queries to store in the output directory.",
    )

    args = parser.parse_args()
    main(args.filename, args.output_directory, args.num_queries)
