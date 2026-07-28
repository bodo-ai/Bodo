"""
Create a directory-based Iceberg catalog and generate TPC-H data as Iceberg tables
using DuckDB's TPCH extension.

Usage (local paths):
    python generate_data_iceberg.py --sf 1 --outdir /path/to/output [--recreate_dir]

Or to write to S3:
    python generate_data_iceberg.py --sf 1 --iceberg_path s3://your-bucket/tpch_sf1_iceberg
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyiceberg.catalog import WAREHOUSE_LOCATION
from tqdm import tqdm

TPCH_TABLES = [
    "lineitem",
    "orders",
    "partsupp",
    "customer",
    "part",
    "supplier",
    "nation",
    "region",
]


def is_s3_path(path: str) -> bool:
    return urlparse(path).scheme == "s3"


def generate_duckdb_parquet(sf: int, parquet_path: str):
    """
    Generate TPC-H data using DuckDB and export it as Parquet files.
    Each variable length table e.g. lineitem, will have SF parquet files
    generated.

    Args:
        sf (int): The scale factor for the TPC-H data.
        parquet_path (str): The output directory where the Parquet files will be stored.
    """
    # Tables whose row count looks like SF * X
    # TODO(Scott): For smaller tables e.g. suppliers, we may want to combine
    # multiple Parquet files to avoid writing too many small files.
    variable_tables = [
        "lineitem",  # ~ 6M x SF
        "orders",  # ~ 1.5M x SF
        "partsupp",  # ~ 0.8M x SF
        "customer",  # ~ 0.3M x SF
        "part",  # ~ 0.2M x SF
        "supplier",  # ~ 10K x SF
    ]

    # Single Parquet file tables
    small_tables = ["nation", "region"]

    with tempfile.TemporaryDirectory() as tmpdir:
        duckdb_path = f"{tmpdir}/tpch_sf{sf}_duckdb.db"

        with duckdb.connect(str(duckdb_path)) as con:
            con.execute("INSTALL tpch")
            con.execute("LOAD tpch")

            for child in tqdm(range(sf), desc="Generating TPC-H data"):
                con.execute("CALL dbgen(sf=?, children=?, step=?)", [sf, sf, child])

                # Export every variable-sized table
                for table in variable_tables:
                    os.makedirs(f"{parquet_path}/{table}", exist_ok=True)
                    con.execute(f"""
                        COPY {table}
                        TO '{parquet_path}/{table}/part-{child:03d}.parquet'
                        (FORMAT PARQUET)
                    """)

                    con.execute(f"DELETE FROM {table}")

            for table in small_tables:
                os.makedirs(f"{parquet_path}/{table}", exist_ok=True)
                con.execute(f"""
                    COPY {table}
                    TO '{parquet_path}/{table}/part-000.parquet'
                    (FORMAT PARQUET)
                """)


def create_iceberg_tables(parquet_path: str, iceberg_path: str, sf: int):
    """
    Create Iceberg tables from the generated Parquet files.

    Args:
        parquet_path (str): The directory containing the Parquet files.
        iceberg_path (str): The output directory where the Iceberg tables will be stored.
    """
    from bodo.io.iceberg.catalog.dir import DirCatalog

    if not is_s3_path(iceberg_path):
        os.makedirs(iceberg_path, exist_ok=True)

    warehouse = (
        os.path.abspath(iceberg_path) if not is_s3_path(iceberg_path) else iceberg_path
    )

    catalog = DirCatalog(f"TPCH_SF{sf}", **{WAREHOUSE_LOCATION: warehouse})

    for table in TPCH_TABLES:
        table_dir = Path(parquet_path) / table

        dataset = ds.dataset(table_dir, format="parquet")
        schema = dataset.schema

        # Uses large write threshold so number of files matches parquet dataset
        iceberg_table = catalog.create_table(
            table,
            schema,
            properties={
                "write.target-file-size-bytes": str(100 * 1024**3),  # 100 GiB
            },
        )

        for pq_file in tqdm(
            sorted(table_dir.glob("*.parquet")), desc=f"Copying {table} to Iceberg: "
        ):
            table_fragment = pq.read_table(pq_file)
            iceberg_table.append(table_fragment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sf",
        type=int,
        required=True,
        help="The scale factor for the TPC-H data.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="/tmp",
        help="The output directory where the Parquet files will be stored."
        "The Iceberg tables will be stored here if --iceberg_path is"
        "not specified (in subdirectory `tpch_sf{sf}_iceberg`).",
    )
    parser.add_argument(
        "--iceberg_path",
        type=str,
        required=False,
        help="The output directory where the Iceberg tables will be stored. "
        "If not specified, it will default to a subdirectory in --outdir.",
    )
    parser.add_argument(
        "--recreate_dir",
        action="store_true",
        help="If provided, the output directory will be recreated before generating Parquet files.",
    )
    args = parser.parse_args()

    if args.recreate_dir:
        if os.path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        os.makedirs(args.outdir, exist_ok=True)

    parquet_path = f"{args.outdir}/tpch_sf{args.sf}_pq"
    iceberg_path = args.iceberg_path or f"{args.outdir}/tpch_sf{args.sf}_iceberg"

    try:
        generate_duckdb_parquet(args.sf, parquet_path)
        create_iceberg_tables(parquet_path, iceberg_path, args.sf)
    finally:
        shutil.rmtree(parquet_path, ignore_errors=True)


if __name__ == "__main__":
    main()
