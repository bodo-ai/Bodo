# Original from https://gist.githubusercontent.com/UranusSeven/55817bf0f304cc24f5eb63b2f1c3e2cd/raw/796dbd2fce6441821fc0b5bc51491edb49639c55/tpch.py
import argparse
import importlib
import inspect
import time

import polars as pl
import settings


def _dask_fs(local=True):
    if local:
        return "pyarrow"
    else:
        import boto3
        from pyarrow.fs import S3FileSystem

        session = boto3.session.Session()
        credentials = session.get_credentials()

        fs = S3FileSystem(
            secret_key=credentials.secret_key,
            access_key=credentials.access_key,
            region="us-east-2",
            session_token=credentials.token,
        )
        return fs


def run_queries(
    root: str, queries: list[int], scale_factor: float, backend: str = "polars"
):
    config = settings.Settings()
    config.scale_factor = scale_factor
    root = config.dataset_base_dir

    total_start = time.time()
    for query in queries:
        if backend in ("duckdb", "polars"):
            qi = importlib.import_module(f"queries.{backend}.q{query}")
        elif backend == "dask":
            import queries.dask.queries as dask_queries

            dask_fs = _dask_fs()
            qi = getattr(dask_queries, f"query_{query:02d}")

        query_start = time.time()
        if backend == "duckdb":
            qires = qi.q()
        elif backend == "polars":
            qargs = list(
                filter(lambda x: x != "kwargs", inspect.signature(qi.q).parameters)
            )
            qires = qi.q(
                *[
                    pl.scan_parquet(f"{root}/{x}.parquet").with_columns(
                        pl.all().name.to_lowercase()
                    )
                    for x in qargs
                ]
            )
            qires.collect()
        elif backend == "dask":
            qires = qi(root, dask_fs, scale_factor).compute()

        print(f"Query execution time (s): {time.time() - query_start}")
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        default="s3://bodo-example-data/tpch/SF1",
        help="The folder containing TPCH data",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Space separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        required=False,
        default=1.0,
        help="Scale factor (used in query 11).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=False,
        default="polars",
        help="Backend (polars or duckdb).",
    )

    args = parser.parse_args()
    data_set = args.folder
    scale_factor = args.scale_factor
    backend = args.backend
    assert backend in ("polars", "duckdb", "dask"), (
        "Backend must be 'polars', 'duckdb' or 'dask'"
    )

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    run_queries(data_set, queries=queries, scale_factor=scale_factor, backend=backend)


if __name__ == "__main__":
    main()
