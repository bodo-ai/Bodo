"""This script runs the full Bodo benchmark on the Bodo Platform.
First, ensure that you have bodosdk installed (pip install bodosdk), you will
also need to have an account on the Bodo Platform and the following environment
variables set:
    * BODO_CLIENT_ID: Bodo Platform client id
    * BODO_SECRET_KEY: Bodo Platform secret key
Refer to the SDK guide for more details:

https://docs.bodo.ai/latest/guides/using_bodo_platform/bodo_platform_sdk_guide/#installation

NOTE: This script assumes that you have the file `dataframe_queries.py`
copied in your current workspace.

usage:
    python run_bodo.py --folder <data_folder> --scale_factor <scale_factor> --queries <query_numbers>
"""

import argparse

from bodosdk import BodoWorkspaceClient

NUM_WORKERS = 4
WORKER_INSTANCE = "r6i.16xlarge"


def run_bodo_benchmark(folder, queries, scale_factor):
    bodo_workspace = BodoWorkspaceClient()
    benchmark_cluster = bodo_workspace.ClusterClient.create(
        name="Benchmark Bodo",
        instance_type=WORKER_INSTANCE,
        workers_quantity=NUM_WORKERS,
    )
    benchmark_cluster.wait_for_status(["RUNNING"])

    for query in queries:
        args = {
            "folder": folder,
            "scale_factor": str(scale_factor),
            "queries": str(query),
        }
        arg_str = " ".join(f"--{key} {value}" for key, value in args.items())

        benchmark_job = benchmark_cluster.run_job(
            code_type="PYTHON",
            source={"type": "WORKSPACE", "path": "/"},
            exec_file="dataframe_queries.py",
            args=arg_str,
        )
        print(benchmark_job.wait_for_status(["SUCCEEDED"]).get_stdout())

    # cleanup:
    benchmark_cluster.stop(wait=True)
    benchmark_cluster.delete(wait=True)


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    run_bodo_benchmark(args.folder, queries, args.scale_factor)


if __name__ == "__main__":
    main()
