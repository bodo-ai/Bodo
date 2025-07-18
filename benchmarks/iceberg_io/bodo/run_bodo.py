"""This script runs the full Bodo benchmark on the Bodo Platform.
First, ensure that you have bodosdk installed (pip install bodosdk), you will
also need to have an account on the Bodo Platform and the following environment
variables set:
    * BODO_CLIENT_ID: Bodo Platform client id
    * BODO_SECRET_KEY: Bodo Platform secret key
The workspace will need an instance role with the name "s3tables" that provides S3FullAccess and S3TablesFullAccess.
Refer to the SDK guide for more details:

https://docs.bodo.ai/latest/guides/using_bodo_platform/bodo_platform_sdk_guide/#installation

NOTE: This script assumes that you have the file `copy_orders_bodo.py`
copied in your current workspace.

usage:
    python run_bodo.py
"""

import argparse

from bodosdk import BodoWorkspaceClient


def run_bodo_benchmark(num_workers: int = 1):
    bodo_workspace = BodoWorkspaceClient()

    instance_roles = bodo_workspace.InstanceRoleClient.list(
        filters={"names": ["s3tables"]}
    )
    assert len(instance_roles) == 1, (
        "Instance role 's3tables' not found. Please create it with S3TablesFullAccess."
    )
    instance_role = bodo_workspace.InstanceRoleClient.get(instance_roles[0].id)

    benchmark_cluster = bodo_workspace.ClusterClient.create(
        name="Benchmark Bodo",
        instance_type="c6i.32xlarge",
        workers_quantity=num_workers,
        instance_role=instance_role,
    )
    benchmark_cluster.wait_for_status(["RUNNING"])

    # run the job three times
    for _ in range(3):
        benchmark_job = benchmark_cluster.run_job(
            code_type="PYTHON",
            source={"type": "WORKSPACE", "path": "/"},
            exec_file="copy_orders_bodo.py",
        )
        print(benchmark_job.wait_for_status(["SUCCEEDED"]).get_stdout())

    # cleanup:
    benchmark_cluster.stop(wait=True)
    benchmark_cluster.delete(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers in cluster."
    )
    args = parser.parse_args()

    run_bodo_benchmark(args.num_workers)


if __name__ == "__main__":
    main()
