"""This script runs the full bodo benchmark on the platform.
First, ensure that you have bodosdk installed: `pip install bodosdk`, you will
also need to have an account on the bodo platform and the following environment
variables set:
    * BODO_CLIENT_ID: Bodo platform client id
    * BODO_SECRET_KEY: Bodo platform secret key
Refer to our SDK guide for more details:

https://docs.bodo.ai/latest/guides/using_bodo_platform/bodo_platform_sdk_guide/#installation

NOTE: This script assumes that you have the file `nyc_taxi_precipitation.py`
copied in your current workspace.

usage:
    python run_bodo.py
"""

from bodosdk import BodoWorkspaceClient


def run_bodo_benchmark():
    bodo_workspace = BodoWorkspaceClient()
    benchmark_cluster = bodo_workspace.ClusterClient.create(
        name="Benchmark Bodo", instance_type="c6i.16xlarge", workers_quantity=4
    )
    benchmark_cluster.wait_for_status(["RUNNING"])

    # run the job three times
    for _ in range(3):
        benchmark_job = benchmark_cluster.run_job(
            code_type="PYTHON",
            source={"type": "WORKSPACE", "path": "/"},
            exec_file="nyc_taxi_precipitation.py",
        )
        print(benchmark_job.wait_for_status(["SUCCEEDED"]).get_stdout())

    # cleanup:
    benchmark_cluster.stop(wait=True)
    benchmark_cluster.delete(wait=True)


if __name__ == "__main__":
    run_bodo_benchmark()
