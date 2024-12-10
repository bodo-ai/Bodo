"""This script runs the full bodo benchmark on the platform.
First, ensure that you have bodosdk installed: `pip install bodosdk`, you will
also need to have an account on the bodo platform and the following environment
variables set:
    * GITHUB_USERNAME: Your github username
    * GITHUB_PAT: Your github access token
    * BODO_CLIENT_ID: Bodo platform client id
    * BODO_SECRET_KEY: Bodo platform secret key
Refer to our SDK guide for more details:
    https://docs.bodo.ai/latest/guides/using_bodo_platform/bodo_platform_sdk_guide/#installation

usage:
    python run_bodo.py
"""

import os

from bodosdk import BodoWorkspaceClient

bodo_workspace = BodoWorkspaceClient()
benchmark_cluster = bodo_workspace.ClusterClient.create(
    name="Benchmark Bodo", instance_type="c6i.8xlarge", workers_quantity=4
)
benchmark_cluster.wait_for_status(["RUNNING"])

benchmark_cluster.start(wait=True)
benchmark_job = benchmark_cluster.run_job(
    code_type="PYTHON",
    source={
        "type": "GIT",
        "repoUrl": "https://github.com/bodo-ai/Bodo.git",
        "username": os.environ["GITHUB_USERNAME"],
        "token": os.environ["GITHUB_PAT"],
    },
    exec_file="benchmarks/nyc_taxi_precipitation.py",
)
print(benchmark_job.wait_for_status(["SUCCEEDED"]).get_stderr())

# cleanup:
benchmark_cluster.stop(wait=True)
benchmark_cluster.delete(wait=True)
