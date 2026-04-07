import argparse
import datetime
import os
import time

import dask
import dask_cudf
from dask.dataframe import DataFrame
from dask_cuda import LocalCUDACluster
from distributed import Client


def q5(root: str) -> DataFrame:
    """Implementation of TPC-H Query 5 using Dask-CuDF.

    Args:
        root: Path to the root directory containing the parquet files.

    Returns:
        DataFrame: A Dask DataFrame representing the query result.
    """
    region = dask_cudf.read_parquet(f"{root}/region.pq")
    nation = dask_cudf.read_parquet(f"{root}/nation.pq")
    customer = dask_cudf.read_parquet(f"{root}/customer.pq")
    lineitem = dask_cudf.read_parquet(f"{root}/lineitem.pq")
    orders = dask_cudf.read_parquet(f"{root}/orders.pq")
    supplier = dask_cudf.read_parquet(f"{root}/supplier.pq")

    var1 = "ASIA"
    var2 = datetime.date(1996, 1, 1)
    var3 = datetime.date(1997, 1, 1)

    jn1 = customer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn3 = jn2.merge(
        supplier,
        left_on=["L_SUPPKEY", "C_NATIONKEY"],
        right_on=["S_SUPPKEY", "S_NATIONKEY"],
    )
    jn4 = jn3.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn5 = jn4.merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")

    jn5 = jn5[jn5["R_NAME"] == var1]
    jn5 = jn5[(jn5["O_ORDERDATE"] >= var2) & (jn5["O_ORDERDATE"] < var3)]
    jn5["REVENUE"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME")["REVENUE"].sum()
    return gb.reset_index().sort_values("REVENUE", ascending=False)


def run_query_with_timing(root: str) -> tuple[DataFrame, float]:
    """Function for running Q5 and getting the result and total execution time."""
    start_time = time.time()
    result = q5(root).compute()
    total_time = time.time() - start_time

    return result, total_time


def run_query_dispatch(
    root: str, client: Client, run_multi_node: bool
) -> tuple[DataFrame, float]:
    """Dispatches the query to the Dask cluster and returns the result and execution time."""
    if run_multi_node:
        future = client.submit(run_query_with_timing, root)
        return future.result()
    else:
        return run_query_with_timing(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "data", "tpch", "SF10"
        ),
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of Dask workers (GPUs) to use. For multi-node runs, this specifies the number of instances to launch and Dask-CUDA will launch one worker per GPU.",
    )
    parser.add_argument("--n_iters", type=int, default=1)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="If set, a warmup run of the query will be executed before timing.",
    )
    parser.add_argument(
        "--log_timings",
        type=str,
        default=None,
        help="Path to CSV file where timings will be logged.",
    )
    parser.add_argument(
        "--print_output",
        action="store_true",
    )

    # Multi-Node Arguments:
    parser.add_argument(
        "--run_multi_node",
        action="store_true",
        help="If set, run the query in a multi-node Dask cluster (requires dask cloud provider).",
    )
    parser.add_argument(
        "--instance_profile_name",
        type=str,
        default=None,
        help="IAM instance profile name for EC2 instances for accessing S3",
    )
    parser.add_argument(
        "--subnet_id",
        type=str,
        default=None,
        help="Subnet ID for EC2 instances (within us-east-2)",
    )

    args = parser.parse_args()

    if args.log_timings and not os.path.exists(args.log_timings):
        with open(args.log_timings, "w") as f:
            f.write(
                "scale_factor,storage_type,n_gpus,implementation,time_seconds,params\n"
            )

    scale_factor = args.root.split("/")[-1].replace("SF", "")
    if scale_factor.isdigit():
        scale_factor = int(scale_factor)
    else:
        scale_factor = 0

    storage_type = "s3" if args.root.startswith("s3://") else "local"

    if args.run_multi_node:
        from dask_cloudprovider.aws import EC2Cluster

        # Use GPU AMI with Nvidia drivers pre-installed to speed up cluster startup time
        # The specific AMI below was obtained from the following command:
        # AMI with Nvidia drivers and docker pre-installed:
        # (Avoid bootstrap time)
        # aws ssm get-parameter \
        #     --region us-east-2 \
        #     --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
        #     --query 'Parameter.Value' \
        #     --output text
        ami = "ami-0600d0aaccc95db72"

        # Instance profile with permissions required for writing and potentially reading from S3
        instance_profile = (
            None
            if args.instance_profile_name is None
            else {"Name": args.instance_profile_name}
        )

        # See https://docs.rapids.ai/deployment/stable/cloud/aws/ec2-multi/
        # for more details about cluster configuration and setup.
        cluster = EC2Cluster(
            instance_type="g7e.12xlarge",
            docker_image="nvcr.io/nvidia/rapidsai/base:26.02-cuda13-py3.13",
            worker_class="dask_cuda.CUDAWorker",
            worker_options={"rmm_managed_memory": True},
            docker_args="--shm-size=256m -e EXTRA_CONDA_PACKAGES=s3fs",
            n_workers=args.n_workers,
            filesystem_size=250,  # GB
            region="us-east-2",
            subnet_id=args.subnet_id,
            ami=ami,
            iam_instance_profile=instance_profile,
            bootstrap=False,
            security=False,
        )
    else:
        # Configure Dask to have longer worker timeouts for long-running tasks.
        dask.config.set({"distributed.comm.timeouts.tcp": "900s"})
        dask.config.set({"distributed.comm.timeouts.connect": "600s"})

        cluster = LocalCUDACluster(n_workers=args.n_workers, enable_cudf_spill=True)
    client = Client(cluster)

    if args.warmup:
        try:
            print("Running warmup...")
            run_query_dispatch(args.root, client, args.run_multi_node)
            print("Warmup complete.")
        except Exception as e:
            print(f"Error during warmup run: {e}")
    for i in range(args.n_iters):
        try:
            res, total_time = run_query_dispatch(args.root, client, args.run_multi_node)

            if args.print_output:
                print(res)

            print(
                f"Q5 dask (sf={scale_factor}, n_gpus={args.n_workers}): {i} took {total_time:.4f} s"
            )

            if args.log_timings:
                extra_params = (
                    "cluster=multi-node" if args.run_multi_node else "cluster=local"
                )
                with open(args.log_timings, "a") as f:
                    f.write(
                        f"{scale_factor},{storage_type},{args.n_workers},dask,{total_time:.4f},{extra_params}\n"
                    )
        except Exception as e:
            print(
                f"Error executing query sf={scale_factor}, n_gpus={args.n_workers}: {e}"
            )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
