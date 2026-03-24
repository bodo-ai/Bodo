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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "data", "tpch", "SF10"
        ),
    )
    parser.add_argument("--n_workers", type=int, default=1)
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
    args = parser.parse_args()

    if args.log_timings and not os.path.exists(args.log_timings):
        with open(args.log_timings, "w") as f:
            f.write("scale_factor,n_gpus,implementation,time_seconds,params\n")

    scale_factor = args.root.split("/")[-1].replace("SF", "")
    if scale_factor.isdigit():
        scale_factor = int(scale_factor)
    else:
        scale_factor = 0

    # Configure Dask to have longer worker timeouts for long-running tasks.
    dask.config.set({"distributed.comm.timeouts.tcp": "900s"})
    dask.config.set({"distributed.comm.timeouts.connect": "600s"})

    _ = Client(LocalCUDACluster(n_workers=args.n_workers, enable_cudf_spill=True))

    if args.warmup:
        try:
            print("Running warmup...")
            q5(args.root).compute()
            print("Warmup complete.")
        except Exception as e:
            print(f"Error during warmup run: {e}")
    for i in range(args.n_iters):
        try:
            t0 = time.time()
            result = q5(args.root).compute()
            print(result)
            total_time = time.time() - t0
            print(
                f"Q5 dask (sf={scale_factor}, n_gpus={args.n_workers}): {i} took {total_time:.4f} s"
            )

            if args.log_timings:
                with open(args.log_timings, "a") as f:
                    f.write(f"{scale_factor},{args.n_workers},dask,{total_time:.4f},\n")
        except Exception as e:
            print(
                f"Error executing query sf={scale_factor}, n_gpus={args.n_workers}: {e}"
            )


if __name__ == "__main__":
    main()
