"""
Usage:
    python run_polars.py --root <path_to_parquet_files> --engine {cudf,dask} --n_workers <num_gpus>

Run python run_polars.py --help for more details on command line arguments.
"""

import argparse
import os
from datetime import date

import boto3
import polars as pl
from linetimer import CodeTimer


def q(root, scale_factor) -> pl.LazyFrame:
    """
    select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
    from
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region
    where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and l_suppkey = s_suppkey
        and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = '[REGION]'
        and o_orderdate >= date '[DATE]'
        and o_orderdate < date '[DATE]' + interval '1' year
    group by
        n_name
        order by
    revenue desc;
    """

    customer = pl.scan_parquet(f"{root}/customer.pq/*.pq")
    orders = pl.scan_parquet(f"{root}/orders.pq/*.pq")
    lineitem = pl.scan_parquet(f"{root}/lineitem.pq/*.pq")
    # For larger scale factors, supplier becomes multi-part.
    if scale_factor >= 1000:
        supplier = pl.scan_parquet(f"{root}/supplier.pq/*.pq")
    else:
        supplier = pl.scan_parquet(f"{root}/supplier.pq")
    region = pl.scan_parquet(f"{root}/region.pq")
    nation = pl.scan_parquet(f"{root}/nation.pq")

    var1 = "ASIA"
    var2 = date(1996, 1, 1)
    var3 = date(1997, 1, 1)

    return (
        customer.join(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
        .join(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
        .join(
            supplier,
            left_on=["L_SUPPKEY", "C_NATIONKEY"],
            right_on=["S_SUPPKEY", "S_NATIONKEY"],
        )
        .join(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
        .join(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")
        .filter(pl.col("R_NAME") == var1)
        .filter(pl.col("O_ORDERDATE").is_between(var2, var3, closed="left"))
        .with_columns(
            (pl.col("L_EXTENDEDPRICE") * (1 - pl.col("L_DISCOUNT"))).alias("REVENUE")
        )
        .group_by("N_NAME")
        .agg(pl.sum("REVENUE"))
        .sort(by="REVENUE", descending=True)
    )


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
        "--visualize_plan",
        type=str,
        default="",
        help="Base name for visualized query plans. If provided, both optimized and unoptimized plans will be visualized and saved as <name>_optimized.png and <name>_unoptimized.png.",
    )
    parser.add_argument(
        "--engine", type=str, default="cudf", choices=["cudf", "dask", "cpu"]
    )
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=1)
    parser.add_argument(
        "--log_timings",
        type=str,
        default="timings.csv",
        help="Path to CSV file where timings will be logged. If not provided, timings will not be logged to a file.",
    )
    parser.add_argument(
        "--store_output",
        action="store_true",
    )
    parser.add_argument(
        "--print_output",
        action="store_true",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
    )
    args = parser.parse_args()

    scale_factor = args.root.split("/")[-1].replace("SF", "")
    if scale_factor.isdigit():
        scale_factor = int(scale_factor)
    else:
        scale_factor = 0

    storage_type = "s3" if args.root.startswith("s3://") else "local"
    if storage_type == "s3":
        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()

        # Variables required for using kvikio for S3 reads (set before creating engine).
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token
        os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
        os.environ["AWS_REGION"] = "us-east-2"

    if args.engine != "dask" and args.n_workers > 1:
        raise ValueError(
            f"engine={args.engine} does not support multiple workers. Please set n_workers to 1 or choose 'dask' engine."
        )

    if args.engine == "cudf":
        pl_engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    elif args.engine == "cpu":
        pl_engine = "streaming"
    else:
        import dask
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster

        # Configure Dask to have longer worker timeouts for long-running tasks.
        dask.config.set({"distributed.comm.timeouts.tcp": "900s"})
        dask.config.set({"distributed.comm.timeouts.connect": "600s"})

        _ = Client(
            LocalCUDACluster(
                n_workers=args.n_workers,
                enable_cudf_spill=True,
                rmm_async=True,
            )
        )

        pl_engine = pl.GPUEngine(
            executor="streaming",
            executor_options={"cluster": "distributed"},
            raise_on_fail=True,
        )

    if args.visualize_plan:
        result: pl.LazyFrame = q(args.root, scale_factor)

        unoptimized_gviz_out_path = f"{args.visualize_plan}_unoptimized.png"
        result.show_graph(
            engine=pl_engine, optimized=False, output_path=unoptimized_gviz_out_path
        )

        optimized_gviz_out_path = f"{args.visualize_plan}_optimized.png"
        result.show_graph(
            engine=pl_engine, optimized=True, output_path=optimized_gviz_out_path
        )

        print(
            f"Unoptimized query plan visualized and saved to: {unoptimized_gviz_out_path}"
        )
        print(
            f"Optimized query plan visualized and saved to: {optimized_gviz_out_path}"
        )

    if args.log_timings and not os.path.exists(args.log_timings):
        with open(args.log_timings, "w") as f:
            f.write(
                "scale_factor,storage_type,n_gpus,implementation,time_seconds,extras\n"
            )

    if args.warmup:
        try:
            print("Running warmup...")
            q(args.root, scale_factor).collect(engine=pl_engine)
            print("Warmup complete.")
        except Exception as e:
            print(f"Error during warmup run: {e}")
    for i in range(args.n_iters):
        try:
            with CodeTimer(
                f"Q5 Polars GPU Streaming (sf={scale_factor} engine={args.engine}, n_gpus={args.n_workers}): {i}",
                unit="s",
            ) as timer:
                result = q(args.root, scale_factor).collect(engine=pl_engine)
                if args.print_output:
                    print(result)

            if args.store_output:
                output_path = f"q5_polars_{args.engine}_sf{scale_factor}_gpu{args.n_workers}_iter{i}.parquet"
                result.to_pandas().to_parquet(output_path)
                print(f"Query output saved to {output_path}")

            if args.log_timings:
                with open(args.log_timings, "a") as f:
                    f.write(
                        f"{scale_factor},{storage_type},{args.n_workers},Polars[{args.engine}],{timer.took:.4f},\n"
                    )
        except Exception as e:
            print(
                f"Error executing query sf={scale_factor} engine={args.engine}, n_gpus={args.n_workers}: {e}"
            )


if __name__ == "__main__":
    main()
