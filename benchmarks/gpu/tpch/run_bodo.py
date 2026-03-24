import argparse
import datetime
import os
import time

import boto3
import pandas
from pandas import DataFrame


def q5(root: str, pd) -> DataFrame:
    """Implementation of of TPC-H Query 5 using the Pandas API.

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

    Args:
        root: Path to the root directory containing the parquet files.
        pd: The Pandas module or a compatible API.

    Returns:
        DataFrame: The query result.
    """
    lineitem = pd.read_parquet(f"{root}/lineitem.pq")
    orders = pd.read_parquet(f"{root}/orders.pq")
    customer = pd.read_parquet(f"{root}/customer.pq")
    nation = pd.read_parquet(f"{root}/nation.pq")
    region = pd.read_parquet(f"{root}/region.pq")
    supplier = pd.read_parquet(f"{root}/supplier.pq")

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

    gb = jn5.groupby("N_NAME", as_index=False)["REVENUE"].sum()
    result_df = gb
    result_df = gb.sort_values("REVENUE", ascending=False)

    return result_df


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
        help="If set, run a warmup run that is not timed.",
    )
    parser.add_argument(
        "--log_timings",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--print_output",
        action="store_true",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="bodo",
        choices=["bodo", "cudf"],
    )
    parser.add_argument(
        "--answer_path",
        type=str,
        default=None,
        help="Path to saved answer, in parquet format for asserting output matches expected.",
    )

    # Bodo Config
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="If set, run the query with parallel execution disabled (no spawn mode).",
    )
    parser.add_argument(
        "--dump_plan",
        action="store_true",
    )
    parser.add_argument(
        "--tracedir",
        type=str,
        default=None,
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

        # Variables required for using kvikio for S3 reads.
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token
        os.environ["AWS_DEFAULT_REGION"] = (
            session.region_name if session.region_name else "us-east-2"
        )
        os.environ["AWS_REGION"] = (
            session.region_name if session.region_name else "us-east-2"
        )

    if args.library == "cudf":
        import cudf

        pd_impl = cudf
    else:
        # Bodo envs (set prior to importing bodo.pandas)
        os.environ["BODO_GPU"] = "1"
        os.environ["BODO_NUM_WORKERS"] = str(args.n_workers)
        if args.batch_size:
            os.environ["BODO_GPU_STREAMING_BATCH_SIZE"] = str(args.batch_size)
        os.environ["BODO_DATAFRAME_LIBRARY_DUMP_PLANS"] = "1" if args.dump_plan else "0"
        os.environ["BODO_DATAFRAME_LIBRARY_RUN_PARALLEL"] = (
            "0" if args.no_parallel else "1"
        )
        if args.tracedir:
            os.environ["BODO_TRACING_LEVEL"] = "1"
            os.environ["BODO_TRACING_OUTPUT_DIR"] = args.tracedir

        import bodo.pandas

        pd_impl = bodo.pandas

        print("Bodo Config:")
        print(f"  Number of Workers: {args.n_workers}")
        print(f"  Streaming Batch Size: {args.batch_size}")
        print(f"  Run Parallel: {not args.no_parallel}")
        print(f"  Dump Plan: {args.dump_plan}")
        print(f"  Trace Dir: {args.tracedir}")

    if args.log_timings and not os.path.exists(args.log_timings):
        with open(args.log_timings, "w") as f:
            f.write(
                "scale_factor,storage_type,n_gpus,implementation,time_seconds,params\n"
            )

    n_correct = 0
    if args.warmup:
        try:
            print("Running warmup...")
            warmup_res = q5(args.root, pd_impl)
            if args.library == "bodo":
                warmup_res = warmup_res.execute_plan()
            print("Warmup complete.")
        except Exception as e:
            print(f"Error during warmup run: {e}")
    for i in range(args.n_iters):
        try:
            t0 = time.time()
            result = q5(args.root, pd_impl)
            if args.library == "bodo":
                result = result.execute_plan()
            total_time = time.time() - t0
            print(
                f"Q5 {args.library} (sf={scale_factor}, n_gpus={args.n_workers}): {i} took {total_time:.4f} s"
            )
            if args.print_output:
                print(result)

            if args.answer_path:
                from bodo.tests.utils import _test_equal

                answer_df = pandas.read_parquet(args.answer_path)
                _test_equal(result, answer_df)
                n_correct += 1

            if args.log_timings:
                batch_size_str = (
                    f"batch_size={args.batch_size}" if args.batch_size else ""
                )
                with open(args.log_timings, "a") as f:
                    f.write(
                        f"{scale_factor},{storage_type},{args.n_workers},{args.library},{total_time:.4f},{batch_size_str}\n"
                    )
        except Exception as e:
            print(
                f"Error executing query library={args.library}, sf={scale_factor}, n_gpus={args.n_workers}: {e}"
            )

    if args.answer_path:
        print(f"Correct Results: {n_correct}/{args.n_iters}")


if __name__ == "__main__":
    main()
