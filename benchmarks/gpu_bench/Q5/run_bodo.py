import argparse
import datetime
import os

from linetimer import CodeTimer


def q(root, pd):
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
    result_df = gb.sort_values("REVENUE", ascending=False)

    return result_df


def main():
    parser = argparse.ArgumentParser()
    # Common Args
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
        "--log_timings",
        type=str,
        default="timings.csv",
    )
    # PD implementation (bodo or cudf)
    parser.add_argument(
        "--library",
        type=str,
        default="bodo",
        choices=["bodo", "cudf"],
    )

    args = parser.parse_args()

    if args.log_timings and not os.path.exists(args.log_timings):
        with open(args.log_timings, "w") as f:
            f.write("scale_factor,n_gpus,implementation,time_seconds\n")

    scale_factor = args.root.split("/")[-1].replace("SF", "")
    if scale_factor.isdigit():
        scale_factor = int(scale_factor)
    else:
        scale_factor = 0

    if args.library == "cudf":
        import cudf

        pd_impl = cudf
    else:
        import bodo.pandas as pd

        pd_impl = pd
        os.environ["BODO_NUM_WORKERS"] = str(args.n_workers)

    for i in range(args.n_iters):
        try:
            with CodeTimer(
                f"Q5 {args.library} (sf={scale_factor}, n_gpus={args.n_workers}): {i}",
                unit="s",
            ) as timer:
                print(q(args.root, pd_impl))

            if args.log_timings:
                with open(args.log_timings, "a") as f:
                    f.write(
                        f"{scale_factor},{args.n_workers},{args.library},{timer.took:.4f}\n"
                    )
        except Exception as e:
            print(
                f"Error executing query library={args.library}, sf={scale_factor}, n_gpus={args.n_workers}: {e}"
            )


if __name__ == "__main__":
    main()
