# Original from https://gist.githubusercontent.com/UranusSeven/55817bf0f304cc24f5eb63b2f1c3e2cd/raw/796dbd2fce6441821fc0b5bc51491edb49639c55/tpch.py
import argparse
import datetime
import functools
import inspect
import time
import warnings
from collections.abc import Callable

import pandas as pd

import bodo.pandas
import bodo.spawn.spawner as spawner


def load_lineitem(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/lineitem.pq"
    df = pd.read_parquet(data_path)
    df["L_SHIPDATE"] = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    return df


def load_part(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/part.pq"
    df = pd.read_parquet(data_path)
    return df


def load_orders(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/orders.pq"
    df = pd.read_parquet(data_path)
    df["O_ORDERDATE"] = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    return df


def load_customer(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/customer.pq"
    df = pd.read_parquet(data_path)
    return df


def load_nation(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/nation.pq"
    df = pd.read_parquet(data_path)
    return df


def load_region(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/region.pq"
    df = pd.read_parquet(data_path)
    return df


def load_supplier(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/supplier.pq"
    df = pd.read_parquet(data_path)
    return df


def load_partsupp(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/partsupp.pq"
    df = pd.read_parquet(data_path)
    return df


def timethis(q: Callable, name: str | None = None):
    @functools.wraps(q)
    def wrapped(*args, **kwargs):
        t = time.time()
        q(*args, **kwargs)
        msg = name or f"{q.__name__.upper()} Execution time (s):"
        print(f"{msg} {time.time() - t:f}")

    return wrapped


_query_to_args: dict[int, list[str]] = {}


def collect_datasets(func: Callable):
    _query_to_args[int(func.__name__[1:])] = list(inspect.signature(func).parameters)
    return func


def tpch_q01(lineitem, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/dask/q1.py
    """
    var1 = datetime.date(1998, 9, 2)
    filt = lineitem[lineitem["L_SHIPDATE"] <= var1]

    filt["DISC_PRICE"] = filt.L_EXTENDEDPRICE * (1.0 - filt.L_DISCOUNT)
    filt["CHARGE"] = filt.L_EXTENDEDPRICE * (1.0 - filt.L_DISCOUNT) * (1.0 + filt.L_TAX)

    gb = filt.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)
    agg = gb.agg(
        SUM_QTY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum"),
        SUM_BASE_PRICE=pd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="sum"),
        SUM_DISC_PRICE=pd.NamedAgg(column="DISC_PRICE", aggfunc="sum"),
        SUM_CHARGE=pd.NamedAgg(column="CHARGE", aggfunc="sum"),
        AVG_QTY=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean"),
        AVG_PRICE=pd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="mean"),
        AVG_DISC=pd.NamedAgg(column="L_DISCOUNT", aggfunc="mean"),
        COUNT_ORDER=pd.NamedAgg(column="L_ORDERKEY", aggfunc="size"),
    )

    result_df = agg.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    return result_df


def tpch_q02(part, partsupp, supplier, nation, region, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q2.py
    """
    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    jn = (
        part.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
        .merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
        .merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
        .merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")
    )

    jn = jn[jn["P_SIZE"] == var1]
    jn = jn[jn["P_TYPE"].str.endswith(var2)]
    jn = jn[jn["R_NAME"] == var3]

    gb = jn.groupby("P_PARTKEY", as_index=False)
    agg = gb["PS_SUPPLYCOST"].min()
    jn2 = agg.merge(jn, on=["P_PARTKEY", "PS_SUPPLYCOST"])

    sel = jn2.loc[
        :,
        [
            "S_ACCTBAL",
            "S_NAME",
            "N_NAME",
            "P_PARTKEY",
            "P_MFGR",
            "S_ADDRESS",
            "S_PHONE",
            "S_COMMENT",
        ],
    ]

    sort = sel.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    result_df = sort.head(100)

    return result_df


def tpch_q03(lineitem, orders, customer, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q3.py
    """
    var1 = "HOUSEHOLD"
    var2 = datetime.date(1995, 3, 4)

    fcustomer = customer[customer["C_MKTSEGMENT"] == var1]

    jn1 = fcustomer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")

    jn2 = jn2[jn2["O_ORDERDATE"] < var2]
    jn2 = jn2[jn2["L_SHIPDATE"] > var2]
    jn2["REVENUE"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)

    gb = jn2.groupby(["O_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False)
    agg = gb["REVENUE"].sum()

    sel = agg.loc[:, ["O_ORDERKEY", "REVENUE", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    sel = sel.rename(columns={"O_ORDERKEY": "L_ORDERKEY"})

    sorted = sel.sort_values(by=["REVENUE", "O_ORDERDATE"], ascending=[False, True])
    result_df = sorted.head(10)

    return result_df


def tpch_q04(lineitem, orders, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = datetime.date(1993, 11, 1)
    var2 = datetime.date(1993, 8, 1)

    flineitem = lineitem[lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE]
    forders = orders[(orders.O_ORDERDATE < var1) & (orders.O_ORDERDATE >= var2)]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = (
        jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"]
        .count()
        .sort_values(["O_ORDERPRIORITY"])
    )
    total.columns = ["O_ORDERPRIORITY", "ORDER_COUNT"]
    return total


def tpch_q05(lineitem, orders, customer, nation, region, supplier, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q5.py
    """
    var1 = "ASIA"
    var2 = datetime.date(1996, 1, 1)
    var3 = datetime.date(1997, 1, 1)

    jn1 = region.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
    jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
    jn3 = jn2.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn5 = jn4.merge(
        supplier,
        left_on=["L_SUPPKEY", "N_NATIONKEY"],
        right_on=["S_SUPPKEY", "S_NATIONKEY"],
    )

    jn5 = jn5[jn5["R_NAME"] == var1]
    jn5 = jn5[(jn5["O_ORDERDATE"] >= var2) & (jn5["O_ORDERDATE"] < var3)]
    jn5["REVENUE"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)

    gb = jn5.groupby("N_NAME", as_index=False)["REVENUE"].sum()
    result_df = gb.sort_values("REVENUE", ascending=False)

    return result_df


def tpch_q06(lineitem, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q6.py
    """
    var1 = datetime.date(1996, 1, 1)
    var2 = datetime.date(1997, 1, 1)
    var3 = 0.08
    var4 = 0.1
    var5 = 24

    filt = lineitem[(lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)]
    filt = filt[(filt["L_DISCOUNT"] >= var3) & (filt["L_DISCOUNT"] <= var4)]
    filt = filt[filt["L_QUANTITY"] < var5]
    result_value = (filt["L_EXTENDEDPRICE"] * filt["L_DISCOUNT"]).sum()
    result_df = pd.DataFrame({"REVENUE": [result_value]})

    return result_df


def tpch_q07(lineitem, supplier, orders, customer, nation, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q7.py
    """
    var1 = "FRANCE"
    var2 = "GERMANY"
    var3 = datetime.date(1995, 1, 1)
    var4 = datetime.date(1997, 1, 1)

    n1 = nation[(nation["N_NAME"] == var1)]
    n2 = nation[(nation["N_NAME"] == var2)]

    # Part 1
    jn1 = customer.merge(n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn2 = jn1.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn2.rename(columns={"N_NAME": "CUST_NATION"})
    jn3 = jn2.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn5 = jn4.merge(n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    df1 = jn5.rename(columns={"N_NAME": "SUPP_NATION"})

    # Part 2
    jn1 = customer.merge(n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn2 = jn1.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn2.rename(columns={"N_NAME": "CUST_NATION"})
    jn3 = jn2.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn5 = jn4.merge(n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    df2 = jn5.rename(columns={"N_NAME": "SUPP_NATION"})

    # Combine
    total = pd.concat([df1, df2])

    total = total[(total["L_SHIPDATE"] >= var3) & (total["L_SHIPDATE"] < var4)]
    total["VOLUME"] = total["L_EXTENDEDPRICE"] * (1.0 - total["L_DISCOUNT"])
    total["L_YEAR"] = total["L_SHIPDATE"].dt.year

    gb = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False)
    agg = gb.agg(REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum"))

    result_df = agg.sort_values(by=["SUPP_NATION", "CUST_NATION", "L_YEAR"])
    return result_df


def tpch_q08(
    part, lineitem, supplier, orders, customer, nation, region, pd=bodo.pandas
):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q8.py
    """
    var1 = "BRAZIL"
    var2 = "AMERICA"
    var3 = "ECONOMY ANODIZED STEEL"
    var4 = datetime.date(1995, 1, 1)
    var5 = datetime.date(1997, 1, 1)

    n1 = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
    n2 = nation.loc[:, ["N_NATIONKEY", "N_NAME"]]

    jn1 = part.merge(lineitem, left_on="P_PARTKEY", right_on="L_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn4 = jn3.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn5 = jn4.merge(n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn6 = jn5.merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")

    jn6 = jn6[(jn6["R_NAME"] == var2)]

    jn7 = jn6.merge(n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn7 = jn7[(jn7["O_ORDERDATE"] >= var4) & (jn7["O_ORDERDATE"] < var5)]
    jn7 = jn7[jn7["P_TYPE"] == var3]

    jn7["O_YEAR"] = jn7["O_ORDERDATE"].dt.year
    jn7["VOLUME"] = jn7["L_EXTENDEDPRICE"] * (1.0 - jn7["L_DISCOUNT"])
    jn7 = jn7.rename(columns={"N_NAME": "NATION"})

    # denominator: total volume per year
    denom = (
        jn7.groupby("O_YEAR", as_index=False)["VOLUME"]
        .sum()
        .rename(columns={"VOLUME": "TOTAL_VOLUME"})
    )

    # numerator: Brazil volume per year
    num = (
        jn7[jn7["NATION"] == var1]
        .groupby("O_YEAR", as_index=False)["VOLUME"]
        .sum()
        .rename(columns={"VOLUME": "BRAZIL_VOLUME"})
    )

    # join and compute ratio
    agg = denom.merge(num, on="O_YEAR", how="left")
    agg["MKT_SHARE"] = (agg["BRAZIL_VOLUME"] / agg["TOTAL_VOLUME"]).round(2)

    result_df = agg.sort_values("O_YEAR")
    result_df = result_df[["O_YEAR", "MKT_SHARE"]]

    return result_df


def tpch_q09(lineitem, orders, part, nation, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "ghost"

    part = part[part.P_NAME.str.contains(var1)]

    jn1 = part.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(
        lineitem,
        left_on=["PS_PARTKEY", "PS_SUPPKEY"],
        right_on=["L_PARTKEY", "L_SUPPKEY"],
    )
    jn4 = jn3.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn5 = jn4.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn5["O_YEAR"] = jn5["O_ORDERDATE"].dt.year
    jn5["NATION"] = jn5["N_NAME"]
    jn5["AMOUNT"] = (
        jn5["L_EXTENDEDPRICE"] * (1 - jn5["L_DISCOUNT"])
        - jn5["PS_SUPPLYCOST"] * jn5["L_QUANTITY"]
    )

    gb = jn5.groupby(["NATION", "O_YEAR"], as_index=False)
    agg = gb.agg(SUM_PROFIT=pd.NamedAgg(column="AMOUNT", aggfunc="sum"))
    agg["SUM_PROFIT"] = agg.SUM_PROFIT.round(2)
    result_df = agg.sort_values(by=["NATION", "O_YEAR"], ascending=[True, False])

    return result_df


def tpch_q10(lineitem, orders, customer, nation, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = datetime.date(1994, 11, 1)
    var2 = datetime.date(1995, 2, 1)

    forders = orders[(orders.O_ORDERDATE >= var1) & (orders.O_ORDERDATE < var2)]
    flineitem = lineitem[lineitem.L_RETURNFLAG == "R"]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["REVENUE"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    agg = jn3.groupby(
        [
            "C_CUSTKEY",
            "C_NAME",
            "C_ACCTBAL",
            "C_PHONE",
            "N_NAME",
            "C_ADDRESS",
            "C_COMMENT",
        ],
        as_index=False,
        sort=False,
    )["REVENUE"].sum()
    agg["REVENUE"] = agg.REVENUE.round(2)
    total = agg.sort_values("REVENUE", ascending=False)
    return total.head(20)


def tpch_q11(partsupp, supplier, nation, scale_factor=1.0, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "GERMANY"
    var2 = 0.0001 / scale_factor

    jn1 = partsupp.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn2 = jn1.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn2 = jn2[jn2["N_NAME"] == var1]

    threshold = (jn2["PS_SUPPLYCOST"] * jn2["PS_AVAILQTY"]).sum() * var2

    jn2["VALUE"] = jn2["PS_SUPPLYCOST"] * jn2["PS_AVAILQTY"]

    gb = jn2.groupby("PS_PARTKEY", as_index=False)["VALUE"].sum()

    filt = gb[gb["VALUE"] > threshold]
    filt["VALUE"] = filt.VALUE.round(2)
    result_df = filt.sort_values(by="VALUE", ascending=False)

    return result_df


def tpch_q12(lineitem, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = datetime.date(1994, 1, 1)
    var2 = datetime.date(1995, 1, 1)

    jn1 = orders.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn1 = jn1[
        (jn1["L_SHIPMODE"].isin(("MAIL", "SHIP")))
        & (jn1["L_COMMITDATE"] < jn1["L_RECEIPTDATE"])
        & (jn1["L_SHIPDATE"] < jn1["L_COMMITDATE"])
        & (jn1["L_RECEIPTDATE"] >= var1)
        & (jn1["L_RECEIPTDATE"] < var2)
    ]

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    gb = jn1.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg(
        HIGH_LINE_COUNT=g1, LOW_LINE_COUNT=g2
    )
    result_df = gb.sort_values("L_SHIPMODE")

    return result_df


def tpch_q13(customer, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "special"
    var2 = "requests"

    orders = orders[~orders["O_COMMENT"].str.contains(f"{var1}.*{var2}")]

    jn1 = customer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left")

    agg1 = jn1.groupby("C_CUSTKEY", as_index=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    agg2 = agg1.groupby("C_COUNT", as_index=False).agg(
        CUSTDIST=pd.NamedAgg(column="C_CUSTKEY", aggfunc="size")
    )

    result_df = agg2.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])

    return result_df


def tpch_q14(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = datetime.date(1994, 3, 1)
    var2 = datetime.date(1994, 4, 1)

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")

    jn1 = jn1[(jn1["L_SHIPDATE"] >= var1) & (jn1["L_SHIPDATE"] < var2)]

    # Promo revenue by line; CASE clause
    jn1["PROMO_REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])
    mask = jn1["P_TYPE"].str.match("PROMO*")
    jn1["PROMO_REVENUE"] = jn1["PROMO_REVENUE"].where(mask, 0.00)

    total_promo_revenue = jn1["PROMO_REVENUE"].sum()
    total_revenue = (jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])).sum()

    # aggregate promo revenue calculation
    ratio = 100.00 * total_promo_revenue / total_revenue
    result_df = pd.DataFrame({"PROMO_REVENUE": [round(ratio, 2)]})

    return result_df


def tpch_q15(lineitem, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = datetime.date(1996, 1, 1)
    var2 = datetime.date(1996, 4, 1)

    jn1 = lineitem[(lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)]

    jn1["REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])

    agg = jn1.groupby("L_SUPPKEY", as_index=False).agg(
        TOTAL_REVENUE=pd.NamedAgg(column="REVENUE", aggfunc="sum")
    )
    revenue = agg.rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})

    jn2 = supplier.merge(
        revenue, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
    )

    max_revenue = revenue["TOTAL_REVENUE"].max()
    jn2 = jn2[jn2["TOTAL_REVENUE"] == max_revenue]

    result_df = jn2[
        ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
    ].sort_values(by="S_SUPPKEY")

    return result_df


def tpch_q16(part, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "Brand#45"

    supplier["IS_COMPLAINT"] = supplier["S_COMMENT"].str.contains(
        "Customer.*Complaints"
    )

    complaint_suppkeys = supplier[supplier["IS_COMPLAINT"]]["S_SUPPKEY"]

    jn1 = partsupp[~partsupp["PS_SUPPKEY"].isin(complaint_suppkeys)]
    jn2 = jn1.merge(part, left_on="PS_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn2[
        (jn2["P_BRAND"] != var1)
        & (~jn2["P_TYPE"].str.startswith("MEDIUM POLISHED"))
        & (jn2["P_SIZE"].isin((49, 14, 23, 45, 19, 3, 36, 9)))
    ]

    agg = jn2.groupby(by=["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)[
        "PS_SUPPKEY"
    ].nunique()
    agg = agg.rename(columns={"PS_SUPPKEY": "SUPPLIER_CNT"})

    result_df = agg.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )

    return result_df


def tpch_q17(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "Brand#23"
    var2 = "MED BOX"

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn1 = jn1[((jn1["P_BRAND"] == var1) & (jn1["P_CONTAINER"] == var2))]

    agg = jn1.groupby("L_PARTKEY", as_index=False).agg(
        L_QUANTITY_AVG=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )

    jn4 = jn1.merge(agg, left_on="L_PARTKEY", right_on="L_PARTKEY", how="left")
    jn4 = jn4[jn4["L_QUANTITY"] < 0.2 * jn4["L_QUANTITY_AVG"]]
    total = jn4["L_EXTENDEDPRICE"].sum() / 7.0

    result_df = pd.DataFrame({"AVG_YEARLY": [round(total, 2)]})

    return result_df


def tpch_q18(lineitem, orders, customer, pd=bodo.pandas):
    """Adapted from:
    github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = 300

    agg1 = lineitem.groupby("L_ORDERKEY", as_index=False, sort=False)[
        "L_QUANTITY"
    ].sum()
    filt = agg1[agg1.L_QUANTITY > var1]
    jn1 = filt.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    agg2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
        sort=False,
    )["L_QUANTITY"].sum()
    total = agg2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    return total.head(100)


def tpch_q19(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn1 = jn1[
        (
            (jn1["P_BRAND"] == "Brand#31")
            & (jn1["P_CONTAINER"].isin(("SM CASE", "SM BOX", "SM PACK", "SM PKG")))
            & ((jn1["L_QUANTITY"] >= 4) & (jn1["L_QUANTITY"] <= 14))
            & (jn1["P_SIZE"] <= 5)
            & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
            & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
        )
        | (
            (jn1["P_BRAND"] == "Brand#43")
            & (jn1["P_CONTAINER"].isin(("MED BAG", "MED BOX", "MED PKG", "MED PACK")))
            & ((jn1["L_QUANTITY"] >= 15) & (jn1["L_QUANTITY"] <= 25))
            & ((jn1["P_SIZE"] >= 1) & (jn1["P_SIZE"] <= 10))
            & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
            & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
        )
        | (
            (jn1["P_BRAND"] == "Brand#43")
            & (jn1["P_CONTAINER"].isin(("LG CASE", "LG BOX", "LG PACK", "LG PKG")))
            & ((jn1["L_QUANTITY"] >= 26) & (jn1["L_QUANTITY"] <= 36))
            & (jn1["P_SIZE"] <= 15)
            & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
            & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
        )
    ]

    total = (jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])).sum()

    result_df = pd.DataFrame({"REVENUE": [round(total, 2)]})

    return result_df


def tpch_q20(lineitem, part, nation, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = datetime.date(1996, 1, 1)
    var2 = datetime.date(1997, 1, 1)
    var3 = "JORDAN"
    var4 = "azure"

    flineitem = lineitem[
        (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
    ]
    agg = flineitem.groupby(["L_SUPPKEY", "L_PARTKEY"], as_index=False).agg(
        SUM_QUANTITY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum")
    )
    agg["SUM_QUANTITY"] = agg["SUM_QUANTITY"] * 0.5

    fnation = nation[nation["N_NAME"] == var3]

    jn1 = supplier.merge(fnation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    fpart = part[part["P_NAME"].str.startswith(var4)]

    jn2 = partsupp.merge(fpart, left_on="PS_PARTKEY", right_on="P_PARTKEY")
    jn3 = jn2.merge(
        agg,
        left_on=["PS_SUPPKEY", "PS_PARTKEY"],
        right_on=["L_SUPPKEY", "L_PARTKEY"],
    )
    jn3 = jn3[jn3["PS_AVAILQTY"] > jn3["SUM_QUANTITY"]]
    jn4 = jn1.merge(jn3, left_on="S_SUPPKEY", right_on="PS_SUPPKEY")

    result_df = jn4[["S_NAME", "S_ADDRESS"]].sort_values("S_NAME", ascending=True)

    return result_df


def tpch_q21(lineitem, orders, supplier, nation, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "SAUDI ARABIA"

    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False).agg(
        NUM_SUPPLIERS=pd.NamedAgg(column="L_SUPPKEY", aggfunc="nunique")
    )
    gb1 = gb1[gb1["NUM_SUPPLIERS"] > 1]

    flineitem = lineitem[lineitem["L_RECEIPTDATE"] > lineitem["L_COMMITDATE"]]
    jn1 = gb1.merge(flineitem, on="L_ORDERKEY")

    gb2 = jn1.groupby("L_ORDERKEY", as_index=False).agg(
        NUNIQUE_COL=pd.NamedAgg(column="L_SUPPKEY", aggfunc="nunique")
    )

    jn2 = gb2.merge(jn1, on="L_ORDERKEY")
    jn3 = jn2.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn5 = jn4.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn5 = jn5[
        (
            (jn5["NUNIQUE_COL"] == 1)
            & (jn5["N_NAME"] == var1)
            & (jn5["O_ORDERSTATUS"] == "F")
        )
    ]
    gb3 = jn5.groupby("S_NAME", as_index=False).agg(
        NUMWAIT=pd.NamedAgg(column="NUNIQUE_COL", aggfunc="size")
    )

    result_df = gb3.sort_values(["NUMWAIT", "S_NAME"], ascending=[False, True]).head(
        100
    )

    return result_df


def tpch_q22(customer, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    customer["CNTRYCODE"] = customer["C_PHONE"].str.strip().str.slice(0, 2)
    fcustomers = customer[
        customer["CNTRYCODE"].isin(("13", "31", "23", "29", "30", "18", "17"))
    ]

    average_c_acctbal = fcustomers[fcustomers["C_ACCTBAL"] > 0.0]["C_ACCTBAL"].mean()
    custsale = fcustomers[fcustomers["C_ACCTBAL"] > average_c_acctbal]

    jn1 = custsale.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left")
    jn1 = jn1[jn1["O_CUSTKEY"].isnull()]

    agg1 = jn1.groupby("CNTRYCODE", as_index=False).agg(
        NUMCUST=pd.NamedAgg(column="C_ACCTBAL", aggfunc="size"),
        TOTACCTBAL=pd.NamedAgg(column="C_ACCTBAL", aggfunc="sum"),
    )

    result_df = agg1.sort_values("CNTRYCODE", ascending=True)

    return result_df


show_output = False


def exec_func(res):
    global show_output

    if show_output:
        print(res)
    elif isinstance(
        res, (bodo.pandas.BodoDataFrame, bodo.pandas.BodoSeries, bodo.pandas.BodoScalar)
    ):
        res.execute_plan()


@timethis
@collect_datasets
def q01(lineitem, pd):
    exec_func(tpch_q01(lineitem, pd))


@timethis
@collect_datasets
def q02(part, partsupp, supplier, nation, region, pd):
    exec_func(tpch_q02(part, partsupp, supplier, nation, region, pd))


@timethis
@collect_datasets
def q03(lineitem, orders, customer, pd):
    exec_func(tpch_q03(lineitem, orders, customer, pd))


@timethis
@collect_datasets
def q04(lineitem, orders, pd):
    exec_func(tpch_q04(lineitem, orders, pd))


@timethis
@collect_datasets
def q05(lineitem, orders, customer, nation, region, supplier, pd):
    exec_func(tpch_q05(lineitem, orders, customer, nation, region, supplier, pd))


@timethis
@collect_datasets
def q06(lineitem, pd):
    exec_func(tpch_q06(lineitem, pd))


@timethis
@collect_datasets
def q07(lineitem, supplier, orders, customer, nation, pd):
    exec_func(tpch_q07(lineitem, supplier, orders, customer, nation, pd))


@timethis
@collect_datasets
def q08(part, lineitem, supplier, orders, customer, nation, region, pd):
    exec_func(tpch_q08(part, lineitem, supplier, orders, customer, nation, region, pd))


@timethis
@collect_datasets
def q09(lineitem, orders, part, nation, partsupp, supplier, pd):
    exec_func(tpch_q09(lineitem, orders, part, nation, partsupp, supplier, pd))


@timethis
@collect_datasets
def q10(lineitem, orders, customer, nation, pd):
    exec_func(tpch_q10(lineitem, orders, customer, nation, pd))


@timethis
@collect_datasets
def q11(partsupp, supplier, nation, scale_factor, pd):
    exec_func(tpch_q11(partsupp, supplier, nation, scale_factor, pd))


@timethis
@collect_datasets
def q12(lineitem, orders, pd):
    exec_func(tpch_q12(lineitem, orders, pd))


@timethis
@collect_datasets
def q13(customer, orders, pd):
    exec_func(tpch_q13(customer, orders, pd))


@timethis
@collect_datasets
def q14(lineitem, part, pd):
    exec_func(tpch_q14(lineitem, part, pd))


@timethis
@collect_datasets
def q15(lineitem, supplier, pd):
    exec_func(tpch_q15(lineitem, supplier, pd))


@timethis
@collect_datasets
def q16(part, partsupp, supplier, pd):
    exec_func(tpch_q16(part, partsupp, supplier, pd))


@timethis
@collect_datasets
def q17(lineitem, part, pd):
    exec_func(tpch_q17(lineitem, part, pd))


@timethis
@collect_datasets
def q18(lineitem, orders, customer, pd):
    exec_func(tpch_q18(lineitem, orders, customer, pd))


@timethis
@collect_datasets
def q19(lineitem, part, pd):
    exec_func(tpch_q19(lineitem, part, pd))


@timethis
@collect_datasets
def q20(lineitem, part, nation, partsupp, supplier, pd):
    exec_func(tpch_q20(lineitem, part, nation, partsupp, supplier, pd))


@timethis
@collect_datasets
def q21(lineitem, orders, supplier, nation, pd):
    exec_func(tpch_q21(lineitem, orders, supplier, nation, pd))


@timethis
@collect_datasets
def q22(customer, orders, pd):
    exec_func(tpch_q22(customer, orders, pd))


def _load_args(query: int, root: str, scale_factor: float, backend):
    args = []
    for arg in _query_to_args[query]:
        if arg == "scale_factor":
            args.append(scale_factor)
        elif arg == "pd":
            args.append(backend)
        else:
            args.append(globals()[f"load_{arg}"](root, pd=backend))
    return args


def run_queries(root: str, queries: list[int], scale_factor: float, backend):
    if backend is bodo.pandas and bodo.dataframe_library_run_parallel:
        spawner.submit_func_to_workers(lambda: warnings.filterwarnings("ignore"), [])

    total_start = time.time()
    for query in queries:
        q = globals()[f"q{query:02}"]

        def query_func():
            q(*_load_args(query, root, scale_factor, backend))

        query_func = timethis(
            query_func, name=f"Q{query:02} Execution time (including read_parquet) (s):"
        )

        # Warm up run:
        query_func()

        # Second run for timing:
        query_func()
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
        default="bodo",
        help="Which backend to use, bodo or pandas.",
    )
    parser.add_argument(
        "--show_output",
        action="store_true",
        required=False,
        help="Whether to print the output.",
    )

    args = parser.parse_args()
    data_set = args.folder
    scale_factor = args.scale_factor
    backend = args.backend
    assert backend in ["bodo", "pandas", "pd"]

    global show_output
    show_output = args.show_output

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    warnings.filterwarnings("ignore")

    backend_module = bodo.pandas if backend == "bodo" else pd
    run_queries(
        data_set, queries=queries, scale_factor=scale_factor, backend=backend_module
    )


if __name__ == "__main__":
    print(f"Running TPC-H against pd v{pd.__version__}")
    main()
