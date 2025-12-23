import argparse
import time
import warnings
from collections.abc import Callable

import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import SparkSession


def load_lineitem(data_folder: str):
    print("Loading lineitem")
    data_path = data_folder + "/lineitem.pq"
    df = ps.read_parquet(data_path)
    df["L_SHIPDATE"] = ps.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = ps.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = ps.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    return df


def load_part(data_folder: str):
    print("Loading part")
    data_path = data_folder + "/part.pq"
    df = ps.read_parquet(data_path)
    print("Done loading part")
    return df


def load_orders(data_folder: str):
    print("Loading orders")
    data_path = data_folder + "/orders.pq"
    df = ps.read_parquet(data_path)
    df["O_ORDERDATE"] = ps.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    print("Done loading orders")
    return df


def load_customer(data_folder: str):
    print("Loading customer")
    data_path = data_folder + "/customer.pq"
    df = ps.read_parquet(data_path)
    print("Done loading customer")
    return df


def load_nation(data_folder: str):
    print("Loading nation")
    data_path = data_folder + "/nation.pq"
    df = ps.read_parquet(data_path)
    print("Done loading nation")
    return df


def load_region(data_folder: str):
    print("Loading region")
    data_path = data_folder + "/region.pq"
    df = ps.read_parquet(data_path)
    print("Done loading region")
    return df


def load_supplier(data_folder: str):
    print("Loading supplier")
    data_path = data_folder + "/supplier.pq"
    df = ps.read_parquet(data_path)
    print("Done loading supplier")
    return df


def load_partsupp(data_folder: str):
    print("Loading partsupp")
    data_path = data_folder + "/partsupp.pq"
    df = ps.read_parquet(data_path)
    print("Done loading partsupp")
    return df


def tpch_q02(data_folder: str, scale_factor: float = 1.0) -> pd.DataFrame:
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q2.py
    """
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)

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


def tpch_q03(data_folder: str, scale_factor: float = 1.0):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q3.py
    """
    customer = load_customer(data_folder)
    orders = load_orders(data_folder)
    lineitem = load_lineitem(data_folder)

    var1 = "HOUSEHOLD"
    var2 = pd.Timestamp("1995-03-04")

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


def tpch_q05(data_folder: str, scale_factor: float = 1.0):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q5.py
    """
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)

    var1 = "ASIA"
    var2 = pd.Timestamp("1996-01-01")
    var3 = pd.Timestamp("1997-01-01")

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


def tpch_q06(data_folder: str, scale_factor: float = 1.0):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q6.py
    """
    lineitem = load_lineitem(data_folder)

    var1 = pd.Timestamp("1996-01-01")
    var2 = pd.Timestamp("1997-01-01")
    var3 = 0.08
    var4 = 0.1
    var5 = 24

    filt = lineitem[(lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)]
    filt = filt[(filt["L_DISCOUNT"] >= var3) & (filt["L_DISCOUNT"] <= var4)]
    filt = filt[filt["L_QUANTITY"] < var5]
    result_value = (filt["L_EXTENDEDPRICE"] * filt["L_DISCOUNT"]).sum()
    result_df = pd.DataFrame({"REVENUE": [result_value]})

    return result_df


def tpch_q07(data_folder: str, scale_factor: float = 1.0):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q7.py
    """
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)

    var1 = "FRANCE"
    var2 = "GERMANY"
    var3 = pd.Timestamp("1995-01-01")
    var4 = pd.Timestamp("1997-01-01")

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
    total = ps.concat([df1, df2])

    total = total[(total["L_SHIPDATE"] >= var3) & (total["L_SHIPDATE"] < var4)]
    total["VOLUME"] = total["L_EXTENDEDPRICE"] * (1.0 - total["L_DISCOUNT"])
    total["L_YEAR"] = total["L_SHIPDATE"].dt.year

    gb = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"])
    agg = gb.agg(REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum"))
    agg = agg.reset_index()

    result_df = agg.sort_values(by=["SUPP_NATION", "CUST_NATION", "L_YEAR"])
    return result_df


def tpch_q08(data_folder: str, scale_factor: float = 1.0):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q8.py
    """
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    part = load_part(data_folder)
    nation = load_nation(data_folder)
    supplier = load_supplier(data_folder)
    customer = load_customer(data_folder)
    region = load_region(data_folder)

    var1 = "BRAZIL"
    var2 = "AMERICA"
    var3 = "ECONOMY ANODIZED STEEL"
    var4 = pd.Timestamp("1995-01-01")
    var5 = pd.Timestamp("1997-01-01")

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


def tpch_q09(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)

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

    gb = jn5.groupby(["NATION", "O_YEAR"])
    agg = gb.agg(SUM_PROFIT=pd.NamedAgg(column="AMOUNT", aggfunc="sum"))
    agg = agg.reset_index()
    agg["SUM_PROFIT"] = agg.SUM_PROFIT.round(2)
    result_df = agg.sort_values(by=["NATION", "O_YEAR"], ascending=[True, False])

    return result_df


def tpch_q10(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)

    var1 = pd.Timestamp("1994-11-01")
    var2 = pd.Timestamp("1995-02-01")

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
    )["REVENUE"].sum()
    agg = agg.reset_index()
    agg["REVENUE"] = agg.REVENUE.round(2)
    total = agg.sort_values("REVENUE", ascending=False)
    return total.head(20)


def tpch_q11(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)

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


def tpch_q14(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)

    var1 = pd.Timestamp("1994-03-01")
    var2 = var1 + pd.DateOffset(months=1)

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


def tpch_q15(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)

    var1 = pd.Timestamp("1996-01-01")
    var2 = var1 + pd.DateOffset(months=3)

    jn1 = lineitem[(lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)]

    jn1["REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])

    agg = jn1.groupby("L_SUPPKEY").agg(
        TOTAL_REVENUE=pd.NamedAgg(column="REVENUE", aggfunc="sum")
    )
    agg = agg.reset_index()
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


def tpch_q17(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)

    var1 = "Brand#23"
    var2 = "MED BOX"

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn1 = jn1[((jn1["P_BRAND"] == var1) & (jn1["P_CONTAINER"] == var2))]

    agg = jn1.groupby("L_PARTKEY").agg(
        L_QUANTITY_AVG=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )
    agg = agg.reset_index()

    jn4 = jn1.merge(agg, left_on="L_PARTKEY", right_on="L_PARTKEY", how="left")
    jn4 = jn4[jn4["L_QUANTITY"] < 0.2 * jn4["L_QUANTITY_AVG"]]
    total = jn4["L_EXTENDEDPRICE"].sum() / 7.0

    result_df = pd.DataFrame({"AVG_YEARLY": [round(total, 2)]})

    return result_df


def tpch_q18(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)

    var1 = 300

    agg1 = lineitem.groupby("L_ORDERKEY", as_index=False)["L_QUANTITY"].sum()
    filt = agg1[agg1.L_QUANTITY > var1]
    jn1 = filt.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    agg2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
    )["L_QUANTITY"].sum()
    total = agg2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    return total.head(100)


def tpch_q19(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)

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


def tpch_q20(data_folder: str, scale_factor: float = 1.0):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)

    var1 = pd.Timestamp("1996-01-01")
    var2 = pd.Timestamp("1997-01-01")
    var3 = "JORDAN"
    var4 = "azure"

    flineitem = lineitem[
        (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
    ]
    agg = flineitem.groupby(["L_SUPPKEY", "L_PARTKEY"]).agg(
        SUM_QUANTITY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum")
    )
    agg = agg.reset_index()
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


def run_query_single(query_func: Callable, data_folder: str, scale_factor: float = 1.0):
    res = query_func(data_folder, scale_factor)
    if not isinstance(res, pd.DataFrame):
        res = res.to_pandas()
    return res


def run_queries(data_folder: str, queries: list[int], scale_factor: float = 1.0):
    t1 = time.time()

    for query in queries:
        query_func = globals().get(f"tpch_q{query:02}")

        if query_func is None:
            print(f"Query {query:02} not implemented yet.")
            continue

        # Warm up run
        run_query_single(query_func, data_folder, scale_factor)

        t2 = time.time()
        run_query_single(query_func, data_folder, scale_factor)
        print(f"Query {query:02} took {time.time() - t2:.2f} seconds")

    print(f"Total time: {time.time() - t1:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        default="data/tpch-datagen/data",
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
    folder = args.folder
    scale_factor = args.scale_factor

    (
        SparkSession.builder.appName("SQL Queries with Spark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1,")
        .getOrCreate()
    )
    ps.set_option("compute.default_index_type", "distributed-sequence")

    queries = args.queries or list(range(1, 23))

    warnings.filterwarnings("ignore")

    run_queries(folder, queries, scale_factor)


if __name__ == "__main__":
    main()
