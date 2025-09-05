# Original from https://gist.githubusercontent.com/UranusSeven/55817bf0f304cc24f5eb63b2f1c3e2cd/raw/796dbd2fce6441821fc0b5bc51491edb49639c55/tpch.py
import argparse
import functools
import inspect
import time
from collections.abc import Callable

import pandas as pd

import bodo.pandas


@functools.lru_cache
def load_lineitem(data_folder: str, pd=bodo.pandas):
    print("Loading lineitem")
    data_path = data_folder + "/lineitem.pq"
    df = pd.read_parquet(data_path)
    df["L_SHIPDATE"] = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    print("Done loading lineitem")
    return df


@functools.lru_cache
def load_part(data_folder: str, pd=bodo.pandas):
    print("Loading part")
    data_path = data_folder + "/part.pq"
    df = pd.read_parquet(data_path)
    print("Done loading part")
    return df


@functools.lru_cache
def load_orders(data_folder: str, pd=bodo.pandas):
    print("Loading orders")
    data_path = data_folder + "/orders.pq"
    df = pd.read_parquet(data_path)
    df["O_ORDERDATE"] = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    print("Done loading orders")
    return df


@functools.lru_cache
def load_customer(data_folder: str, pd=bodo.pandas):
    print("Loading customer")
    data_path = data_folder + "/customer.pq"
    df = pd.read_parquet(data_path)
    print("Done loading customer")
    return df


@functools.lru_cache
def load_nation(data_folder: str, pd=bodo.pandas):
    print("Loading nation")
    data_path = data_folder + "/nation.pq"
    df = pd.read_parquet(data_path)
    print("Done loading nation")
    return df


@functools.lru_cache
def load_region(data_folder: str, pd=bodo.pandas):
    print("Loading region")
    data_path = data_folder + "/region.pq"
    df = pd.read_parquet(data_path)
    print("Done loading region")
    return df


@functools.lru_cache
def load_supplier(data_folder: str, pd=bodo.pandas):
    print("Loading supplier")
    data_path = data_folder + "/supplier.pq"
    df = pd.read_parquet(data_path)
    print("Done loading supplier")
    return df


@functools.lru_cache
def load_partsupp(data_folder: str, pd=bodo.pandas):
    print("Loading partsupp")
    data_path = data_folder + "/partsupp.pq"
    df = pd.read_parquet(data_path)
    print("Done loading partsupp")
    return df


def timethis(q: Callable):
    @functools.wraps(q)
    def wrapped(*args, **kwargs):
        t = time.time()
        q(*args, **kwargs)
        print(f"{q.__name__.upper()} Execution time (s): {time.time() - t:f}")

    return wrapped


_query_to_args: dict[int, list[str]] = {}


def collect_datasets(func: Callable):
    _query_to_args[int(func.__name__[1:])] = list(inspect.signature(func).parameters)
    return func


def tpch_q01(lineitem, pd=bodo.pandas):
    date = pd.Timestamp("1998-09-02")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "L_DISCOUNT",
            "L_TAX",
            "L_RETURNFLAG",
            "L_LINESTATUS",
            "L_SHIPDATE",
            "L_ORDERKEY",
        ],
    ]
    sel = lineitem_filtered.L_SHIPDATE <= date
    lineitem_filtered = lineitem_filtered[sel]
    lineitem_filtered["AVG_QTY"] = lineitem_filtered.L_QUANTITY
    lineitem_filtered["AVG_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE
    lineitem_filtered["DISC_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE * (
        1 - lineitem_filtered.L_DISCOUNT
    )
    lineitem_filtered["CHARGE"] = (
        lineitem_filtered.L_EXTENDEDPRICE
        * (1 - lineitem_filtered.L_DISCOUNT)
        * (1 + lineitem_filtered.L_TAX)
    )
    gb = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)[
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "DISC_PRICE",
            "CHARGE",
            "AVG_QTY",
            "AVG_PRICE",
            "L_DISCOUNT",
            "L_ORDERKEY",
        ]
    ]
    total = gb.agg(
        {
            "L_QUANTITY": "sum",
            "L_EXTENDEDPRICE": "sum",
            "DISC_PRICE": "sum",
            "CHARGE": "sum",
            "AVG_QTY": "mean",
            "AVG_PRICE": "mean",
            "L_DISCOUNT": "mean",
            "L_ORDERKEY": "count",
        }
    )
    total.columns = [
        "L_RETURNFLAG",
        "L_LINESTATUS",
        "SUM_QTY",
        "SUM_BASE_PRICE",
        "SUM_DISC_PRICE",
        "SUM_CHARGE",
        "AVG_QTY",
        "AVG_PRICE",
        "AVG_DISC",
        "COUNT_ORDER",
    ]
    total = total.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    return total


def tpch_q02(part, partsupp, supplier, nation, region, pd=bodo.pandas):
    nation_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME", "N_REGIONKEY"]]
    region_filtered = region[(region["R_NAME"] == "EUROPE")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    r_n_merged = nation_filtered.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    r_n_merged = r_n_merged.loc[:, ["N_NATIONKEY", "N_NAME"]]
    supplier_filtered = supplier.loc[
        :,
        [
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_NATIONKEY",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    s_r_n_merged = r_n_merged.merge(
        supplier_filtered, left_on="N_NATIONKEY", right_on="S_NATIONKEY", how="inner"
    )
    s_r_n_merged = s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY", "PS_SUPPLYCOST"]]
    ps_s_r_n_merged = s_r_n_merged.merge(
        partsupp_filtered, left_on="S_SUPPKEY", right_on="PS_SUPPKEY", how="inner"
    )
    ps_s_r_n_merged = ps_s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_PARTKEY",
            "PS_SUPPLYCOST",
        ],
    ]
    part_filtered = part.loc[:, ["P_PARTKEY", "P_MFGR", "P_SIZE", "P_TYPE"]]
    part_filtered = part_filtered[
        (part_filtered["P_SIZE"] == 15)
        & (part_filtered["P_TYPE"].str.endswith("BRASS"))
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_MFGR"]]
    merged_df = part_filtered.merge(
        ps_s_r_n_merged, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    merged_df = merged_df.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_SUPPLYCOST",
            "P_PARTKEY",
            "P_MFGR",
        ],
    ]
    min_values = merged_df.groupby("P_PARTKEY", as_index=False, sort=False)[
        "PS_SUPPLYCOST"
    ].min()
    min_values.columns = ["P_PARTKEY_CPY", "MIN_SUPPLYCOST"]
    merged_df = merged_df.merge(
        min_values,
        left_on=["P_PARTKEY", "PS_SUPPLYCOST"],
        right_on=["P_PARTKEY_CPY", "MIN_SUPPLYCOST"],
        how="inner",
    )
    total = merged_df.loc[
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
    total = total.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    return total


def tpch_q03(lineitem, orders, customer, pd=bodo.pandas):
    date = pd.Timestamp("1995-03-04")
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    orders_filtered = orders.loc[
        :, ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY"]
    ]
    customer_filtered = customer.loc[:, ["C_MKTSEGMENT", "C_CUSTKEY"]]
    lsel = lineitem_filtered.L_SHIPDATE > date
    osel = orders_filtered.O_ORDERDATE < date
    csel = customer_filtered.C_MKTSEGMENT == "HOUSEHOLD"
    flineitem = lineitem_filtered[lsel]
    forders = orders_filtered[osel]
    fcustomer = customer_filtered[csel]
    jn1 = fcustomer.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn2["REVENUE"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
    total = (
        jn2.groupby(
            ["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False, sort=False
        )["REVENUE"]
        .sum()
        .sort_values(["REVENUE"], ascending=False)
    )
    res = total.loc[:, ["L_ORDERKEY", "REVENUE", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    return res.head(10)


def tpch_q04(lineitem, orders, pd=bodo.pandas):
    date1 = pd.Timestamp("1993-11-01")
    date2 = pd.Timestamp("1993-08-01")
    lsel = lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE
    osel = (orders.O_ORDERDATE < date1) & (orders.O_ORDERDATE >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = (
        jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"]
        .count()
        .sort_values(["O_ORDERPRIORITY"])
    )
    total.columns = ["O_ORDERPRIORITY", "ORDER_COUNT"]
    return total


def tpch_q05(lineitem, orders, customer, nation, region, supplier, pd=bodo.pandas):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    rsel = region.R_NAME == "ASIA"
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    forders = orders[osel]
    fregion = region[rsel]
    jn1 = fregion.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
    jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
    jn3 = jn2.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn5 = supplier.merge(
        jn4, left_on=["S_SUPPKEY", "S_NATIONKEY"], right_on=["L_SUPPKEY", "N_NATIONKEY"]
    )
    jn5["REVENUE"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME", as_index=False, sort=False)["REVENUE"].sum()
    total = gb.sort_values("REVENUE", ascending=False)
    return total


def tpch_q06(lineitem, pd=bodo.pandas):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    lineitem_filtered = lineitem.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    sel = (
        (lineitem_filtered.L_SHIPDATE >= date1)
        & (lineitem_filtered.L_SHIPDATE < date2)
        & (lineitem_filtered.L_DISCOUNT >= 0.08)
        & (lineitem_filtered.L_DISCOUNT <= 0.1)
        & (lineitem_filtered.L_QUANTITY < 24)
    )
    flineitem = lineitem_filtered[sel]
    total = (flineitem.L_EXTENDEDPRICE * flineitem.L_DISCOUNT).sum()
    return total


def tpch_q07(lineitem, supplier, orders, customer, nation, pd=bodo.pandas):
    """This version is faster than q07_old. Keeping the old one for reference"""
    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= pd.Timestamp("1995-01-01"))
        & (lineitem["L_SHIPDATE"] < pd.Timestamp("1997-01-01"))
    ]
    lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].dt.year
    lineitem_filtered["VOLUME"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_YEAR", "VOLUME"]
    ]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    n1 = nation[(nation["N_NAME"] == "FRANCE")].loc[:, ["N_NATIONKEY", "N_NAME"]]
    n2 = nation[(nation["N_NAME"] == "GERMANY")].loc[:, ["N_NATIONKEY", "N_NAME"]]

    # ----- do nation 1 -----
    N1_C = customer_filtered.merge(
        n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_C = N1_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N1_C_O = N1_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N1_C_O = N1_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N2_S = supplier_filtered.merge(
        n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_S = N2_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N2_S_L = N2_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N2_S_L = N2_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total1 = N1_C_O.merge(
        N2_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total1 = total1.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # ----- do nation 2 -----
    N2_C = customer_filtered.merge(
        n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_C = N2_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N2_C_O = N2_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N2_C_O = N2_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N1_S = supplier_filtered.merge(
        n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_S = N1_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N1_S_L = N1_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N1_S_L = N1_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total2 = N2_C_O.merge(
        N1_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total2 = total2.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # concat results
    total = pd.concat([total1, total2])

    total = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False).agg(
        REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum")
    )
    total = total.sort_values(
        by=["SUPP_NATION", "CUST_NATION", "L_YEAR"], ascending=[True, True, True]
    )
    return total


def tpch_q08(
    part, lineitem, supplier, orders, customer, nation, region, pd=bodo.pandas
):
    part_filtered = part[(part["P_TYPE"] == "ECONOMY ANODIZED STEEL")]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY"]]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_SUPPKEY", "L_ORDERKEY"]]
    lineitem_filtered["VOLUME"] = lineitem["L_EXTENDEDPRICE"] * (
        1.0 - lineitem["L_DISCOUNT"]
    )
    total = part_filtered.merge(
        lineitem_filtered, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY", "L_ORDERKEY", "VOLUME"]]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["L_ORDERKEY", "VOLUME", "S_NATIONKEY"]]
    orders_filtered = orders[
        (orders["O_ORDERDATE"] >= pd.Timestamp("1995-01-01"))
        & (orders["O_ORDERDATE"] < pd.Timestamp("1997-01-01"))
    ]
    orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].dt.year
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_YEAR"]]
    total = total.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_CUSTKEY", "O_YEAR"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    total = total.merge(
        customer_filtered, left_on="O_CUSTKEY", right_on="C_CUSTKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "C_NATIONKEY"]]
    n1_filtered = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
    n2_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME"]].rename(
        columns={"N_NAME": "NATION"}
    )
    total = total.merge(
        n1_filtered, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "N_REGIONKEY"]]
    total = total.merge(
        n2_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "N_REGIONKEY", "NATION"]]
    region_filtered = region[(region["R_NAME"] == "AMERICA")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    total = total.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "NATION"]]

    def udf(df):
        demonimator = df["VOLUME"].sum()
        df = df[df["NATION"] == "BRAZIL"]
        numerator = df["VOLUME"].sum()
        return round(numerator / demonimator, 2)

    total = total.groupby("O_YEAR", as_index=False).apply(udf)
    total.columns = ["O_YEAR", "MKT_SHARE"]
    total = total.sort_values(by=["O_YEAR"], ascending=[True])
    return total


def tpch_q09(lineitem, orders, part, nation, partsupp, supplier, pd=bodo.pandas):
    psel = part.P_NAME.str.contains("ghost")
    fpart = part[psel]
    jn1 = lineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    jn4 = partsupp.merge(
        jn3, left_on=["PS_PARTKEY", "PS_SUPPKEY"], right_on=["L_PARTKEY", "L_SUPPKEY"]
    )
    jn5 = jn4.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1 - jn5.L_DISCOUNT) - (
        (1 * jn5.PS_SUPPLYCOST) * jn5.L_QUANTITY
    )
    jn5["O_YEAR"] = jn5.O_ORDERDATE.dt.year
    gb = jn5.groupby(["N_NAME", "O_YEAR"], as_index=False, sort=False)["TMP"].sum()
    total = gb.sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
    total.columns = ["NATION", "O_YEAR", "SUM_PROFIT"]
    return total


def tpch_q10(lineitem, orders, customer, nation, pd=bodo.pandas):
    date1 = pd.Timestamp("1994-11-01")
    date2 = pd.Timestamp("1995-02-01")
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    lsel = lineitem.L_RETURNFLAG == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["TMP"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    gb = jn3.groupby(
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
    )["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)
    total = total.rename(columns={"TMP": "REVENUE"})
    return total.head(20)


def tpch_q11(partsupp, supplier, nation, scale_factor=1, pd=bodo.pandas):
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
    )
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    ps_supp_merge = ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
    nation_filtered = nation[(nation["N_NAME"] == "GERMANY")]
    nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001
    total = ps_supp_n_merge.groupby(["PS_PARTKEY"], as_index=False, sort=False).agg(
        VALUE=pd.NamedAgg(column="TOTAL_COST", aggfunc="sum")
    )
    total = total[total["VALUE"] > sum_val]
    total = total.sort_values("VALUE", ascending=False)
    return total


def tpch_q12(lineitem, orders, pd=bodo.pandas):
    date1 = pd.Timestamp("1994-01-01")
    date2 = pd.Timestamp("1995-01-01")
    sel = (
        (lineitem.L_RECEIPTDATE < date2)
        & (lineitem.L_COMMITDATE < date2)
        & (lineitem.L_SHIPDATE < date2)
        & (lineitem.L_SHIPDATE < lineitem.L_COMMITDATE)
        & (lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE)
        & (lineitem.L_RECEIPTDATE >= date1)
        & ((lineitem.L_SHIPMODE == "MAIL") | (lineitem.L_SHIPMODE == "SHIP"))
    )
    flineitem = lineitem[sel]
    jn = flineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    total = jn.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
    # total = total.reset_index()  # reset index to keep consistency with pandas
    total = total.sort_values("L_SHIPMODE")
    return total


def tpch_q13(customer, orders, pd=bodo.pandas):
    customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
    orders_filtered = orders[
        ~orders["O_COMMENT"].str.contains(r"special[\S|\s]*requests")
    ]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]
    count_df = c_o_merged.groupby(["C_CUSTKEY"], as_index=False, sort=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    total = count_df.groupby(["C_COUNT"], as_index=False, sort=False).size()
    total.columns = ["C_COUNT", "CUSTDIST"]
    total = total.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])
    return total


def tpch_q14(lineitem, part, pd=bodo.pandas):
    startDate = pd.Timestamp("1994-03-01")
    endDate = pd.Timestamp("1994-04-01")
    p_type_like = "PROMO"
    part_filtered = part.loc[:, ["P_PARTKEY", "P_TYPE"]]
    lineitem_filtered = lineitem.loc[
        :, ["L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE", "L_PARTKEY"]
    ]
    sel = (lineitem_filtered.L_SHIPDATE >= startDate) & (
        lineitem_filtered.L_SHIPDATE < endDate
    )
    flineitem = lineitem_filtered[sel]
    jn = flineitem.merge(part_filtered, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn["TMP"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
    total = jn[jn.P_TYPE.str.startswith(p_type_like)].TMP.sum() * 100 / jn.TMP.sum()

    result_df = pd.DataFrame({"PROMO_REVENUE": [total]})
    return result_df


def tpch_q15(lineitem, supplier, pd=bodo.pandas):
    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= pd.Timestamp("1996-01-01"))
        & (
            lineitem["L_SHIPDATE"]
            < (pd.Timestamp("1996-01-01") + pd.DateOffset(months=3))
        )
    ]
    lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[:, ["L_SUPPKEY", "REVENUE_PARTS"]]
    revenue_table = (
        lineitem_filtered.groupby("L_SUPPKEY", as_index=False, sort=False)
        .agg(TOTAL_REVENUE=pd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))
        .rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})
    )
    max_revenue = revenue_table["TOTAL_REVENUE"].max()
    revenue_table = revenue_table[revenue_table["TOTAL_REVENUE"] == max_revenue]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE"]]
    total = supplier_filtered.merge(
        revenue_table, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
    )
    total = total.loc[
        :, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
    ]
    return total


def tpch_q16(part, partsupp, supplier, pd=bodo.pandas):
    part_filtered = part[
        (part["P_BRAND"] != "Brand#45")
        & (~part["P_TYPE"].str.contains("^MEDIUM POLISHED"))
        & part["P_SIZE"].isin([49, 14, 23, 45, 19, 3, 36, 9])
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_BRAND", "P_TYPE", "P_SIZE"]]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    total = part_filtered.merge(
        partsupp_filtered, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    supplier_filtered = supplier[
        supplier["S_COMMENT"].str.contains(r"Customer(\S|\s)*Complaints")
    ]
    supplier_filtered = supplier_filtered.loc[:, ["S_SUPPKEY"]].drop_duplicates()
    # left merge to select only PS_SUPPKEY values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left"
    )
    total = total[total["S_SUPPKEY"].isna()]
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    total = total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False, sort=False)[
        "PS_SUPPKEY"
    ].nunique()
    total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
    total = total.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )
    return total


def tpch_q17(lineitem, part, pd=bodo.pandas):
    left = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY", "L_EXTENDEDPRICE"]]
    right = part[((part["P_BRAND"] == "Brand#23") & (part["P_CONTAINER"] == "MED BOX"))]
    right = right.loc[:, ["P_PARTKEY"]]
    line_part_merge = left.merge(
        right, left_on="L_PARTKEY", right_on="P_PARTKEY", how="inner"
    )
    line_part_merge = line_part_merge.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "P_PARTKEY"]
    ]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY"]]
    lineitem_avg = lineitem_filtered.groupby(
        ["L_PARTKEY"], as_index=False, sort=False
    ).agg(avg=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean"))
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="left"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = pd.DataFrame({"AVG_YEARLY": [total["L_EXTENDEDPRICE"].sum() / 7.0]})
    return total


def tpch_q18(lineitem, orders, customer, pd=bodo.pandas):
    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False, sort=False)["L_QUANTITY"].sum()
    fgb1 = gb1[gb1.L_QUANTITY > 300]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
        sort=False,
    )["L_QUANTITY"].sum()
    total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    return total.head(100)


def tpch_q19(lineitem, part, pd=bodo.pandas):
    Brand31 = "Brand#31"
    Brand43 = "Brand#43"
    SMBOX = "SM BOX"
    SMCASE = "SM CASE"
    SMPACK = "SM PACK"
    SMPKG = "SM PKG"
    MEDBAG = "MED BAG"
    MEDBOX = "MED BOX"
    MEDPACK = "MED PACK"
    MEDPKG = "MED PKG"
    LGBOX = "LG BOX"
    LGCASE = "LG CASE"
    LGPACK = "LG PACK"
    LGPKG = "LG PKG"
    DELIVERINPERSON = "DELIVER IN PERSON"
    AIR = "AIR"
    AIRREG = "AIRREG"
    lsel = (
        (
            ((lineitem.L_QUANTITY <= 36) & (lineitem.L_QUANTITY >= 26))
            | ((lineitem.L_QUANTITY <= 25) & (lineitem.L_QUANTITY >= 15))
            | ((lineitem.L_QUANTITY <= 14) & (lineitem.L_QUANTITY >= 4))
        )
        & (lineitem.L_SHIPINSTRUCT == DELIVERINPERSON)
        & ((lineitem.L_SHIPMODE == AIR) | (lineitem.L_SHIPMODE == AIRREG))
    )
    psel = (part.P_SIZE >= 1) & (
        (
            (part.P_SIZE <= 5)
            & (part.P_BRAND == Brand31)
            & (part.P_CONTAINER.isin([SMBOX, SMCASE, SMPACK, SMPKG]))
        )
        | (
            (part.P_SIZE <= 10)
            & (part.P_BRAND == Brand43)
            & (part.P_CONTAINER.isin([MEDBAG, MEDBOX, MEDPACK, MEDPKG]))
        )
        | (
            (part.P_SIZE <= 15)
            & (part.P_BRAND == Brand43)
            & (part.P_CONTAINER.isin([LGBOX, LGCASE, LGPACK, LGPKG]))
        )
    )
    flineitem = lineitem[lsel]
    fpart = part[psel]
    jn = flineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jnsel = (
        (
            (jn.P_BRAND == Brand31)
            & (jn.P_CONTAINER.isin([SMBOX, SMCASE, SMPACK, SMPKG]))
            & (jn.L_QUANTITY >= 4)
            & (jn.L_QUANTITY <= 14)
            & (jn.P_SIZE <= 5)
        )
        | (
            (jn.P_BRAND == Brand43)
            & (jn.P_CONTAINER.isin([MEDBAG, MEDBOX, MEDPACK, MEDPKG]))
            & (jn.L_QUANTITY >= 15)
            & (jn.L_QUANTITY <= 25)
            & (jn.P_SIZE <= 10)
        )
        | (
            (jn.P_BRAND == Brand43)
            & (jn.P_CONTAINER.isin([LGBOX, LGCASE, LGPACK, LGPKG]))
            & (jn.L_QUANTITY >= 26)
            & (jn.L_QUANTITY <= 36)
            & (jn.P_SIZE <= 15)
        )
    )
    jn = jn[jnsel]
    total = (jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)).sum()
    return total


def tpch_q20(lineitem, part, nation, partsupp, supplier, pd=bodo.pandas):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    psel = part.P_NAME.str.startswith("azure")
    nsel = nation.N_NAME == "JORDAN"
    lsel = (lineitem.L_SHIPDATE >= date1) & (lineitem.L_SHIPDATE < date2)
    fpart = part[psel]
    fnation = nation[nsel]
    flineitem = lineitem[lsel]
    jn1 = fpart.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
    jn2 = jn1.merge(
        flineitem,
        left_on=["PS_PARTKEY", "PS_SUPPKEY"],
        right_on=["L_PARTKEY", "L_SUPPKEY"],
    )
    gb = jn2.groupby(
        ["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"], as_index=False, sort=False
    )["L_QUANTITY"].sum()
    gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
    jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
    total = jn4.sort_values("S_NAME").drop_duplicates()
    return total


def tpch_q21(lineitem, orders, supplier, nation, pd=bodo.pandas):
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
    ]

    # Keep all rows that have another row in linetiem with the same orderkey and different suppkey
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
        .groupby("L_ORDERKEY", as_index=False, sort=False)["L_SUPPKEY"]
        .nunique()
    )
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] > 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Keep all rows that have l_receiptdate > l_commitdate
    lineitem_filtered = lineitem_filtered[
        lineitem_filtered["L_RECEIPTDATE"] > lineitem_filtered["L_COMMITDATE"]
    ]
    lineitem_filtered = lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]

    # Merge Filter + Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    # Not Exists: Check the exists condition isn't still satisfied on the output.
    lineitem_orderkeys = lineitem_filtered.groupby(
        "L_ORDERKEY", as_index=False, sort=False
    )["L_SUPPKEY"].nunique()
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] == 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Merge Filter + Not Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    orders_filtered = orders.loc[:, ["O_ORDERSTATUS", "O_ORDERKEY"]]
    orders_filtered = orders_filtered[orders_filtered["O_ORDERSTATUS"] == "F"]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY"]]
    total = lineitem_filtered.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY"]]

    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY", "S_NAME"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["S_NATIONKEY", "S_NAME"]]
    nation_filtered = nation.loc[:, ["N_NAME", "N_NATIONKEY"]]
    nation_filtered = nation_filtered[nation_filtered["N_NAME"] == "SAUDI ARABIA"]
    total = total.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["S_NAME"]]
    total = total.groupby("S_NAME", as_index=False, sort=False).size()
    total.columns = ["S_NAME", "NUMWAIT"]
    total = total.sort_values(by=["NUMWAIT", "S_NAME"], ascending=[False, True])
    return total


def tpch_q22(customer, orders, pd=bodo.pandas):
    customer_filtered = customer.loc[:, ["C_ACCTBAL", "C_CUSTKEY"]]
    customer_filtered["CNTRYCODE"] = customer["C_PHONE"].str.slice(0, 2)
    customer_filtered = customer_filtered[
        (customer["C_ACCTBAL"] > 0.00)
        & customer_filtered["CNTRYCODE"].isin(
            ["13", "31", "23", "29", "30", "18", "17"]
        )
    ]
    avg_value = customer_filtered["C_ACCTBAL"].mean()
    customer_filtered = customer_filtered[customer_filtered["C_ACCTBAL"] > avg_value]
    # Select only the keys that don't match by performing a left join and only selecting columns with an na value
    orders_filtered = orders.loc[:, ["O_CUSTKEY"]].drop_duplicates()
    customer_keys = customer_filtered.loc[:, ["C_CUSTKEY"]].drop_duplicates()
    customer_selected = customer_keys.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    customer_selected = customer_selected[customer_selected["O_CUSTKEY"].isna()]
    customer_selected = customer_selected.loc[:, ["C_CUSTKEY"]]
    customer_selected = customer_selected.merge(
        customer_filtered, on="C_CUSTKEY", how="inner"
    )
    customer_selected = customer_selected.loc[:, ["CNTRYCODE", "C_ACCTBAL"]]
    total = customer_selected.groupby(["CNTRYCODE"], as_index=False, sort=False).agg(
        ["size", "sum"]
    )
    total.columns = ["CNTRYCODE", "NUMCUST", "TOTACCTBAL"]
    total = total.sort_values(by=["CNTRYCODE"], ascending=[True])
    return total


def new_tpch_q01(lineitem, pd=bodo.pandas):
    """This query reports the amount of business that was billed, shipped, and returned.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/dask/q1.py

    SQL:
    select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
    from
        lineitem
    where
        l_shipdate <= date '1998-12-01' - interval ':1' day
    group by
        l_returnflag,
        l_linestatus
    order by
        l_returnflag,
        l_linestatus
    """
    var1 = pd.Timestamp("1998-09-02")
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


def new_tpch_q02(part, partsupp, supplier, nation, region, pd=bodo.pandas):
    """This query finds which supplier should be selected to place an order for a given part in a given region.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q2.py

    SQL:
    select
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
    from
        part,
        supplier,
        partsupp,
        nation,
        region
    where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and p_size = :1
        and p_type like '%:2'
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = ':3'
        and ps_supplycost = (
            select
                min(ps_supplycost)
            from
                partsupp,
                supplier,
                nation,
                region
            where
                p_partkey = ps_partkey
                and s_suppkey = ps_suppkey
                and s_nationkey = n_nationkey
                and n_regionkey = r_regionkey
                and r_name = ':3'
        )
    order by
        s_acctbal desc,
        n_name,
        s_name,
        p_partkey
    LIMIT 100;
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


def new_tpch_q03(lineitem, orders, customer, pd=bodo.pandas):
    """This query retrieves the 10 unshipped orders with the highest value.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q3.py

    SQL:
    select
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        o_orderdate,
        o_shippriority
    from
        customer,
        orders,
        lineitem
    where
        c_mktsegment = ':1'
        and c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate < date ':2'
        and l_shipdate > date ':2'
    group by
        l_orderkey,
        o_orderdate,
        o_shippriority
    order by
        revenue desc,
        o_orderdate
    LIMIT 10;
    """
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


# TODO: open an issue
def new_tpch_q04(lineitem, orders, pd=bodo.pandas):
    """This query determines how well the order priority system is working and gives an assessment of customer satisfaction.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q4.py

    SQL:
    select
        o_orderpriority,
        count(*) as order_count
    from
        orders
    where
        o_orderdate >= date ':1'
        and o_orderdate < date ':1' + interval '3' month
        and exists (
            select
                *
            from
                lineitem
            where
                l_orderkey = o_orderkey
                and l_commitdate < l_receiptdate
        )
    group by
        o_orderpriority
    order by
        o_orderpriority
    LIMIT 1;
    """
    var1 = pd.Timestamp("1993-08-01")
    var2 = pd.Timestamp("1993-11-01")

    jn = lineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")

    jn = jn[(jn["O_ORDERDATE"] >= var1) & (jn["O_ORDERDATE"] < var2)]
    jn = jn[jn["L_COMMITDATE"] < jn["L_RECEIPTDATE"]]

    jn = jn.drop_duplicates(subset=["O_ORDERPRIORITY", "L_ORDERKEY"])

    gb = jn.groupby("O_ORDERPRIORITY", as_index=False)
    agg = gb.agg(ORDER_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count"))

    result_df = agg.sort_values(["O_ORDERPRIORITY"])

    return result_df


def new_tpch_q05(lineitem, orders, customer, nation, region, supplier, pd=bodo.pandas):
    """This query lists the revenue volume done through local suppliers.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q5.py

    SQL:
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
        and r_name = ':1'
        and o_orderdate >= date ':2'
        and o_orderdate < date ':2' + interval '1' year
    group by
        n_name
    order by
        revenue desc
    LIMIT 1;
    """
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


def new_tpch_q06(lineitem, pd=bodo.pandas):
    """This query quantifies the amount of revenue increase that would have resulted from eliminating certain company wide discounts in a given percentage range in a given year. Asking this type of "what if" query can be used to look
    for ways to increase revenues.

    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q6.py

    select
        sum(l_extendedprice * l_discount) as revenue
    from
        lineitem
    where
        l_shipdate >= date ':1'
        and l_shipdate < date ':1' + interval '1' year
        and l_discount between :2 - 0.01 and :2 + 0.01
        and l_quantity < :3
    LIMIT 1;
    """
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


def new_tpch_q07(lineitem, supplier, orders, customer, nation, pd=bodo.pandas):
    """
    Adapted Pandas code from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q7.py

    SQL:
    select
        supp_nation,
        cust_nation,
        l_year,
        sum(volume) as revenue
    from
        (
            select
                n1.n_name as supp_nation,
                n2.n_name as cust_nation,
                extract(year from l_shipdate) as l_year,
                l_extendedprice * (1 - l_discount) as volume
            from
                supplier,
                lineitem,
                orders,
                customer,
                nation n1,
                nation n2
            where
                s_suppkey = l_suppkey
                and o_orderkey = l_orderkey
                and c_custkey = o_custkey
                and s_nationkey = n1.n_nationkey
                and c_nationkey = n2.n_nationkey
                and (
                    (n1.n_name = ':1' and n2.n_name = ':2')
                    or (n1.n_name = ':2' and n2.n_name = ':1')
                )
                and l_shipdate between date '1995-01-01' and date '1996-12-31'
        ) as shipping
    group by
        supp_nation,
        cust_nation,
        l_year
    order by
        supp_nation,
        cust_nation,
        l_year
    LIMIT 1;
    """

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
    total = pd.concat([df1, df2])

    total = total[(total["L_SHIPDATE"] >= var3) & (total["L_SHIPDATE"] < var4)]
    total["VOLUME"] = total["L_EXTENDEDPRICE"] * (1.0 - total["L_DISCOUNT"])
    total["L_YEAR"] = total["L_SHIPDATE"].dt.year

    gb = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False)
    agg = gb.agg(REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum"))

    result_df = agg.sort_values(by=["SUPP_NATION", "CUST_NATION", "L_YEAR"])

    return result_df


def new_tpch_q08(
    part, lineitem, supplier, orders, customer, nation, region, pd=bodo.pandas
):
    """
    Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q8.py

    SQL:
    select
        o_year,
        sum(case
            when nation = ':1' then volume
            else 0
        end) / sum(volume) as mkt_share
    from
        (
            select
                extract(year from o_orderdate) as o_year,
                l_extendedprice * (1 - l_discount) as volume,
                n2.n_name as nation
            from
                part,
                supplier,
                lineitem,
                orders,
                customer,
                nation n1,
                nation n2,
                region
            where
                p_partkey = l_partkey
                and s_suppkey = l_suppkey
                and l_orderkey = o_orderkey
                and o_custkey = c_custkey
                and c_nationkey = n1.n_nationkey
                and n1.n_regionkey = r_regionkey
                and r_name = ':2'
                and s_nationkey = n2.n_nationkey
                and o_orderdate between date '1995-01-01' and date '1996-12-31'
                and p_type = ':3'
        ) as all_nations
    group by
        o_year
    order by
        o_year
    LIMIT 1;
    """
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

    def udf(df):
        demonimator = df["VOLUME"].sum()
        df = df[df["NATION"] == var1]
        numerator = df["VOLUME"].sum()
        return round(numerator / demonimator, 2)

    gb = jn7.groupby("O_YEAR", as_index=False)
    agg = gb.apply(udf, include_groups=False)
    agg.columns = ["O_YEAR", "MKT_SHARE"]
    result_df = agg.sort_values("O_YEAR")

    return result_df


def new_tpch_q09(lineitem, orders, part, nation, partsupp, supplier, pd=bodo.pandas):
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
    result_df = agg.sort_values(by=["NATION", "O_YEAR"], ascending=[True, False])

    return result_df


def new_tpch_q10(lineitem, orders, customer, nation, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1994-11-01")
    var2 = pd.Timestamp("1995-02-01")

    osel = (orders.O_ORDERDATE >= var1) & (orders.O_ORDERDATE < var2)
    lsel = lineitem.L_RETURNFLAG == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["REVENUE"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    gb = jn3.groupby(
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
    total = gb.sort_values("REVENUE", ascending=False)
    return total.head(20)


def new_tpch_q11(partsupp, supplier, nation, scale_factor=1.0, pd=bodo.pandas):
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
    rounded = filt
    result_df = rounded.sort_values(by="VALUE", ascending=False)

    return result_df


def new_tpch_q12(lineitem, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1994-01-01")
    var2 = pd.Timestamp("1995-01-01")

    jn1 = orders.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn2 = jn1[
        (jn1["L_SHIPMODE"].isin(("MAIL", "SHIP")))
        & (jn1["L_COMMITDATE"] < jn1["L_RECEIPTDATE"])
        & (jn1["L_SHIPDATE"] < jn1["L_COMMITDATE"])
        & (jn1["L_RECEIPTDATE"] >= var1)
        & (jn1["L_RECEIPTDATE"] < var2)
    ]

    # Bodo change: convert .where/agg part to use agg with UDF
    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    gb = jn2.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
    result_df = gb.sort_values("L_SHIPMODE")

    return result_df


def new_tpch_q13(customer, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "special"
    var2 = "requests"

    orders = orders[~orders["O_COMMENT"].str.contains(f"{var1}.*{var2}")]

    jn1 = customer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left")

    gb1 = jn1.groupby("C_CUSTKEY", as_index=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    subquery = gb1[["C_CUSTKEY", "C_COUNT"]]

    gb2 = subquery.groupby("C_COUNT", as_index=False).agg(
        CUSTDIST=pd.NamedAgg(column="C_CUSTKEY", aggfunc="size")
    )

    result_df = gb2.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])

    return result_df


# TODO: support where
def new_tpch_q14(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1994-03-01")
    var2 = var1 + pd.DateOffset(months=1)

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")

    jn2 = jn1[(jn1["L_SHIPDATE"] >= var1) & (jn1["L_SHIPDATE"] < var2)]

    # Promo revenue by line; CASE clause
    jn2["PROMO_REVENUE"] = jn2["L_EXTENDEDPRICE"] * (1 - jn2["L_DISCOUNT"])
    mask = jn2["P_TYPE"].str.match("PROMO*")
    jn2["PROMO_REVENUE"] = jn2["PROMO_REVENUE"].where(mask, 0.00)

    total_promo_revenue = jn2["PROMO_REVENUE"].sum()
    total_revenue = (jn2["L_EXTENDEDPRICE"] * (1 - jn2["L_DISCOUNT"])).sum()

    # aggregate promo revenue calculation
    ratio = 100.00 * total_promo_revenue / total_revenue
    result_df = pd.DataFrame({"PROMO_REVENUE": [ratio]})

    return result_df


def new_tpch_q15(lineitem, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1996-01-01")
    var2 = var1 + pd.DateOffset(months=3)

    jn1 = lineitem[(lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)]

    jn1["REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])

    gb = jn1.groupby("L_SUPPKEY", as_index=False).agg(
        TOTAL_REVENUE=pd.NamedAgg(column="REVENUE", aggfunc="sum")
    )
    revenue = gb.rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})

    jn2 = supplier.merge(
        revenue, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
    )

    max_revenue = revenue["TOTAL_REVENUE"].max()

    jn3 = jn2[jn2["TOTAL_REVENUE"] == max_revenue]

    result_df = jn3[
        ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
    ].sort_values(by="S_SUPPKEY")

    return result_df


# TODO fix:
# Unsupported expression type in projection 13 (NOT #[8.0])
def new_tpch_q16(part, partsupp, supplier, pd=bodo.pandas):
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
    jn3 = jn2[
        (jn2["P_BRAND"] != var1)
        & (~jn2["P_TYPE"].str.startswith("MEDIUM POLISHED"))
        & (jn2["P_SIZE"].isin((49, 14, 23, 45, 19, 3, 36, 9)))
    ]

    gb = jn3.groupby(by=["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)[
        "PS_SUPPKEY"
    ].nunique()
    agg = gb.rename(columns={"PS_SUPPKEY": "SUPPLIER_CNT"})

    result_df = agg.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )

    return result_df


def new_tpch_q17(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "Brand#23"
    var2 = "MED BOX"

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn1[((jn1["P_BRAND"] == var1) & (jn1["P_CONTAINER"] == var2))]

    gb = jn2.groupby("L_PARTKEY", as_index=False).agg(
        L_QUANTITY_AVG=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )

    jn4 = jn2.merge(gb, left_on="L_PARTKEY", right_on="L_PARTKEY", how="left")
    jn5 = jn4[jn4["L_QUANTITY"] < 0.2 * jn4["L_QUANTITY_AVG"]]
    total = jn5["L_EXTENDEDPRICE"].sum() / 7.0

    result_df = pd.DataFrame({"AVG_YEARLY": [round(total, 2)]})

    return result_df


def new_tpch_q18(lineitem, orders, customer, pd=bodo.pandas):
    """Adapted from:
    github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = 300

    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False, sort=False)["L_QUANTITY"].sum()
    fgb1 = gb1[gb1.L_QUANTITY > var1]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
        sort=False,
    )["L_QUANTITY"].sum()
    total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    return total.head(100)


def new_tpch_q19(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn1[
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

    total = (jn2["L_EXTENDEDPRICE"] * (1 - jn2["L_DISCOUNT"])).sum()

    result_df = pd.DataFrame({"REVENUE": [total]})

    return result_df


def new_tpch_q20(lineitem, part, nation, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1996-01-01")
    var2 = pd.Timestamp("1997-01-01")
    var3 = "JORDAN"
    var4 = "azure"

    flineitem = lineitem[
        (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
    ]
    gb = flineitem.groupby(["L_SUPPKEY", "L_PARTKEY"], as_index=False).agg(
        SUM_QUANTITY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum")
    )
    gb["SUM_QUANTITY"] = gb["SUM_QUANTITY"] * 0.5

    fnation = nation[nation["N_NAME"] == var3]

    jn1 = supplier.merge(fnation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    fpart = part[part["P_NAME"].str.startswith(var4)]

    jn2 = partsupp.merge(fpart, left_on="PS_PARTKEY", right_on="P_PARTKEY")
    jn3 = jn2.merge(
        gb,
        left_on=["PS_SUPPKEY", "PS_PARTKEY"],
        right_on=["L_SUPPKEY", "L_PARTKEY"],
    )
    jn3 = jn3[jn3["PS_AVAILQTY"] > jn3["SUM_QUANTITY"]]
    jn4 = jn1.merge(jn3, left_on="S_SUPPKEY", right_on="PS_SUPPKEY")

    result_df = jn4[["S_NAME", "S_ADDRESS"]].sort_values("S_NAME", ascending=True)

    return result_df


def new_tpch_q21(lineitem, orders, supplier, nation, pd=bodo.pandas):
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

    filt = jn5[
        (
            (jn5["NUNIQUE_COL"] == 1)
            & (jn5["N_NAME"] == var1)
            & (jn5["O_ORDERSTATUS"] == "F")
        )
    ]
    gb3 = filt.groupby("S_NAME", as_index=False).agg(
        NUMWAIT=pd.NamedAgg(column="NUNIQUE_COL", aggfunc="size")
    )

    result_df = gb3.sort_values(["NUMWAIT", "S_NAME"], ascending=[False, True]).head(
        100
    )

    return result_df


def new_tpch_q22(customer, orders, pd=bodo.pandas):
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


@timethis
@collect_datasets
def q01(lineitem):
    print(tpch_q01(lineitem))


@timethis
@collect_datasets
def q02(part, partsupp, supplier, nation, region):
    print(tpch_q02(part, partsupp, supplier, nation, region))


@timethis
@collect_datasets
def q03(lineitem, orders, customer):
    print(tpch_q03(lineitem, orders, customer))


@timethis
@collect_datasets
def q04(lineitem, orders):
    print(tpch_q04(lineitem, orders))


@timethis
@collect_datasets
def q05(lineitem, orders, customer, nation, region, supplier):
    print(tpch_q05(lineitem, orders, customer, nation, region, supplier))


@timethis
@collect_datasets
def q06(lineitem):
    print(tpch_q06(lineitem))


@timethis
@collect_datasets
def q07(lineitem, supplier, orders, customer, nation):
    print(tpch_q07(lineitem, supplier, orders, customer, nation))


@timethis
@collect_datasets
def q08(part, lineitem, supplier, orders, customer, nation, region):
    print(tpch_q08(part, lineitem, supplier, orders, customer, nation, region))


@timethis
@collect_datasets
def q09(lineitem, orders, part, nation, partsupp, supplier):
    print(tpch_q09(lineitem, orders, part, nation, partsupp, supplier))


@timethis
@collect_datasets
def q10(lineitem, orders, customer, nation):
    print(tpch_q10(lineitem, orders, customer, nation))


@timethis
@collect_datasets
def q11(partsupp, supplier, nation, scale_factor):
    print(tpch_q11(partsupp, supplier, nation, scale_factor))


@timethis
@collect_datasets
def q12(lineitem, orders):
    print(tpch_q12(lineitem, orders))


@timethis
@collect_datasets
def q13(customer, orders):
    print(tpch_q13(customer, orders))


@timethis
@collect_datasets
def q14(lineitem, part):
    print(tpch_q14(lineitem, part))


@timethis
@collect_datasets
def q15(lineitem, supplier):
    print(tpch_q15(lineitem, supplier))


@timethis
@collect_datasets
def q16(part, partsupp, supplier):
    print(tpch_q16(part, partsupp, supplier))


@timethis
@collect_datasets
def q17(lineitem, part):
    print(tpch_q17(lineitem, part))


@timethis
@collect_datasets
def q18(lineitem, orders, customer):
    print(tpch_q18(lineitem, orders, customer))


@timethis
@collect_datasets
def q19(lineitem, part):
    print(tpch_q19(lineitem, part))


@timethis
@collect_datasets
def q20(lineitem, part, nation, partsupp, supplier):
    print(tpch_q20(lineitem, part, nation, partsupp, supplier))


@timethis
@collect_datasets
def q21(lineitem, orders, supplier, nation):
    print(tpch_q21(lineitem, orders, supplier, nation))


@timethis
@collect_datasets
def q22(customer, orders):
    print(tpch_q22(customer, orders))


def run_queries(root: str, queries: list[int], scale_factor: float):
    total_start = time.time()
    print("Start data loading")
    queries_to_args = {}
    for query in queries:
        args = []
        for arg in _query_to_args[query]:
            if arg == "scale_factor":
                args.append(scale_factor)
            else:
                args.append(globals()[f"load_{arg}"](root))
        queries_to_args[query] = args
    print(f"Data loading time (s): {time.time() - total_start}")

    total_start = time.time()
    for query in queries:
        globals()[f"q{query:02}"](*queries_to_args[query])
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
        help="Comma separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        required=False,
        default=1.0,
        help="Scale factor (used in query 11).",
    )

    args = parser.parse_args()
    data_set = args.folder
    scale_factor = args.scale_factor

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    run_queries(data_set, queries=queries, scale_factor=scale_factor)


if __name__ == "__main__":
    print(f"Running TPC-H against pd v{pd.__version__}")
    main()
