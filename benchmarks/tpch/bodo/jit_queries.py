"""
TPC-H 22 Queries with Bodo
Usage:
    python bodo_queries.py --folder [folder]

Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.
"""

import argparse
import time

import numpy as np
import pandas as pd

import bodo


@bodo.jit(cache=True)
def run_queries(data_folder, queries):
    t1 = time.time()
    for i in range(len(queries)):
        if queries[i] == 1:
            q01(data_folder)
        elif queries[i] == 2:
            q02(data_folder)
        elif queries[i] == 3:
            q03(data_folder)
        elif queries[i] == 4:
            q04(data_folder)
        elif queries[i] == 5:
            q05(data_folder)
        elif queries[i] == 6:
            q06(data_folder)
        elif queries[i] == 7:
            q07(data_folder)
        elif queries[i] == 8:
            q08(data_folder)
        elif queries[i] == 9:
            q09(data_folder)
        elif queries[i] == 10:
            q10(data_folder)
        elif queries[i] == 11:
            q11(data_folder)
        elif queries[i] == 12:
            q12(data_folder)
        elif queries[i] == 13:
            q13(data_folder)
        elif queries[i] == 14:
            q14(data_folder)
        elif queries[i] == 15:
            q15(data_folder)
        elif queries[i] == 16:
            q16(data_folder)
        elif queries[i] == 17:
            q17(data_folder)
        elif queries[i] == 18:
            q18(data_folder)
        elif queries[i] == 19:
            q19(data_folder)
        elif queries[i] == 20:
            q20(data_folder)
        elif queries[i] == 21:
            q21(data_folder)
        elif queries[i] == 22:
            q22(data_folder)
    print("Total Query time (s): ", time.time() - t1)


@bodo.jit(cache=True)
def load_lineitem(data_folder):
    t0 = time.time()
    data_path = data_folder + "/lineitem.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load lineitem (s): ", time.time() - t0)
    return df


@bodo.jit(cache=True)
def load_part(data_folder):
    t0 = time.time()
    data_path = data_folder + "/part.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load part (s): ", time.time() - t0)
    return df


@bodo.jit(cache=True)
def load_orders(data_folder):
    t0 = time.time()
    data_path = data_folder + "/orders.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load orders (s): ", time.time() - t0)
    return df


@bodo.jit(cache=True)
def load_customer(data_folder):
    t0 = time.time()
    data_path = data_folder + "/customer.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load customer (s): ", time.time() - t0)
    return df


# Nation is a very small file so set it to replicated
@bodo.jit(distributed=False)
def load_nation(data_folder):
    t0 = time.time()
    data_path = data_folder + "/nation.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load nation (s): ", time.time() - t0)
    return df


# Region is a very small file so set it to replicated
@bodo.jit(distributed=False)
def load_region(data_folder):
    t0 = time.time()
    data_path = data_folder + "/region.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load region (s): ", time.time() - t0)
    return df


@bodo.jit(cache=True)
def load_supplier(data_folder):
    t0 = time.time()
    data_path = data_folder + "/supplier.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load supplier (s): ", time.time() - t0)
    return df


@bodo.jit(cache=True)
def load_partsupp(data_folder):
    t0 = time.time()
    data_path = data_folder + "/partsupp.pq"
    df = pd.read_parquet(
        data_path,
    )
    print("Time to load partsupp (s): ", time.time() - t0)
    return df


@bodo.jit
def q01(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
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
        "L_QUANTITY",
        "L_EXTENDEDPRICE",
        "DISC_PRICE",
        "CHARGE",
        "AVG_QTY",
        "AVG_PRICE",
        "L_DISCOUNT",
        "L_ORDERKEY",
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
    total = total.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    print(total)
    print("Q01 Execution time (s): ", time.time() - t1)


@bodo.jit
def q02(data_folder):
    t1 = time.time()
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)
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
    min_values = merged_df.groupby("P_PARTKEY", as_index=False)["PS_SUPPLYCOST"].min()
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
    print(total)
    print("Q02 Execution time (s): ", time.time() - t1)


@bodo.jit
def q03(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
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
    jn2["TMP"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
    total = (
        jn2.groupby(["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False)[
            "TMP"
        ]
        .sum()
        .sort_values(["TMP"], ascending=False)
    )
    res = total.loc[:, ["L_ORDERKEY", "TMP", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    print(res.head(10))
    print("Q03 Execution time (s): ", time.time() - t1)


@bodo.jit
def q04(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
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
    print(total)
    print("Q04 Execution time (s): ", time.time() - t1)


@bodo.jit
def q05(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)
    supplier = load_supplier(data_folder)
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
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME", as_index=False)["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)
    bodo.parallel_print(total)
    print("Q05 Execution time (s): ", time.time() - t1)


@bodo.jit
def q06(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
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
    print(total)
    print("Q06 Execution time (s): ", time.time() - t1)


@bodo.jit
def q07(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= pd.Timestamp("1995-01-01"))
        & (lineitem["L_SHIPDATE"] < pd.Timestamp("1997-01-01"))
    ]
    lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].apply(
        lambda x: x.year
    )
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
    print(total)
    print("Q07 Execution time (s): ", time.time() - t1)


@bodo.jit
def q08(data_folder):
    t1 = time.time()
    part = load_part(data_folder)
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
    region = load_region(data_folder)
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
    orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].apply(lambda x: x.year)
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
        return numerator / demonimator

    total = total.groupby("O_YEAR", as_index=False).apply(udf)
    total.columns = ["O_YEAR", "MKT_SHARE"]
    total = total.sort_values(by=["O_YEAR"], ascending=[True])
    print(total)
    print("Q08 Execution time (s): ", time.time() - t1)


@bodo.jit
def q09(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    part = load_part(data_folder)
    nation = load_nation(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
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
    jn5["O_YEAR"] = jn5.O_ORDERDATE.apply(lambda x: x.year)
    gb = jn5.groupby(["N_NAME", "O_YEAR"], as_index=False)["TMP"].sum()
    total = gb.sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
    print(total)
    print("Q09 Execution time (s): ", time.time() - t1)


@bodo.jit
def q10(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    nation = load_nation(data_folder)
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
    )["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)
    print(total.head(20))
    print("Q10 Execution time (s): ", time.time() - t1)


@bodo.jit
def q11(data_folder):
    t1 = time.time()
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
    )
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
    nation_filtered = nation[(nation["N_NAME"] == "GERMANY")]
    nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001
    total = ps_supp_n_merge.groupby(["PS_PARTKEY"], as_index=False).agg(
        VALUE=pd.NamedAgg(column="TOTAL_COST", aggfunc="sum")
    )
    total = total[total["VALUE"] > sum_val]
    total = total.sort_values("VALUE", ascending=False)
    print(total)
    print("Q11 Execution time (s): ", time.time() - t1)


@bodo.jit
def q12(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
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
    total = total.sort_values("L_SHIPMODE")
    print(total)
    print("Q12 Execution time (s): ", time.time() - t1)


@bodo.jit
def q13(data_folder):
    t1 = time.time()
    customer = load_customer(data_folder)
    orders = load_orders(data_folder)
    customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
    orders_filtered = orders[
        ~orders["O_COMMENT"].str.contains("special(\S|\s)*requests")
    ]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]
    count_df = c_o_merged.groupby(["C_CUSTKEY"], as_index=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    total = count_df.groupby(["C_COUNT"], as_index=False).size()
    total.columns = ["C_COUNT", "CUSTDIST"]
    total = total.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])
    print(total)
    print("Q13 Execution time (s): ", time.time() - t1)


@bodo.jit
def q14(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)
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
    print(total)
    print("Q14 Execution time (s): ", time.time() - t1)


@bodo.jit
def q15(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    supplier = load_supplier(data_folder)
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
        lineitem_filtered.groupby("L_SUPPKEY", as_index=False)
        .agg(TOTAL_REVENUE=pd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))
        .rename(columns={"L_SUPPKEY": "SUPPLIER_NO"}, copy=False)
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
    print(total)
    print("Q15 Execution time (s): ", time.time() - t1)


@bodo.jit
def q16(data_folder):
    t1 = time.time()
    part = load_part(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
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
        supplier["S_COMMENT"].str.contains("Customer(\S|\s)*Complaints")
    ]
    supplier_filtered = supplier_filtered.loc[:, ["S_SUPPKEY"]].drop_duplicates()
    # left merge to select only PS_SUPPKEY values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left"
    )
    total = total[total["S_SUPPKEY"].isna()]
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    total = total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)[
        "PS_SUPPKEY"
    ].nunique()
    total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
    total = total.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )
    print(total)
    print("Q16 Execution time (s): ", time.time() - t1)


@bodo.jit
def q17(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)
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
    lineitem_avg = lineitem_filtered.groupby(["L_PARTKEY"], as_index=False).agg(
        avg=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = pd.DataFrame({"avg_yearly": [total["L_EXTENDEDPRICE"].sum() / 7.0]})
    print(total)
    print("Q17 Execution time (s): ", time.time() - t1)


@bodo.jit
def q18(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    customer = load_customer(data_folder)
    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False)["L_QUANTITY"].sum()
    fgb1 = gb1[gb1.L_QUANTITY > 300]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
    )["L_QUANTITY"].sum()
    total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    print(total.head(100))
    print("Q18 Execution time (s): ", time.time() - t1)


@bodo.jit
def q19(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)
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
    print(total)
    print("Q19 Execution time (s): ", time.time() - t1)


@bodo.jit
def q20(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    part = load_part(data_folder)
    nation = load_nation(data_folder)
    partsupp = load_partsupp(data_folder)
    supplier = load_supplier(data_folder)
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
    gb = jn2.groupby(["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"], as_index=False)[
        "L_QUANTITY"
    ].sum()
    gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
    jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
    total = jn4.sort_values("S_NAME").drop_duplicates()
    print(total)
    print("Q20 Execution time (s): ", time.time() - t1)


@bodo.jit
def q21(data_folder):
    t1 = time.time()
    lineitem = load_lineitem(data_folder)
    orders = load_orders(data_folder)
    supplier = load_supplier(data_folder)
    nation = load_nation(data_folder)
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
    ]

    # Keep all rows that have another row in linetiem with the same orderkey and different suppkey
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
        .groupby("L_ORDERKEY", as_index=False)["L_SUPPKEY"]
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
    lineitem_orderkeys = lineitem_filtered.groupby("L_ORDERKEY", as_index=False)[
        "L_SUPPKEY"
    ].nunique()
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
    total = total.groupby("S_NAME", as_index=False).size()
    total.columns = ["S_NAME", "NUMWAIT"]
    total = total.sort_values(by=["NUMWAIT", "S_NAME"], ascending=[False, True])
    print(total)
    print("Q21 Execution time (s): ", time.time() - t1)


@bodo.jit
def q22(data_folder):
    t1 = time.time()
    customer = load_customer(data_folder)
    orders = load_orders(data_folder)
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
    agg1 = customer_selected.groupby(["CNTRYCODE"], as_index=False).size()
    agg1.columns = ["CNTRYCODE", "NUMCUST"]
    agg2 = customer_selected.groupby(["CNTRYCODE"], as_index=False).agg(
        TOTACCTBAL=pd.NamedAgg(column="C_ACCTBAL", aggfunc="sum")
    )
    total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
    total = total.sort_values(by=["CNTRYCODE"], ascending=[True])
    print(total)
    print("Q22 Execution time (s): ", time.time() - t1)


def main():
    parser = argparse.ArgumentParser(description="Bodo tpch-queries")
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
    args = parser.parse_args()
    folder = args.folder
    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    qarr = np.array(queries)
    run_queries(folder, qarr)


if __name__ == "__main__":
    main()
