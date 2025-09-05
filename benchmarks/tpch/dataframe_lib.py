# Original from https://gist.githubusercontent.com/UranusSeven/55817bf0f304cc24f5eb63b2f1c3e2cd/raw/796dbd2fce6441821fc0b5bc51491edb49639c55/tpch.py
import argparse
import functools
import inspect
import time
from collections.abc import Callable

import bodo.pandas as pd


@functools.lru_cache
def load_lineitem(data_folder: str, pd=pd):
    print("Loading lineitem")
    data_path = data_folder + "/lineitem.pq"
    df = pd.read_parquet(data_path)
    df["L_SHIPDATE"] = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    print("Done loading lineitem")
    return df


@functools.lru_cache
def load_part(data_folder: str, pd=pd):
    print("Loading part")
    data_path = data_folder + "/part.pq"
    df = pd.read_parquet(data_path)
    print("Done loading part")
    return df


@functools.lru_cache
def load_orders(data_folder: str, pd=pd):
    print("Loading orders")
    data_path = data_folder + "/orders.pq"
    df = pd.read_parquet(data_path)
    df["O_ORDERDATE"] = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    print("Done loading orders")
    return df


@functools.lru_cache
def load_customer(data_folder: str, pd=pd):
    print("Loading customer")
    data_path = data_folder + "/customer.pq"
    df = pd.read_parquet(data_path)
    print("Done loading customer")
    return df


@functools.lru_cache
def load_nation(data_folder: str, pd=pd):
    print("Loading nation")
    data_path = data_folder + "/nation.pq"
    df = pd.read_parquet(data_path)
    print("Done loading nation")
    return df


@functools.lru_cache
def load_region(data_folder: str, pd=pd):
    print("Loading region")
    data_path = data_folder + "/region.pq"
    df = pd.read_parquet(data_path)
    print("Done loading region")
    return df


@functools.lru_cache
def load_supplier(data_folder: str, pd=pd):
    print("Loading supplier")
    data_path = data_folder + "/supplier.pq"
    df = pd.read_parquet(data_path)
    print("Done loading supplier")
    return df


@functools.lru_cache
def load_partsupp(data_folder: str, pd=pd):
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


_query_to_datasets: dict[int, list[str]] = {}


def collect_datasets(func: Callable):
    _query_to_datasets[int(func.__name__[1:])] = list(
        inspect.signature(func).parameters
    )
    return func


def tpch_q01(lineitem, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/dask/q1.py
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


def tpch_q02(part, partsupp, supplier, nation, region, pd=pd):
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


def tpch_q03(lineitem, orders, customer, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q3.py
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


# TODO: Fix fallback in drop_duplicates
def tpch_q04(lineitem, orders, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q4.py
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


def tpch_q05(lineitem, orders, customer, nation, region, supplier, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q5.py
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


def tpch_q06(lineitem, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q6.py
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


def tpch_q07(lineitem, supplier, orders, customer, nation, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q7.py
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


def tpch_q08(part, lineitem, supplier, orders, customer, nation, region, pd=pd):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q8.py
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


def tpch_q09(lineitem, orders, part, nation, partsupp, supplier, pd=pd):
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
    return total


def tpch_q10(lineitem, orders, customer, nation, pd=pd):
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
    return total.head(20)


def tpch_q11(partsupp, supplier, nation, pd=pd):
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


def tpch_q12(lineitem, orders, pd=pd):
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


def tpch_q13(customer, orders, pd=pd):
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


def tpch_q14(lineitem, part, pd=pd):
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
    return total


def tpch_q15(lineitem, supplier, pd=pd):
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


def tpch_q16(part, partsupp, supplier, pd=pd):
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


def tpch_q17(lineitem, part, pd=pd):
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
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = pd.DataFrame({"avg_yearly": [total["L_EXTENDEDPRICE"].sum() / 7.0]})
    return total


def tpch_q18(lineitem, orders, customer, pd=pd):
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


def tpch_q19(lineitem, part, pd=pd):
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


def tpch_q20(lineitem, part, nation, partsupp, supplier, pd=pd):
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


def tpch_q21(lineitem, orders, supplier, nation, pd=pd):
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


def tpch_q22(customer, orders, pd=pd):
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
def q11(partsupp, supplier, nation):
    print(tpch_q11(partsupp, supplier, nation))


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


def run_queries(
    root: str,
    queries: list[int],
):
    total_start = time.time()
    print("Start data loading")
    queries_to_args = {}
    for query in queries:
        args = []
        for dataset in _query_to_datasets[query]:
            args.append(globals()[f"load_{dataset}"](root))
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

    args = parser.parse_args()
    data_set = args.folder

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    run_queries(
        data_set,
        queries=queries,
    )


if __name__ == "__main__":
    print(f"Running TPC-H against pd v{pd.__version__}")
    main()
