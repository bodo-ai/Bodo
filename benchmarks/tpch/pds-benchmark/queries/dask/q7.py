from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import dask.dataframe as dd


Q_NUM = 7


def q() -> None:
    def query() -> pd.DataFrame:
        nation_ds = utils.get_nation_ds()
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        supplier_ds = utils.get_supplier_ds()

        var1 = datetime.strptime("1995-01-01", "%Y-%m-%d")
        var2 = datetime.strptime("1997-01-01", "%Y-%m-%d")

        lineitem_filtered = line_item_ds[
            (line_item_ds["l_shipdate"] >= var1) & (line_item_ds["l_shipdate"] < var2)
        ]
        lineitem_filtered["l_year"] = lineitem_filtered["l_shipdate"].dt.year
        lineitem_filtered["revenue"] = lineitem_filtered["l_extendedprice"] * (
            1.0 - lineitem_filtered["l_discount"]
        )

        supplier_filtered = supplier_ds
        orders_filtered = orders_ds
        customer_filtered = customer_ds
        n1 = nation_ds[(nation_ds["n_name"] == "FRANCE")]
        n2 = nation_ds[(nation_ds["n_name"] == "GERMANY")]

        # ----- do nation 1 -----
        N1_C = customer_filtered.merge(
            n1, left_on="c_nationkey", right_on="n_nationkey", how="inner"
        )
        N1_C = N1_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
            columns={"n_name": "cust_nation"}
        )
        N1_C_O = N1_C.merge(
            orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
        )
        N1_C_O = N1_C_O.drop(columns=["c_custkey", "o_custkey"])

        N2_S = supplier_filtered.merge(
            n2, left_on="s_nationkey", right_on="n_nationkey", how="inner"
        )
        N2_S = N2_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
            columns={"n_name": "supp_nation"}
        )
        N2_S_L = N2_S.merge(
            lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
        )
        N2_S_L = N2_S_L.drop(columns=["s_suppkey", "l_suppkey"])

        total1 = N1_C_O.merge(
            N2_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
        )
        total1 = total1.drop(columns=["o_orderkey", "l_orderkey"])

        # ----- do nation 2 ----- (same as nation 1 section but with nation 2)
        N2_C = customer_filtered.merge(
            n2, left_on="c_nationkey", right_on="n_nationkey", how="inner"
        )
        N2_C = N2_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
            columns={"n_name": "cust_nation"}
        )
        N2_C_O = N2_C.merge(
            orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
        )
        N2_C_O = N2_C_O.drop(columns=["c_custkey", "o_custkey"])

        N1_S = supplier_filtered.merge(
            n1, left_on="s_nationkey", right_on="n_nationkey", how="inner"
        )
        N1_S = N1_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
            columns={"n_name": "supp_nation"}
        )
        N1_S_L = N1_S.merge(
            lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
        )
        N1_S_L = N1_S_L.drop(columns=["s_suppkey", "l_suppkey"])

        total2 = N2_C_O.merge(
            N1_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
        )
        total2 = total2.drop(columns=["o_orderkey", "l_orderkey"])

        # concat results
        total = dd.concat([total1, total2])
        result_df = (
            total.groupby(["supp_nation", "cust_nation", "l_year"])
            .revenue.agg("sum")
            .reset_index()
        )
        result_df.columns = ["supp_nation", "cust_nation", "l_year", "revenue"]

        return result_df.sort_values(
            by=["supp_nation", "cust_nation", "l_year"],
            ascending=True,
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
