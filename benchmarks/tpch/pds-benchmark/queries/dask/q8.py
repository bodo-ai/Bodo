from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 8


def q() -> None:
    def query() -> pd.DataFrame:
        var1 = datetime.strptime("1995-01-01", "%Y-%m-%d")
        var2 = datetime.strptime("1997-01-01", "%Y-%m-%d")

        supplier = utils.get_supplier_ds()
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()
        customer = utils.get_customer_ds()
        nation = utils.get_nation_ds()
        region = utils.get_region_ds()
        part = utils.get_part_ds()
        part = part[part["p_type"] == "ECONOMY ANODIZED STEEL"][["p_partkey"]]
        lineitem["volume"] = lineitem["l_extendedprice"] * (
            1.0 - lineitem["l_discount"]
        )
        total = part.merge(
            lineitem, left_on="p_partkey", right_on="l_partkey", how="inner"
        )

        total = total.merge(
            supplier, left_on="l_suppkey", right_on="s_suppkey", how="inner"
        )

        orders = orders[
            (orders["o_orderdate"] >= var1) & (orders["o_orderdate"] < var2)
        ]
        orders["o_year"] = orders["o_orderdate"].dt.year
        total = total.merge(
            orders, left_on="l_orderkey", right_on="o_orderkey", how="inner"
        )

        total = total.merge(
            customer, left_on="o_custkey", right_on="c_custkey", how="inner"
        )

        n1_filtered = nation[["n_nationkey", "n_regionkey"]]
        total = total.merge(
            n1_filtered, left_on="c_nationkey", right_on="n_nationkey", how="inner"
        )

        n2_filtered = nation[["n_nationkey", "n_name"]].rename(
            columns={"n_name": "nation"}
        )
        total = total.merge(
            n2_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
        )

        region = region[region["r_name"] == "AMERICA"][["r_regionkey"]]
        total = total.merge(
            region, left_on="n_regionkey", right_on="r_regionkey", how="inner"
        )

        mkt_brazil = (
            total[total["nation"] == "BRAZIL"]
            .groupby("o_year")
            .volume.sum()
            .reset_index()
        )
        mkt_total = total.groupby("o_year").volume.sum().reset_index()

        final = mkt_total.merge(
            mkt_brazil,
            left_on="o_year",
            right_on="o_year",
            suffixes=("_mkt", "_brazil"),
        )

        final["mkt_share"] = (final.volume_brazil / final.volume_mkt).round(2)
        return final.sort_values(by=["o_year"], ascending=[True])[
            ["o_year", "mkt_share"]
        ].compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
