from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 20


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()
        nation = utils.get_nation_ds()
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        shipdate_from = datetime.strptime("1994-01-01", "%Y-%m-%d")
        shipdate_to = datetime.strptime("1995-01-01", "%Y-%m-%d")

        res_1 = lineitem[
            (lineitem["l_shipdate"] >= shipdate_from)
            & (lineitem["l_shipdate"] < shipdate_to)
        ]
        res_1 = (
            res_1.groupby(["l_suppkey", "l_partkey"])["l_quantity"]
            .sum()
            .rename("sum_quantity")
            .reset_index()
        )
        res_1["sum_quantity"] = res_1["sum_quantity"] * 0.5
        res_2 = nation[nation["n_name"] == "CANADA"]
        res_3 = supplier.merge(res_2, left_on="s_nationkey", right_on="n_nationkey")
        res_4 = part[part["p_name"].str.strip().str.startswith("forest")]

        q_final = partsupp.merge(
            res_4, how="leftsemi", left_on="ps_partkey", right_on="p_partkey"
        ).merge(
            res_1,
            left_on=["ps_suppkey", "ps_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        q_final = q_final[q_final["ps_availqty"] > q_final["sum_quantity"]]
        q_final = res_3.merge(
            q_final, how="leftsemi", left_on="s_suppkey", right_on="ps_suppkey"
        )
        q_final["s_address"] = q_final["s_address"].str.strip()
        return (
            q_final[["s_name", "s_address"]]
            .sort_values("s_name", ascending=True)
            .compute()
        )

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
