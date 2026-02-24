from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 21


def q() -> None:
    def query() -> pd.DataFrame:
        supplier = utils.get_supplier_ds()
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()
        nation = utils.get_nation_ds()

        NATION = "SAUDI ARABIA"

        res_1 = (
            lineitem.groupby("l_orderkey")["l_suppkey"]
            .nunique()
            .rename("nunique_col")
            .reset_index()
        )
        res_1 = res_1[res_1["nunique_col"] > 1].merge(
            lineitem[lineitem["l_receiptdate"] > lineitem["l_commitdate"]],
            on="l_orderkey",
        )

        q_final = (
            res_1.groupby("l_orderkey")["l_suppkey"]
            .nunique()
            .rename("nunique_col")
            .reset_index()
            .merge(res_1, on="l_orderkey", suffixes=("", "_right"))
            .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
            .merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        )

        predicate = (
            (q_final["nunique_col"] == 1)
            & (q_final["n_name"] == NATION)
            & (q_final["o_orderstatus"] == "F")
        )

        return (
            q_final[predicate]
            .groupby("s_name")
            .size()
            .rename("numwait")
            .reset_index()
            .sort_values(["numwait", "s_name"], ascending=[False, True])
            .head(100, compute=False)
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
