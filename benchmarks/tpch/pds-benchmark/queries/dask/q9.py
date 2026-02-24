from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 9


def q() -> None:
    def query() -> pd.DataFrame:
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()
        nation = utils.get_nation_ds()
        part = part[part.p_name.str.contains("green")]

        subquery = (
            part.merge(
                partsupp, left_on="p_partkey", right_on="ps_partkey", how="inner"
            )
            .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="inner")
            .merge(
                lineitem,
                left_on=["ps_partkey", "ps_suppkey"],
                right_on=["l_partkey", "l_suppkey"],
                how="inner",
            )
            .merge(orders, left_on="l_orderkey", right_on="o_orderkey", how="inner")
            .merge(nation, left_on="s_nationkey", right_on="n_nationkey", how="inner")
        )
        subquery["o_year"] = subquery.o_orderdate.dt.year
        subquery["nation"] = subquery.n_name
        subquery["amount"] = (
            subquery.l_extendedprice * (1 - subquery.l_discount)
            - subquery.ps_supplycost * subquery.l_quantity
        )
        subquery = subquery[["o_year", "nation", "amount"]]

        return (
            subquery.groupby(["nation", "o_year"])
            .amount.sum()
            .round(2)
            .reset_index()
            .rename(columns={"amount": "sum_profit"})
            .sort_values(by=["nation", "o_year"], ascending=[True, False])
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
