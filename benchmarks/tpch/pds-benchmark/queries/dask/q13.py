from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 13


def q() -> None:
    def query() -> pd.DataFrame:
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        orders = orders[~orders.o_comment.str.contains("special.*requests")]
        subquery = customer.merge(
            orders, left_on="c_custkey", right_on="o_custkey", how="left"
        )
        subquery = (
            subquery.groupby("c_custkey")
            .o_orderkey.count()
            .to_frame()
            .reset_index()
            .rename(columns={"o_orderkey": "c_count"})[["c_custkey", "c_count"]]
        )
        return (
            subquery.groupby("c_count")
            .size()
            .to_frame()
            .rename(columns={0: "custdist"})
            .reset_index()
            .sort_values(by=["custdist", "c_count"], ascending=[False, False])
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
