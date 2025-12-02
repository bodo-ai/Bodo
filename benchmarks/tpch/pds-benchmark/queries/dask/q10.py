from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 10


def q() -> None:
    def query() -> pd.DataFrame:
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()

        orderdate_from = datetime.strptime("1993-10-01", "%Y-%m-%d")
        orderdate_to = datetime.strptime("1994-01-01", "%Y-%m-%d")

        orders = orders[
            (orders.o_orderdate >= orderdate_from) & (orders.o_orderdate < orderdate_to)
        ]
        lineitem = lineitem[lineitem.l_returnflag == "R"]

        query = (
            lineitem.merge(
                orders, left_on="l_orderkey", right_on="o_orderkey", how="inner"
            )
            .merge(customer, left_on="o_custkey", right_on="c_custkey", how="inner")
            .merge(nation, left_on="c_nationkey", right_on="n_nationkey", how="inner")
        )

        # TODO: ideally the filters are pushed up before the merge during optimization

        # query = query[
        #     (query.o_orderdate >= orderdate_from)
        #     & (query.o_orderdate < orderdate_to)
        #     & (query.l_returnflag == "R")
        # ]

        query["revenue"] = query.l_extendedprice * (1 - query.l_discount)
        return (
            query.groupby(
                [
                    "c_custkey",
                    "c_name",
                    "c_acctbal",
                    "c_phone",
                    "n_name",
                    "c_address",
                    "c_comment",
                ]
            )
            .revenue.sum()
            .round(2)
            .reset_index()
            .sort_values(by=["revenue"], ascending=[False])
            .head(20, compute=False)[
                [
                    "c_custkey",
                    "c_name",
                    "revenue",
                    "c_acctbal",
                    "n_name",
                    "c_address",
                    "c_phone",
                    "c_comment",
                ]
            ]
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
