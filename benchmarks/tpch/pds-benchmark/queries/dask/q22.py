from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 22


def q() -> None:
    def query() -> pd.DataFrame:
        orders_ds = utils.get_orders_ds()
        customer_ds = utils.get_customer_ds()

        customers = customer_ds
        customers["cntrycode"] = customers["c_phone"].str.strip().str.slice(0, 2)
        customers = customers[
            customers["cntrycode"].isin(("13", "31", "23", "29", "30", "18", "17"))
        ]

        average_c_acctbal = customers[customers["c_acctbal"] > 0.0]["c_acctbal"].mean()

        custsale = customers[customers["c_acctbal"] > average_c_acctbal]
        custsale = custsale.merge(
            orders_ds, left_on="c_custkey", right_on="o_custkey", how="left"
        )
        custsale = custsale[custsale["o_custkey"].isnull()]
        custsale = custsale.groupby("cntrycode").agg({"c_acctbal": ["size", "sum"]})
        custsale.columns = custsale.columns.get_level_values(-1)
        return (
            custsale.rename(columns={"sum": "totacctbal", "size": "numcust"})
            .reset_index()
            .sort_values("cntrycode", ascending=True)
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
