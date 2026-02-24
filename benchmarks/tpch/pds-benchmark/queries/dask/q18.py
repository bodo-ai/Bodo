from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 18


def q() -> None:
    def query() -> pd.DataFrame:
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()

        # FIXME: https://github.com/dask/dask-expr/issues/867
        qnt_over_300 = (
            lineitem.groupby("l_orderkey").l_quantity.sum(split_out=True).reset_index()
        )
        qnt_over_300 = qnt_over_300[qnt_over_300.l_quantity > 300]

        table = (
            orders.merge(
                qnt_over_300,
                left_on="o_orderkey",
                right_on="l_orderkey",
                how="leftsemi",
            )
            .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="inner")
            .merge(customer, left_on="o_custkey", right_on="c_custkey", how="inner")
        )

        return (
            table.groupby(
                ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"]
            )
            .l_quantity.sum()
            .reset_index()
            .rename(columns={"l_quantity": "col6"})
            .sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
            .head(100, compute=False)
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
