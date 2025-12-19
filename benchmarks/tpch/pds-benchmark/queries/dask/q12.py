from __future__ import annotations

import datetime
import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 12


def q() -> None:
    def query() -> pd.DataFrame:
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()

        receiptdate_from = datetime.datetime.strptime("1994-01-01", "%Y-%m-%d")
        receiptdate_to = receiptdate_from + datetime.timedelta(days=365)

        table = orders.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        table = table[
            (table.l_shipmode.isin(("MAIL", "SHIP")))
            & (table.l_commitdate < table.l_receiptdate)
            & (table.l_shipdate < table.l_commitdate)
            & (table.l_receiptdate >= receiptdate_from)
            & (table.l_receiptdate < receiptdate_to)
        ]

        mask = table.o_orderpriority.isin(("1-URGENT", "2-HIGH"))
        table["high_line_count"] = 0
        table["high_line_count"] = table.high_line_count.where(~mask, 1)
        table["low_line_count"] = 0
        table["low_line_count"] = table.low_line_count.where(mask, 1)

        return (
            table.groupby("l_shipmode")
            .agg({"high_line_count": "sum", "low_line_count": "sum"})
            .reset_index()
            .sort_values(by="l_shipmode")
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
