from __future__ import annotations

from datetime import date

import pandas as pd
from queries.dask import utils

Q_NUM = 4


def q() -> None:
    def query() -> pd.DataFrame:
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()

        var1 = date(1993, 7, 1)
        var2 = date(1993, 10, 1)

        exists = line_item_ds[
            line_item_ds["l_commitdate"] < line_item_ds["l_receiptdate"]
        ]

        jn = orders_ds.merge(
            exists, left_on="o_orderkey", right_on="l_orderkey", how="leftsemi"
        )
        jn = jn[(jn["o_orderdate"] >= var1) & (jn["o_orderdate"] < var2)]

        gb = jn.groupby("o_orderpriority")
        agg = gb.agg(
            order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
        ).reset_index()

        result_df = agg.sort_values(["o_orderpriority"])

        return result_df.compute()  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
