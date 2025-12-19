from __future__ import annotations

from datetime import datetime

import pandas as pd
from queries.dask import utils

Q_NUM = 4


def q() -> None:
    def query() -> pd.DataFrame:
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()

        date1 = datetime.strptime("1993-10-01", "%Y-%m-%d")
        date2 = datetime.strptime("1993-07-01", "%Y-%m-%d")

        lsel = line_item_ds.l_commitdate < line_item_ds.l_receiptdate
        osel = (orders_ds.o_orderdate < date1) & (orders_ds.o_orderdate >= date2)
        flineitem = line_item_ds[lsel]
        forders = orders_ds[osel]
        jn = forders.merge(
            flineitem, how="leftsemi", left_on="o_orderkey", right_on="l_orderkey"
        )
        result_df = (
            jn.groupby("o_orderpriority")
            .size()
            .to_frame("order_count")
            .reset_index()
            .sort_values(["o_orderpriority"])
        )
        return result_df.compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
