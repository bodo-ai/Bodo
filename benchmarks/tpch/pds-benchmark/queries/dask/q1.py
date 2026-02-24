from __future__ import annotations

from datetime import datetime

import pandas as pd
from queries.dask import utils

Q_NUM = 1


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem_ds = utils.get_line_item_ds()

        VAR1 = datetime(1998, 9, 2)

        lineitem_filtered = lineitem_ds[lineitem_ds.l_shipdate <= VAR1]
        lineitem_filtered["sum_qty"] = lineitem_filtered.l_quantity
        lineitem_filtered["sum_base_price"] = lineitem_filtered.l_extendedprice
        lineitem_filtered["avg_qty"] = lineitem_filtered.l_quantity
        lineitem_filtered["avg_price"] = lineitem_filtered.l_extendedprice
        lineitem_filtered["sum_disc_price"] = lineitem_filtered.l_extendedprice * (
            1 - lineitem_filtered.l_discount
        )
        lineitem_filtered["sum_charge"] = (
            lineitem_filtered.l_extendedprice
            * (1 - lineitem_filtered.l_discount)
            * (1 + lineitem_filtered.l_tax)
        )
        lineitem_filtered["avg_disc"] = lineitem_filtered.l_discount
        lineitem_filtered["count_order"] = lineitem_filtered.l_orderkey
        gb = lineitem_filtered.groupby(["l_returnflag", "l_linestatus"])

        total = gb.agg(
            {
                "sum_qty": "sum",
                "sum_base_price": "sum",
                "sum_disc_price": "sum",
                "sum_charge": "sum",
                "avg_qty": "mean",
                "avg_price": "mean",
                "avg_disc": "mean",
                "count_order": "size",
            }
        )

        return (
            total.reset_index().sort_values(["l_returnflag", "l_linestatus"]).compute()
        )

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
