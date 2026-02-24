from __future__ import annotations

from datetime import datetime

import pandas as pd
from queries.dask import utils

Q_NUM = 6


def q() -> None:
    def query() -> pd.DataFrame:
        line_item_ds = utils.get_line_item_ds()

        date1 = datetime.strptime("1994-01-01", "%Y-%m-%d")
        date2 = datetime.strptime("1995-01-01", "%Y-%m-%d")
        var3 = 24

        sel = (
            (line_item_ds.l_shipdate >= date1)
            & (line_item_ds.l_shipdate < date2)
            & (line_item_ds.l_discount >= 0.05)
            & (line_item_ds.l_discount <= 0.07)
            & (line_item_ds.l_quantity < var3)
        )

        flineitem = line_item_ds[sel]
        revenue = (flineitem.l_extendedprice * flineitem.l_discount).to_frame()
        return revenue.sum().to_frame("revenue").compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
