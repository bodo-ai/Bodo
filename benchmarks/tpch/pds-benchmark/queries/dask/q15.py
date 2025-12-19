from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 15


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()

        shipdate_from = datetime.strptime("1996-01-01", "%Y-%m-%d")
        shipdate_to = datetime.strptime("1996-04-01", "%Y-%m-%d")

        # Create revenue view
        lineitem = lineitem[
            (lineitem.l_shipdate >= shipdate_from) & (lineitem.l_shipdate < shipdate_to)
        ]
        lineitem["revenue"] = lineitem.l_extendedprice * (1 - lineitem.l_discount)
        revenue = (
            lineitem.groupby("l_suppkey")
            .revenue.sum()
            .to_frame()
            .reset_index()
            .rename(columns={"revenue": "total_revenue", "l_suppkey": "supplier_no"})
        )

        # Query
        table = supplier.merge(
            revenue, left_on="s_suppkey", right_on="supplier_no", how="inner"
        )
        return (
            table[table.total_revenue == revenue.total_revenue.max()][
                ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
            ]
            .sort_values(by="s_suppkey")
            .compute()
        )

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
