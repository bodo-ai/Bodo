from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from queries.dask import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 3


def q() -> None:
    def query() -> pd.DataFrame:
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()

        var1 = "BUILDING"
        var2 = date(1995, 3, 15)

        lsel = line_item_ds.l_shipdate > var2
        osel = orders_ds.o_orderdate < var2
        csel = customer_ds.c_mktsegment == var1
        flineitem = line_item_ds[lsel]
        forders = orders_ds[osel]
        fcustomer = customer_ds[csel]
        jn1 = fcustomer.merge(forders, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn1.merge(flineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)
        total = jn2.groupby(["l_orderkey", "o_orderdate", "o_shippriority"])[
            "revenue"
        ].sum()
        return (
            total.reset_index()
            .sort_values(["revenue"], ascending=False)
            .head(10, compute=True)[
                ["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]
            ]
        )

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
