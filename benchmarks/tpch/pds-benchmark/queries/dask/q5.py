from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from queries.dask import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 5


def q() -> None:
    def query() -> pd.DataFrame:
        region_ds = utils.get_region_ds()
        nation_ds = utils.get_nation_ds()
        customer_ds = utils.get_customer_ds()
        line_item_ds = utils.get_line_item_ds()
        orders_ds = utils.get_orders_ds()
        supplier_ds = utils.get_supplier_ds()

        date1 = datetime.strptime("1994-01-01", "%Y-%m-%d")
        date2 = datetime.strptime("1995-01-01", "%Y-%m-%d")

        rsel = region_ds.r_name == "ASIA"
        osel = (orders_ds.o_orderdate >= date1) & (orders_ds.o_orderdate < date2)
        forders = orders_ds[osel]
        fregion = region_ds[rsel]
        jn1 = fregion.merge(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        jn2 = jn1.merge(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        jn3 = jn2.merge(forders, left_on="c_custkey", right_on="o_custkey")
        jn4 = jn3.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        jn5 = supplier_ds.merge(
            jn4,
            left_on=["s_suppkey", "s_nationkey"],
            right_on=["l_suppkey", "n_nationkey"],
        )
        jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)
        gb = jn5.groupby("n_name")["revenue"].sum()
        return gb.reset_index().sort_values("revenue", ascending=False).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
