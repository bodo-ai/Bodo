from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 21


def q(
    lineitem: None | pl.LazyFrame = None,
    nation: None | pl.LazyFrame = None,
    orders: None | pl.LazyFrame = None,
    supplier: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if lineitem is None:
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()
        orders = utils.get_orders_ds()
        supplier = utils.get_supplier_ds()

    assert lineitem is not None
    assert nation is not None
    assert orders is not None
    assert supplier is not None

    var1 = "SAUDI ARABIA"

    q1 = (
        lineitem.group_by("l_orderkey")
        .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
        .filter(pl.col("n_supp_by_order") > 1)
        .join(
            lineitem.filter(pl.col("l_receiptdate") > pl.col("l_commitdate")),
            on="l_orderkey",
        )
    )

    return (
        q1.group_by("l_orderkey")
        .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
        .join(q1, on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("n_supp_by_order") == 1)
        .filter(pl.col("n_name") == var1)
        .filter(pl.col("o_orderstatus") == "F")
        .group_by("s_name")
        .agg(pl.len().alias("numwait"))
        .sort(by=["numwait", "s_name"], descending=[True, False])
        .head(100)
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
