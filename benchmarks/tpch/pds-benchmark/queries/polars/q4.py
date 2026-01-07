from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 4


def q(
    lineitem: None | pl.LazyFrame = None,
    orders: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if lineitem is None:
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()

    assert lineitem is not None
    assert orders is not None

    var1 = date(1993, 8, 1)
    var2 = date(1993, 11, 1)

    return (
        # SQL exists translates to semi join in Polars API
        orders.join(
            (lineitem.filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))),
            left_on="o_orderkey",
            right_on="l_orderkey",
            how="semi",
        )
        .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
        .group_by("o_orderpriority")
        .agg(pl.len().alias("order_count"))
        .sort("o_orderpriority")
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
