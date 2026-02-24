from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 12


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

    var1 = "MAIL"
    var2 = "SHIP"
    var3 = date(1994, 1, 1)
    var4 = date(1995, 1, 1)

    return (
        orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .filter(pl.col("l_shipmode").is_in([var1, var2]))
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .filter(pl.col("l_shipdate") < pl.col("l_commitdate"))
        .filter(pl.col("l_receiptdate").is_between(var3, var4, closed="left"))
        .with_columns(
            line_count=pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
        )
        .group_by("l_shipmode")
        .agg(
            high_line_count=pl.col.line_count.sum(),
            low_line_count=pl.col.line_count.not_().sum(),
        )
        .sort("l_shipmode")
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
