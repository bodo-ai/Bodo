from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 14


def q(
    lineitem: None | pl.LazyFrame = None,
    part: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if lineitem is None:
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

    assert lineitem is not None
    assert part is not None

    var1 = date(1994, 3, 1)
    var2 = date(1994, 4, 1)

    return (
        lineitem.join(part, left_on="l_partkey", right_on="p_partkey")
        .filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
        .select(
            (
                100.00
                * pl.when(pl.col("p_type").str.starts_with("PROMO"))
                .then(pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .otherwise(0)
                .sum()
                / (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum()
            )
            .round(2)
            .alias("promo_revenue")
        )
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
