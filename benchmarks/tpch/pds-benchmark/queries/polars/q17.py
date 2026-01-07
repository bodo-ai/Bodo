from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 17


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

    var1 = "Brand#23"
    var2 = "MED BOX"

    q1 = (
        part.filter(pl.col("p_brand") == var1)
        .filter(pl.col("p_container") == var2)
        .join(lineitem, how="left", left_on="p_partkey", right_on="l_partkey")
    )

    return (
        q1.group_by("p_partkey")
        .agg((0.2 * pl.col("l_quantity").mean()).alias("avg_quantity"))
        .select(pl.col("p_partkey").alias("key"), pl.col("avg_quantity"))
        .join(q1, left_on="key", right_on="p_partkey")
        .filter(pl.col("l_quantity") < pl.col("avg_quantity"))
        .select((pl.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly"))
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
