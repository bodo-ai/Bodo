from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 20


def q(
    lineitem: None | pl.LazyFrame = None,
    nation: None | pl.LazyFrame = None,
    partsupp: None | pl.LazyFrame = None,
    supplier: None | pl.LazyFrame = None,
    part: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if lineitem is None:
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()

    assert lineitem is not None
    assert nation is not None
    assert part is not None
    assert partsupp is not None
    assert supplier is not None

    var1 = date(1996, 1, 1)
    var2 = date(1997, 1, 1)
    var3 = "JORDAN"
    var4 = "azure"

    q1 = (
        lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
        .group_by("l_partkey", "l_suppkey")
        .agg((pl.col("l_quantity").sum() * 0.5).alias("sum_quantity"))
    )
    q2 = nation.filter(pl.col("n_name") == var3)
    q3 = supplier.join(q2, left_on="s_nationkey", right_on="n_nationkey")

    return (
        part.filter(pl.col("p_name").str.starts_with(var4))
        .select(pl.col("p_partkey").unique())
        .join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(
            q1,
            left_on=["ps_suppkey", "p_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .filter(pl.col("ps_availqty") > pl.col("sum_quantity"))
        .select(pl.col("ps_suppkey").unique())
        .join(q3, left_on="ps_suppkey", right_on="s_suppkey")
        .select("s_name", "s_address")
        .sort("s_name")
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
