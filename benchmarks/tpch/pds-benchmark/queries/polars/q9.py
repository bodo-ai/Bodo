from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 9


def q(
    customer: None | pl.LazyFrame = None,
    lineitem: None | pl.LazyFrame = None,
    nation: None | pl.LazyFrame = None,
    orders: None | pl.LazyFrame = None,
    partsupp: None | pl.LazyFrame = None,
    supplier: None | pl.LazyFrame = None,
    region: None | pl.LazyFrame = None,
    part: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if lineitem is None:
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()
        orders = utils.get_orders_ds()
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
    assert lineitem is not None
    assert nation is not None
    assert orders is not None
    assert part is not None
    assert partsupp is not None
    assert supplier is not None

    return (
        part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("p_name").str.contains("ghost"))
        .select(
            pl.col("n_name").alias("nation"),
            pl.col("o_orderdate").dt.year().alias("o_year"),
            (
                pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
                - pl.col("ps_supplycost") * pl.col("l_quantity")
            ).alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(pl.sum("amount").round(2).alias("sum_profit"))
        .sort(by=["nation", "o_year"], descending=[False, True])
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
