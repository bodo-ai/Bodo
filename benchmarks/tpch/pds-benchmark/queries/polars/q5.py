from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 5


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
    if customer is None:
        customer = utils.get_customer_ds()
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()
        orders = utils.get_orders_ds()
        region = utils.get_region_ds()
        supplier = utils.get_supplier_ds()

    assert region is not None
    assert customer is not None
    assert lineitem is not None
    assert supplier is not None
    assert nation is not None
    assert orders is not None

    var1 = "ASIA"
    var2 = date(1996, 1, 1)
    var3 = date(1997, 1, 1)

    return (
        region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(pl.col("r_name") == var1)
        .filter(pl.col("o_orderdate").is_between(var2, var3, closed="left"))
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg(pl.sum("revenue"))
        .sort(by="revenue", descending=True)
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
