from datetime import date
from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 10


def q(
    customer: None | pl.LazyFrame = None,
    lineitem: None | pl.LazyFrame = None,
    nation: None | pl.LazyFrame = None,
    orders: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if customer is None:
        customer = utils.get_customer_ds()
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()
        orders = utils.get_orders_ds()

    assert customer is not None
    assert lineitem is not None
    assert nation is not None
    assert orders is not None

    var1 = date(1994, 11, 1)
    var2 = date(1995, 2, 1)

    return (
        customer.join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
        .filter(pl.col("l_returnflag") == "R")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .round(2)
            .alias("revenue")
        )
        .select(
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
        .sort(by="revenue", descending=True)
        .head(20)
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
