from typing import Any

import polars as pl
from queries.polars import utils

Q_NUM = 19


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

    return (
        part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
        .filter(pl.col("l_shipmode").is_in(["AIR", "AIR REG"]))
        .filter(pl.col("l_shipinstruct") == "DELIVER IN PERSON")
        .filter(
            (
                (pl.col("p_brand") == "Brand#12")
                & pl.col("p_container").is_in(
                    ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
                )
                & (pl.col("l_quantity").is_between(1, 11))
                & (pl.col("p_size").is_between(1, 5))
            )
            | (
                (pl.col("p_brand") == "Brand#23")
                & pl.col("p_container").is_in(
                    ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
                )
                & (pl.col("l_quantity").is_between(10, 20))
                & (pl.col("p_size").is_between(1, 10))
            )
            | (
                (pl.col("p_brand") == "Brand#34")
                & pl.col("p_container").is_in(
                    ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
                )
                & (pl.col("l_quantity").is_between(20, 30))
                & (pl.col("p_size").is_between(1, 15))
            )
        )
        .select(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .round(2)
            .alias("revenue")
        )
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
