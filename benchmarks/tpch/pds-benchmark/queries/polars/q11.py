from typing import Any

import polars as pl
from queries.polars import utils
from settings import Settings

settings = Settings()

Q_NUM = 11


def q(
    nation: None | pl.LazyFrame = None,
    partsupp: None | pl.LazyFrame = None,
    supplier: None | pl.LazyFrame = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    if nation is None:
        nation = utils.get_nation_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()

    assert nation is not None
    assert partsupp is not None
    assert supplier is not None

    var1 = "GERMANY"
    var2 = 0.0001 / settings.scale_factor

    q1 = (
        partsupp.join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("n_name") == var1)
    )
    q2 = q1.select(
        (pl.col("ps_supplycost") * pl.col("ps_availqty")).sum().round(2).alias("tmp")
        * var2
    )

    return (
        q1.group_by("ps_partkey")
        .agg(
            (pl.col("ps_supplycost") * pl.col("ps_availqty"))
            .sum()
            .round(2)
            .alias("value")
        )
        .join(q2, how="cross")
        .filter(pl.col("value") > pl.col("tmp"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
