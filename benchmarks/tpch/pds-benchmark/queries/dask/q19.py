from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 19


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

        table = lineitem.merge(part, left_on="l_partkey", right_on="p_partkey")
        table = table[
            (
                (table.p_brand == "Brand#12")
                & (table.p_container.isin(("SM CASE", "SM BOX", "SM PACK", "SM PKG")))
                & ((table.l_quantity >= 1) & (table.l_quantity <= 1 + 10))
                & (table.p_size.between(1, 5))
                & (table.l_shipmode.isin(("AIR", "AIR REG")))
                & (table.l_shipinstruct == "DELIVER IN PERSON")
            )
            | (
                (table.p_brand == "Brand#23")
                & (
                    table.p_container.isin(
                        ("MED BAG", "MED BOX", "MED PKG", "MED PACK")
                    )
                )
                & ((table.l_quantity >= 10) & (table.l_quantity <= 20))
                & (table.p_size.between(1, 10))
                & (table.l_shipmode.isin(("AIR", "AIR REG")))
                & (table.l_shipinstruct == "DELIVER IN PERSON")
            )
            | (
                (table.p_brand == "Brand#34")
                & (table.p_container.isin(("LG CASE", "LG BOX", "LG PACK", "LG PKG")))
                & ((table.l_quantity >= 20) & (table.l_quantity <= 30))
                & (table.p_size.between(1, 15))
                & (table.l_shipmode.isin(("AIR", "AIR REG")))
                & (table.l_shipinstruct == "DELIVER IN PERSON")
            )
        ]
        return (
            (table.l_extendedprice * (1 - table.l_discount))
            .to_frame()
            .sum()
            .round(2)
            .to_frame("revenue")
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
