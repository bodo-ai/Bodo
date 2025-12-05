from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 14


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

        shipdate_from = datetime.strptime("1995-09-01", "%Y-%m-%d")
        shipdate_to = datetime.strptime("1995-10-01", "%Y-%m-%d")

        table = lineitem.merge(
            part, left_on="l_partkey", right_on="p_partkey", how="inner"
        )
        table = table[
            (table.l_shipdate >= shipdate_from) & (table.l_shipdate < shipdate_to)
        ]

        # Promo revenue by line; CASE clause
        table["promo_revenue"] = table.l_extendedprice * (1 - table.l_discount)
        mask = table.p_type.str.match("PROMO*")
        table["promo_revenue"] = table.promo_revenue.where(mask, 0.00)

        total_promo_revenue = (
            table.promo_revenue.to_frame().sum().reset_index(drop=True)
        )
        total_revenue = (
            (table.l_extendedprice * (1 - table.l_discount))
            .to_frame()
            .sum()
            .reset_index(drop=True)
        )
        # aggregate promo revenue calculation
        return (
            (100.00 * total_promo_revenue / total_revenue)
            .round(2)
            .to_frame("promo_revenue")
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
