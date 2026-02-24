from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 17


def q() -> None:
    def query() -> pd.DataFrame:
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

        joined = lineitem.merge(
            part, left_on="l_partkey", right_on="p_partkey", how="inner"
        )
        joined = joined[joined.p_brand == "Brand#23"]
        joined = joined[joined.p_container == "MED BOX"]
        avg_qnty_by_partkey = (
            joined.groupby("l_partkey")
            .l_quantity.mean()
            .to_frame()
            .rename(columns={"l_quantity": "l_quantity_avg"})
        )
        table = joined.merge(
            avg_qnty_by_partkey, left_on="l_partkey", right_index=True, how="left"
        )

        table = table[table.l_quantity < 0.2 * table.l_quantity_avg]
        return (
            (table.l_extendedprice.to_frame().sum() / 7.0)
            .round(2)
            .to_frame("avg_yearly")
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
