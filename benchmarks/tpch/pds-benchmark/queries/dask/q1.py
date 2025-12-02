from __future__ import annotations

from datetime import date

import pandas as pd
from queries.dask import utils

Q_NUM = 1


def q() -> None:
    def query() -> pd.DataFrame:
        line_item_ds = utils.get_line_item_ds()

        var1 = date(1998, 9, 2)

        filt = line_item_ds[line_item_ds["l_shipdate"] <= var1]

        # This is lenient towards pandas as normally an optimizer should decide
        # that this could be computed before the groupby aggregation.
        # Other implementations don't enjoy this benefit.
        filt["disc_price"] = filt.l_extendedprice * (1.0 - filt.l_discount)
        filt["charge"] = (
            filt.l_extendedprice * (1.0 - filt.l_discount) * (1.0 + filt.l_tax)
        )

        # `groupby(as_index=False)` is not yet implemented by Dask:
        # https://github.com/dask/dask/issues/5834
        gb = filt.groupby(["l_returnflag", "l_linestatus"])
        agg = gb.agg(
            sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
            sum_base_price=pd.NamedAgg(column="l_extendedprice", aggfunc="sum"),
            sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
            sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
            avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
            avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
            avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
            count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
        ).reset_index()

        result_df = agg.sort_values(["l_returnflag", "l_linestatus"])

        return result_df.compute()  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
