from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils
from settings import Settings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 11

settings = Settings()


def q() -> None:
    def query() -> pd.DataFrame:
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
        nation = utils.get_nation_ds()

        joined = partsupp.merge(
            supplier, left_on="ps_suppkey", right_on="s_suppkey", how="inner"
        ).merge(nation, left_on="s_nationkey", right_on="n_nationkey", how="inner")
        joined = joined[joined.n_name == "GERMANY"]

        threshold = (
            (joined.ps_supplycost * joined.ps_availqty).sum()
            * 0.0001
            / settings.scale_factor
        )

        joined["value"] = joined.ps_supplycost * joined.ps_availqty

        res = joined.groupby("ps_partkey")["value"].sum()
        res = (
            res[res > threshold]
            .round(2)
            .reset_index()
            .sort_values(by="value", ascending=False)
        )

        return res.compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
