from __future__ import annotations

import warnings

import pandas as pd
from queries.dask import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Q_NUM = 16


def q() -> None:
    def query() -> pd.DataFrame:
        partsupp = utils.get_part_supp_ds()
        part = utils.get_part_ds()
        supplier = utils.get_supplier_ds()

        supplier["is_complaint"] = supplier.s_comment.str.contains(
            "Customer.*Complaints"
        )
        # We can only broadcast 1 partition series objects
        complaint_suppkeys = supplier[supplier.is_complaint].s_suppkey.repartition(
            npartitions=1
        )
        partsupp = partsupp[~partsupp.ps_suppkey.isin(complaint_suppkeys)]

        table = partsupp.merge(part, left_on="ps_partkey", right_on="p_partkey")
        table = table[
            (table.p_brand != "Brand#45")
            & (~table.p_type.str.startswith("MEDIUM POLISHED"))
            & (table.p_size.isin((49, 14, 23, 45, 19, 3, 36, 9)))
        ]
        return (
            table.groupby(by=["p_brand", "p_type", "p_size"])
            .ps_suppkey.nunique()
            .reset_index()
            .rename(columns={"ps_suppkey": "supplier_cnt"})
            .sort_values(
                by=["supplier_cnt", "p_brand", "p_type", "p_size"],
                ascending=[False, True, True, True],
            )
        ).compute()

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
