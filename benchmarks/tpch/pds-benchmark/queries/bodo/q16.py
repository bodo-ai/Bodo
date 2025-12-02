from queries.bodo import utils

import bodo.pandas

Q_NUM = 16


def q(part, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "Brand#45"

    supplier["IS_COMPLAINT"] = supplier["S_COMMENT"].str.contains(
        "Customer.*Complaints"
    )

    complaint_suppkeys = supplier[supplier["IS_COMPLAINT"]]["S_SUPPKEY"]

    jn1 = partsupp[~partsupp["PS_SUPPKEY"].isin(complaint_suppkeys)]
    jn2 = jn1.merge(part, left_on="PS_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn2[
        (jn2["P_BRAND"] != var1)
        & (~jn2["P_TYPE"].str.startswith("MEDIUM POLISHED"))
        & (jn2["P_SIZE"].isin((49, 14, 23, 45, 19, 3, 36, 9)))
    ]

    agg = jn2.groupby(by=["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)[
        "PS_SUPPKEY"
    ].nunique()
    agg = agg.rename(columns={"PS_SUPPKEY": "SUPPLIER_CNT"})

    result_df = agg.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
