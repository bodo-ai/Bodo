from queries.bodo import utils

import bodo.pandas

Q_NUM = 2


def q(part, partsupp, supplier, nation, region, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q2.py
    """
    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    jn = (
        part.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
        .merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
        .merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
        .merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")
    )

    jn = jn[jn["P_SIZE"] == var1]
    jn = jn[jn["P_TYPE"].str.endswith(var2)]
    jn = jn[jn["R_NAME"] == var3]

    gb = jn.groupby("P_PARTKEY", as_index=False)
    agg = gb["PS_SUPPLYCOST"].min()
    jn2 = agg.merge(jn, on=["P_PARTKEY", "PS_SUPPLYCOST"])

    sel = jn2.loc[
        :,
        [
            "S_ACCTBAL",
            "S_NAME",
            "N_NAME",
            "P_PARTKEY",
            "P_MFGR",
            "S_ADDRESS",
            "S_PHONE",
            "S_COMMENT",
        ],
    ]

    sort = sel.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    result_df = sort.head(100)

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
