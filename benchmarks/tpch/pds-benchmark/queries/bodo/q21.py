from queries.bodo import utils

import bodo.pandas

Q_NUM = 21


def q(lineitem, orders, supplier, nation, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "SAUDI ARABIA"

    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False).agg(
        NUM_SUPPLIERS=pd.NamedAgg(column="L_SUPPKEY", aggfunc="nunique")
    )
    gb1 = gb1[gb1["NUM_SUPPLIERS"] > 1]

    flineitem = lineitem[lineitem["L_RECEIPTDATE"] > lineitem["L_COMMITDATE"]]
    jn1 = gb1.merge(flineitem, on="L_ORDERKEY")

    gb2 = jn1.groupby("L_ORDERKEY", as_index=False).agg(
        NUNIQUE_COL=pd.NamedAgg(column="L_SUPPKEY", aggfunc="nunique")
    )

    jn2 = gb2.merge(jn1, on="L_ORDERKEY")
    jn3 = jn2.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn5 = jn4.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn5 = jn5[
        (
            (jn5["NUNIQUE_COL"] == 1)
            & (jn5["N_NAME"] == var1)
            & (jn5["O_ORDERSTATUS"] == "F")
        )
    ]
    gb3 = jn5.groupby("S_NAME", as_index=False).agg(
        NUMWAIT=pd.NamedAgg(column="NUNIQUE_COL", aggfunc="size")
    )

    result_df = gb3.sort_values(["NUMWAIT", "S_NAME"], ascending=[False, True]).head(
        100
    )

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
