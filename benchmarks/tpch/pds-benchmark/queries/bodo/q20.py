from queries.bodo import utils

import bodo.pandas

Q_NUM = 20


def q(lineitem, part, nation, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1996-01-01")
    var2 = pd.Timestamp("1997-01-01")
    var3 = "JORDAN"
    var4 = "azure"

    flineitem = lineitem[
        (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
    ]
    agg = flineitem.groupby(["L_SUPPKEY", "L_PARTKEY"], as_index=False).agg(
        SUM_QUANTITY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum")
    )
    agg["SUM_QUANTITY"] = agg["SUM_QUANTITY"] * 0.5

    fnation = nation[nation["N_NAME"] == var3]

    jn1 = supplier.merge(fnation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    fpart = part[part["P_NAME"].str.startswith(var4)]

    jn2 = partsupp.merge(fpart, left_on="PS_PARTKEY", right_on="P_PARTKEY")
    jn3 = jn2.merge(
        agg,
        left_on=["PS_SUPPKEY", "PS_PARTKEY"],
        right_on=["L_SUPPKEY", "L_PARTKEY"],
    )
    jn3 = jn3[jn3["PS_AVAILQTY"] > jn3["SUM_QUANTITY"]]
    jn4 = jn1.merge(jn3, left_on="S_SUPPKEY", right_on="PS_SUPPKEY")

    result_df = jn4[["S_NAME", "S_ADDRESS"]].sort_values("S_NAME", ascending=True)

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
