from queries.bodo import utils

import bodo.pandas

Q_NUM = 17


def q(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "Brand#23"
    var2 = "MED BOX"

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn1 = jn1[((jn1["P_BRAND"] == var1) & (jn1["P_CONTAINER"] == var2))]

    agg = jn1.groupby("L_PARTKEY", as_index=False).agg(
        L_QUANTITY_AVG=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )

    jn4 = jn1.merge(agg, left_on="L_PARTKEY", right_on="L_PARTKEY", how="left")
    jn4 = jn4[jn4["L_QUANTITY"] < 0.2 * jn4["L_QUANTITY_AVG"]]
    total = jn4["L_EXTENDEDPRICE"].sum() / 7.0

    result_df = pd.DataFrame({"AVG_YEARLY": [round(total, 2)]})

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
