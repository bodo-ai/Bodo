from queries.bodo import utils

import bodo.pandas

Q_NUM = 9


def q(lineitem, orders, part, nation, partsupp, supplier, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "ghost"

    part = part[part.P_NAME.str.contains(var1)]

    jn1 = part.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(
        lineitem,
        left_on=["PS_PARTKEY", "PS_SUPPKEY"],
        right_on=["L_PARTKEY", "L_SUPPKEY"],
    )
    jn4 = jn3.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn5 = jn4.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

    jn5["O_YEAR"] = jn5["O_ORDERDATE"].dt.year
    jn5["NATION"] = jn5["N_NAME"]
    jn5["AMOUNT"] = (
        jn5["L_EXTENDEDPRICE"] * (1 - jn5["L_DISCOUNT"])
        - jn5["PS_SUPPLYCOST"] * jn5["L_QUANTITY"]
    )

    gb = jn5.groupby(["NATION", "O_YEAR"], as_index=False)
    agg = gb.agg(SUM_PROFIT=pd.NamedAgg(column="AMOUNT", aggfunc="sum"))
    agg["SUM_PROFIT"] = agg.SUM_PROFIT.round(2)
    result_df = agg.sort_values(by=["NATION", "O_YEAR"], ascending=[True, False])

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
