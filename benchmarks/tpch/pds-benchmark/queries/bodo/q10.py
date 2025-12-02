from queries.bodo import utils

import bodo.pandas

Q_NUM = 10


def q(lineitem, orders, customer, nation, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1994-11-01")
    var2 = pd.Timestamp("1995-02-01")

    forders = orders[(orders.O_ORDERDATE >= var1) & (orders.O_ORDERDATE < var2)]
    flineitem = lineitem[lineitem.L_RETURNFLAG == "R"]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["REVENUE"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    agg = jn3.groupby(
        [
            "C_CUSTKEY",
            "C_NAME",
            "C_ACCTBAL",
            "C_PHONE",
            "N_NAME",
            "C_ADDRESS",
            "C_COMMENT",
        ],
        as_index=False,
        sort=False,
    )["REVENUE"].sum()
    agg["REVENUE"] = agg.REVENUE.round(2)
    total = agg.sort_values("REVENUE", ascending=False)
    return total.head(20)


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
