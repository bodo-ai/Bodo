from queries.bodo import utils

import bodo.pandas

Q_NUM = 18


def q(lineitem, orders, customer, pd=bodo.pandas):
    """Adapted from:
    github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = 300

    agg1 = lineitem.groupby("L_ORDERKEY", as_index=False, sort=False)[
        "L_QUANTITY"
    ].sum()
    filt = agg1[agg1.L_QUANTITY > var1]
    jn1 = filt.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    agg2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
        sort=False,
    )["L_QUANTITY"].sum()
    total = agg2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    return total.head(100)


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
