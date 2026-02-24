from queries.bodo import utils

import bodo.pandas

Q_NUM = 4


def q(lineitem, orders, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = pd.Timestamp("1993-11-01")
    var2 = pd.Timestamp("1993-08-01")

    flineitem = lineitem[lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE]
    forders = orders[(orders.O_ORDERDATE < var1) & (orders.O_ORDERDATE >= var2)]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = (
        jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"]
        .count()
        .sort_values(["O_ORDERPRIORITY"])
    )
    total.columns = ["O_ORDERPRIORITY", "ORDER_COUNT"]
    return total


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
