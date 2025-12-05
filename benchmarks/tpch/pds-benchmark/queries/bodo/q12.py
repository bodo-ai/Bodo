from queries.bodo import utils

import bodo.pandas

Q_NUM = 12


def q(lineitem, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/xorbitsai/benchmarks/blob/main/tpch/pandas_queries/queries.py
    """
    var1 = pd.Timestamp("1994-01-01")
    var2 = pd.Timestamp("1995-01-01")

    jn1 = orders.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn1 = jn1[
        (jn1["L_SHIPMODE"].isin(("MAIL", "SHIP")))
        & (jn1["L_COMMITDATE"] < jn1["L_RECEIPTDATE"])
        & (jn1["L_SHIPDATE"] < jn1["L_COMMITDATE"])
        & (jn1["L_RECEIPTDATE"] >= var1)
        & (jn1["L_RECEIPTDATE"] < var2)
    ]

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    gb = jn1.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
    result_df = gb.sort_values("L_SHIPMODE")

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
