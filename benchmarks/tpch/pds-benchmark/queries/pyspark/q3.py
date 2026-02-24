import pandas as pd
from queries.pyspark import utils

Q_NUM = 3


def q() -> None:
    def query_func():
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()

        var1 = "HOUSEHOLD"
        var2 = pd.Timestamp("1995-03-04")

        fcustomer = customer[customer["C_MKTSEGMENT"] == var1]

        jn1 = fcustomer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
        jn2 = jn1.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")

        jn2 = jn2[jn2["O_ORDERDATE"] < var2]
        jn2 = jn2[jn2["L_SHIPDATE"] > var2]
        jn2["REVENUE"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)

        gb = jn2.groupby(
            ["O_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False
        )
        agg = gb["REVENUE"].sum()

        sel = agg.loc[:, ["O_ORDERKEY", "REVENUE", "O_ORDERDATE", "O_SHIPPRIORITY"]]
        sel = sel.rename(columns={"O_ORDERKEY": "L_ORDERKEY"})

        sorted = sel.sort_values(by=["REVENUE", "O_ORDERDATE"], ascending=[False, True])
        result_df = sorted.head(10)

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
