import pandas as pd
from queries.pyspark import utils

Q_NUM = 10


def q() -> None:
    def query_func():
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()
        nation = utils.get_nation_ds()

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
        )["REVENUE"].sum()
        agg = agg.reset_index()
        agg["REVENUE"] = agg.REVENUE.round(2)
        total = agg.sort_values("REVENUE", ascending=False)
        return total.head(20)

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
