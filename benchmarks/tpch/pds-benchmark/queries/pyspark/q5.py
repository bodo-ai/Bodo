import pandas as pd
from queries.pyspark import utils

Q_NUM = 5


def q() -> None:
    def query_func():
        region = utils.get_region_ds()
        nation = utils.get_nation_ds()
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()

        var1 = "ASIA"
        var2 = pd.Timestamp("1996-01-01")
        var3 = pd.Timestamp("1997-01-01")

        jn1 = region.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
        jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
        jn3 = jn2.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
        jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
        jn5 = jn4.merge(
            supplier,
            left_on=["L_SUPPKEY", "N_NATIONKEY"],
            right_on=["S_SUPPKEY", "S_NATIONKEY"],
        )

        jn5 = jn5[jn5["R_NAME"] == var1]
        jn5 = jn5[(jn5["O_ORDERDATE"] >= var2) & (jn5["O_ORDERDATE"] < var3)]
        jn5["REVENUE"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)

        gb = jn5.groupby("N_NAME", as_index=False)["REVENUE"].sum()
        result_df = gb.sort_values("REVENUE", ascending=False)

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
