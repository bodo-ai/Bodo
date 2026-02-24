import pandas as pd
import pyspark.pandas as ps
from queries.pyspark import utils

Q_NUM = 7


def q() -> None:
    def query_func():
        customer = utils.get_customer_ds()
        orders = utils.get_orders_ds()
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()
        nation = utils.get_nation_ds()

        var1 = "FRANCE"
        var2 = "GERMANY"
        var3 = pd.Timestamp("1995-01-01")
        var4 = pd.Timestamp("1997-01-01")

        n1 = nation[(nation["N_NAME"] == var1)]
        n2 = nation[(nation["N_NAME"] == var2)]

        # Part 1
        jn1 = customer.merge(n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
        jn2 = jn1.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
        jn2 = jn2.rename(columns={"N_NAME": "CUST_NATION"})
        jn3 = jn2.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
        jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
        jn5 = jn4.merge(n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
        df1 = jn5.rename(columns={"N_NAME": "SUPP_NATION"})

        # Part 2
        jn1 = customer.merge(n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
        jn2 = jn1.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
        jn2 = jn2.rename(columns={"N_NAME": "CUST_NATION"})
        jn3 = jn2.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
        jn4 = jn3.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
        jn5 = jn4.merge(n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
        df2 = jn5.rename(columns={"N_NAME": "SUPP_NATION"})

        # Combine
        total = ps.concat([df1, df2])

        total = total[(total["L_SHIPDATE"] >= var3) & (total["L_SHIPDATE"] < var4)]
        total["VOLUME"] = total["L_EXTENDEDPRICE"] * (1.0 - total["L_DISCOUNT"])
        total["L_YEAR"] = total["L_SHIPDATE"].dt.year

        gb = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"])
        agg = gb.agg(REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum"))
        agg = agg.reset_index()

        result_df = agg.sort_values(by=["SUPP_NATION", "CUST_NATION", "L_YEAR"])
        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
