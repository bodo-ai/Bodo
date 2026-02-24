from queries.pyspark import utils

Q_NUM = 18


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()
        customer = utils.get_customer_ds()

        var1 = 300

        agg1 = lineitem.groupby("L_ORDERKEY", as_index=False)["L_QUANTITY"].sum()
        filt = agg1[agg1.L_QUANTITY > var1]
        jn1 = filt.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
        jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
        agg2 = jn2.groupby(
            ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
            as_index=False,
        )["L_QUANTITY"].sum()
        total = agg2.sort_values(
            ["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True]
        )
        return total.head(100)

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
