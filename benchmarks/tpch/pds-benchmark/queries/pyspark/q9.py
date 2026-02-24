import pandas as pd
from queries.pyspark import utils

Q_NUM = 9


def q() -> None:
    def query_func():
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
        lineitem = utils.get_line_item_ds()
        orders = utils.get_orders_ds()
        nation = utils.get_nation_ds()

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

        gb = jn5.groupby(["NATION", "O_YEAR"])
        agg = gb.agg(SUM_PROFIT=pd.NamedAgg(column="AMOUNT", aggfunc="sum"))
        agg = agg.reset_index()
        agg["SUM_PROFIT"] = agg.SUM_PROFIT.round(2)
        result_df = agg.sort_values(by=["NATION", "O_YEAR"], ascending=[True, False])

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
