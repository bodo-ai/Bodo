import pandas as pd
from queries.pyspark import utils

Q_NUM = 14


def q() -> None:
    def query_func():
        part = utils.get_part_ds()
        lineitem = utils.get_line_item_ds()

        var1 = pd.Timestamp("1994-03-01")
        var2 = var1 + pd.DateOffset(months=1)

        jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")

        jn1 = jn1[(jn1["L_SHIPDATE"] >= var1) & (jn1["L_SHIPDATE"] < var2)]

        # Promo revenue by line; CASE clause
        jn1["PROMO_REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])
        mask = jn1["P_TYPE"].str.match("PROMO*")
        jn1["PROMO_REVENUE"] = jn1["PROMO_REVENUE"].where(mask, 0.00)

        total_promo_revenue = jn1["PROMO_REVENUE"].sum()
        total_revenue = (jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])).sum()

        # aggregate promo revenue calculation
        ratio = 100.00 * total_promo_revenue / total_revenue
        result_df = pd.DataFrame({"PROMO_REVENUE": [round(ratio, 2)]})

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
