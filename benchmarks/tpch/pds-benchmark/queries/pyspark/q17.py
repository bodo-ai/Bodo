import pandas as pd
from queries.pyspark import utils

Q_NUM = 17


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

        var1 = "Brand#23"
        var2 = "MED BOX"

        jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
        jn1 = jn1[((jn1["P_BRAND"] == var1) & (jn1["P_CONTAINER"] == var2))]

        agg = jn1.groupby("L_PARTKEY").agg(
            L_QUANTITY_AVG=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
        )
        agg = agg.reset_index()

        jn4 = jn1.merge(agg, left_on="L_PARTKEY", right_on="L_PARTKEY", how="left")
        jn4 = jn4[jn4["L_QUANTITY"] < 0.2 * jn4["L_QUANTITY_AVG"]]
        total = jn4["L_EXTENDEDPRICE"].sum() / 7.0

        result_df = pd.DataFrame({"AVG_YEARLY": [round(total, 2)]})

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
