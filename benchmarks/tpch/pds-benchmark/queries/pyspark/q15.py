import pandas as pd
from queries.pyspark import utils

Q_NUM = 15


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()

        var1 = pd.Timestamp("1996-01-01")
        var2 = var1 + pd.DateOffset(months=3)

        jn1 = lineitem[
            (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
        ]

        jn1["REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])

        agg = jn1.groupby("L_SUPPKEY").agg(
            TOTAL_REVENUE=pd.NamedAgg(column="REVENUE", aggfunc="sum")
        )
        agg = agg.reset_index()
        revenue = agg.rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})

        jn2 = supplier.merge(
            revenue, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
        )

        max_revenue = revenue["TOTAL_REVENUE"].max()
        jn2 = jn2[jn2["TOTAL_REVENUE"] == max_revenue]

        result_df = jn2[
            ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
        ].sort_values(by="S_SUPPKEY")

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
