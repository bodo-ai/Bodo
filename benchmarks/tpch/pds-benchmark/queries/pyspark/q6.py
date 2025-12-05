import pandas as pd
from queries.pyspark import utils

Q_NUM = 6


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()

        var1 = pd.Timestamp("1996-01-01")
        var2 = pd.Timestamp("1997-01-01")
        var3 = 0.08
        var4 = 0.1
        var5 = 24

        filt = lineitem[
            (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
        ]
        filt = filt[(filt["L_DISCOUNT"] >= var3) & (filt["L_DISCOUNT"] <= var4)]
        filt = filt[filt["L_QUANTITY"] < var5]
        result_value = (filt["L_EXTENDEDPRICE"] * filt["L_DISCOUNT"]).sum()
        result_df = pd.DataFrame({"REVENUE": [result_value]})

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
