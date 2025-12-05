import pandas as pd
from queries.pyspark import utils

Q_NUM = 19


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()
        part = utils.get_part_ds()

        jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
        jn1 = jn1[
            (
                (jn1["P_BRAND"] == "Brand#31")
                & (jn1["P_CONTAINER"].isin(("SM CASE", "SM BOX", "SM PACK", "SM PKG")))
                & ((jn1["L_QUANTITY"] >= 4) & (jn1["L_QUANTITY"] <= 14))
                & (jn1["P_SIZE"] <= 5)
                & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
                & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
            )
            | (
                (jn1["P_BRAND"] == "Brand#43")
                & (
                    jn1["P_CONTAINER"].isin(
                        ("MED BAG", "MED BOX", "MED PKG", "MED PACK")
                    )
                )
                & ((jn1["L_QUANTITY"] >= 15) & (jn1["L_QUANTITY"] <= 25))
                & ((jn1["P_SIZE"] >= 1) & (jn1["P_SIZE"] <= 10))
                & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
                & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
            )
            | (
                (jn1["P_BRAND"] == "Brand#43")
                & (jn1["P_CONTAINER"].isin(("LG CASE", "LG BOX", "LG PACK", "LG PKG")))
                & ((jn1["L_QUANTITY"] >= 26) & (jn1["L_QUANTITY"] <= 36))
                & (jn1["P_SIZE"] <= 15)
                & (jn1["L_SHIPMODE"].isin(("AIR", "AIR REG")))
                & (jn1["L_SHIPINSTRUCT"] == "DELIVER IN PERSON")
            )
        ]

        total = (jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])).sum()

        result_df = pd.DataFrame({"REVENUE": [round(total, 2)]})

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
