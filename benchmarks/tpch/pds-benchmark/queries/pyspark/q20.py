import pandas as pd
from queries.pyspark import utils

Q_NUM = 20


def q() -> None:
    def query_func():
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()
        partsupp = utils.get_part_supp_ds()
        part = utils.get_part_ds()
        nation = utils.get_nation_ds()

        var1 = pd.Timestamp("1996-01-01")
        var2 = pd.Timestamp("1997-01-01")
        var3 = "JORDAN"
        var4 = "azure"

        flineitem = lineitem[
            (lineitem["L_SHIPDATE"] >= var1) & (lineitem["L_SHIPDATE"] < var2)
        ]
        agg = flineitem.groupby(["L_SUPPKEY", "L_PARTKEY"]).agg(
            SUM_QUANTITY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum")
        )
        agg = agg.reset_index()
        agg["SUM_QUANTITY"] = agg["SUM_QUANTITY"] * 0.5

        fnation = nation[nation["N_NAME"] == var3]

        jn1 = supplier.merge(fnation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

        fpart = part[part["P_NAME"].str.startswith(var4)]

        jn2 = partsupp.merge(fpart, left_on="PS_PARTKEY", right_on="P_PARTKEY")
        jn3 = jn2.merge(
            agg,
            left_on=["PS_SUPPKEY", "PS_PARTKEY"],
            right_on=["L_SUPPKEY", "L_PARTKEY"],
        )
        jn3 = jn3[jn3["PS_AVAILQTY"] > jn3["SUM_QUANTITY"]]
        jn4 = jn1.merge(jn3, left_on="S_SUPPKEY", right_on="PS_SUPPKEY")

        result_df = jn4[["S_NAME", "S_ADDRESS"]].sort_values("S_NAME", ascending=True)

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
