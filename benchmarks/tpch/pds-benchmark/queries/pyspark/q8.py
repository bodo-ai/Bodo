import pandas as pd
from queries.pyspark import utils

Q_NUM = 8


def q() -> None:
    def query_func():
        part = utils.get_part_ds()
        lineitem = utils.get_line_item_ds()
        supplier = utils.get_supplier_ds()
        orders = utils.get_orders_ds()
        customer = utils.get_customer_ds()
        nation = utils.get_nation_ds()
        region = utils.get_region_ds()

        var1 = "BRAZIL"
        var2 = "AMERICA"
        var3 = "ECONOMY ANODIZED STEEL"
        var4 = pd.Timestamp("1995-01-01")
        var5 = pd.Timestamp("1997-01-01")

        n1 = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
        n2 = nation.loc[:, ["N_NATIONKEY", "N_NAME"]]

        jn1 = part.merge(lineitem, left_on="P_PARTKEY", right_on="L_PARTKEY")
        jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
        jn3 = jn2.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
        jn4 = jn3.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
        jn5 = jn4.merge(n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
        jn6 = jn5.merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")

        jn6 = jn6[(jn6["R_NAME"] == var2)]

        jn7 = jn6.merge(n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

        jn7 = jn7[(jn7["O_ORDERDATE"] >= var4) & (jn7["O_ORDERDATE"] < var5)]
        jn7 = jn7[jn7["P_TYPE"] == var3]

        jn7["O_YEAR"] = jn7["O_ORDERDATE"].dt.year
        jn7["VOLUME"] = jn7["L_EXTENDEDPRICE"] * (1.0 - jn7["L_DISCOUNT"])
        jn7 = jn7.rename(columns={"N_NAME": "NATION"})

        # denominator: total volume per year
        denom = (
            jn7.groupby("O_YEAR", as_index=False)["VOLUME"]
            .sum()
            .rename(columns={"VOLUME": "TOTAL_VOLUME"})
        )

        # numerator: Brazil volume per year
        num = (
            jn7[jn7["NATION"] == var1]
            .groupby("O_YEAR", as_index=False)["VOLUME"]
            .sum()
            .rename(columns={"VOLUME": "BRAZIL_VOLUME"})
        )

        # join and compute ratio
        agg = denom.merge(num, on="O_YEAR", how="left")
        agg["MKT_SHARE"] = (agg["BRAZIL_VOLUME"] / agg["TOTAL_VOLUME"]).round(2)

        result_df = agg.sort_values("O_YEAR")
        result_df = result_df[["O_YEAR", "MKT_SHARE"]]

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
