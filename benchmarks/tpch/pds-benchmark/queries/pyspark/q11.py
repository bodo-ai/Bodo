from queries.pyspark import utils
from settings import Settings

Q_NUM = 11

settings = Settings()


def q() -> None:
    def query_func():
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
        nation = utils.get_nation_ds()

        var1 = "GERMANY"
        var2 = 0.0001 / settings.scale_factor

        jn1 = partsupp.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
        jn2 = jn1.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")

        jn2 = jn2[jn2["N_NAME"] == var1]

        threshold = (jn2["PS_SUPPLYCOST"] * jn2["PS_AVAILQTY"]).sum() * var2

        jn2["VALUE"] = jn2["PS_SUPPLYCOST"] * jn2["PS_AVAILQTY"]

        gb = jn2.groupby("PS_PARTKEY", as_index=False)["VALUE"].sum()

        filt = gb[gb["VALUE"] > threshold]
        filt["VALUE"] = filt.VALUE.round(2)
        result_df = filt.sort_values(by="VALUE", ascending=False)

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
