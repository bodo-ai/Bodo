from queries.pyspark import utils

Q_NUM = 2


def q() -> None:
    def query_func():
        part = utils.get_part_ds()
        partsupp = utils.get_part_supp_ds()
        supplier = utils.get_supplier_ds()
        nation = utils.get_nation_ds()
        region = utils.get_region_ds()

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        jn = (
            part.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
            .merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
            .merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
            .merge(region, left_on="N_REGIONKEY", right_on="R_REGIONKEY")
        )

        jn = jn[jn["P_SIZE"] == var1]
        jn = jn[jn["P_TYPE"].str.endswith(var2)]
        jn = jn[jn["R_NAME"] == var3]

        gb = jn.groupby("P_PARTKEY", as_index=False)
        agg = gb["PS_SUPPLYCOST"].min()
        jn2 = agg.merge(jn, on=["P_PARTKEY", "PS_SUPPLYCOST"])

        sel = jn2.loc[
            :,
            [
                "S_ACCTBAL",
                "S_NAME",
                "N_NAME",
                "P_PARTKEY",
                "P_MFGR",
                "S_ADDRESS",
                "S_PHONE",
                "S_COMMENT",
            ],
        ]

        sort = sel.sort_values(
            by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
            ascending=[False, True, True, True],
        )
        result_df = sort.head(100)

        return result_df

    _ = utils.get_or_create_spark()

    utils.run_query(Q_NUM, query_func)


if __name__ == "__main__":
    q()
