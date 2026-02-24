from __future__ import annotations

from typing import TYPE_CHECKING

from queries.dask import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 2


def q() -> None:
    def query() -> pd.DataFrame:
        region_ds = utils.get_region_ds()
        nation_filtered = utils.get_nation_ds()
        supplier_filtered = utils.get_supplier_ds()
        part_filtered = utils.get_part_ds()
        partsupp_filtered = utils.get_part_supp_ds()

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        region_filtered = region_ds[(region_ds["r_name"] == var3)]
        r_n_merged = nation_filtered.merge(
            region_filtered, left_on="n_regionkey", right_on="r_regionkey", how="inner"
        )
        s_r_n_merged = r_n_merged.merge(
            supplier_filtered,
            left_on="n_nationkey",
            right_on="s_nationkey",
            how="inner",
        )
        ps_s_r_n_merged = s_r_n_merged.merge(
            partsupp_filtered, left_on="s_suppkey", right_on="ps_suppkey", how="inner"
        )
        part_filtered = part_filtered[
            (part_filtered["p_size"] == var1)
            & (part_filtered["p_type"].str.endswith(var2))
        ]
        merged_df = part_filtered.merge(
            ps_s_r_n_merged, left_on="p_partkey", right_on="ps_partkey", how="inner"
        )
        min_values = merged_df.groupby("p_partkey")["ps_supplycost"].min().reset_index()
        min_values.columns = ["P_PARTKEY_CPY", "MIN_SUPPLYCOST"]
        merged_df = merged_df.merge(
            min_values,
            left_on=["p_partkey", "ps_supplycost"],
            right_on=["P_PARTKEY_CPY", "MIN_SUPPLYCOST"],
            how="inner",
        )
        return (
            merged_df[
                [
                    "s_acctbal",
                    "s_name",
                    "n_name",
                    "p_partkey",
                    "p_mfgr",
                    "s_address",
                    "s_phone",
                    "s_comment",
                ]
            ]
            .sort_values(
                by=[
                    "s_acctbal",
                    "n_name",
                    "s_name",
                    "p_partkey",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
            )
            .head(100, compute=True)
        )

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
