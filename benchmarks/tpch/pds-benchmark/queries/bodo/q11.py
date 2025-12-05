from queries.bodo import utils

import bodo.pandas

Q_NUM = 11


def q(partsupp, supplier, nation, scale_factor=1.0, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "GERMANY"
    var2 = 0.0001 / scale_factor

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


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
