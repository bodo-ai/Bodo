from queries.bodo import utils

import bodo.pandas

Q_NUM = 14


def q(lineitem, part, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = pd.Timestamp("1994-03-01")
    var2 = var1 + pd.DateOffset(months=1)

    jn1 = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")

    jn1 = jn1[(jn1["L_SHIPDATE"] >= var1) & (jn1["L_SHIPDATE"] < var2)]

    # Promo revenue by line; CASE clause
    jn1["PROMO_REVENUE"] = jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])
    mask = jn1["P_TYPE"].str.match("PROMO*")
    jn1["PROMO_REVENUE"] = jn1["PROMO_REVENUE"].where(mask, 0.00)

    total_promo_revenue = jn1["PROMO_REVENUE"].sum()
    total_revenue = (jn1["L_EXTENDEDPRICE"] * (1 - jn1["L_DISCOUNT"])).sum()

    # aggregate promo revenue calculation
    ratio = 100.00 * total_promo_revenue / total_revenue
    result_df = pd.DataFrame({"PROMO_REVENUE": [round(ratio, 2)]})

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
