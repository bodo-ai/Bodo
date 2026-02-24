from queries.bodo import utils

import bodo.pandas

Q_NUM = 13


def q(customer, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    var1 = "special"
    var2 = "requests"

    orders = orders[~orders["O_COMMENT"].str.contains(f"{var1}.*{var2}")]

    jn1 = customer.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left")

    agg1 = jn1.groupby("C_CUSTKEY", as_index=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    agg2 = agg1.groupby("C_COUNT", as_index=False).agg(
        CUSTDIST=pd.NamedAgg(column="C_CUSTKEY", aggfunc="size")
    )

    result_df = agg2.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
