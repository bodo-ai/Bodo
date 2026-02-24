from queries.bodo import utils

import bodo.pandas

Q_NUM = 22


def q(customer, orders, pd=bodo.pandas):
    """Adapted from:
    https://github.com/coiled/benchmarks/blob/13ebb9c72b1941c90b602e3aaea82ac18fafcddc/tests/tpch/dask_queries.py
    """
    customer["CNTRYCODE"] = customer["C_PHONE"].str.strip().str.slice(0, 2)
    fcustomers = customer[
        customer["CNTRYCODE"].isin(("13", "31", "23", "29", "30", "18", "17"))
    ]

    average_c_acctbal = fcustomers[fcustomers["C_ACCTBAL"] > 0.0]["C_ACCTBAL"].mean()
    custsale = fcustomers[fcustomers["C_ACCTBAL"] > average_c_acctbal]

    jn1 = custsale.merge(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left")
    jn1 = jn1[jn1["O_CUSTKEY"].isnull()]

    agg1 = jn1.groupby("CNTRYCODE", as_index=False).agg(
        NUMCUST=pd.NamedAgg(column="C_ACCTBAL", aggfunc="size"),
        TOTACCTBAL=pd.NamedAgg(column="C_ACCTBAL", aggfunc="sum"),
    )

    result_df = agg1.sort_values("CNTRYCODE", ascending=True)

    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
