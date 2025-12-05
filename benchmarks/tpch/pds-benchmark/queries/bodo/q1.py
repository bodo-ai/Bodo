from queries.bodo import utils

import bodo.pandas

Q_NUM = 1


def q(lineitem, pd=bodo.pandas):
    """Pandas code adapted from:
    https://github.com/pola-rs/polars-benchmark/blob/main/queries/dask/q1.py
    """
    var1 = pd.Timestamp("1998-09-02")
    filt = lineitem[lineitem["L_SHIPDATE"] <= var1]

    filt["DISC_PRICE"] = filt.L_EXTENDEDPRICE * (1.0 - filt.L_DISCOUNT)
    filt["CHARGE"] = filt.L_EXTENDEDPRICE * (1.0 - filt.L_DISCOUNT) * (1.0 + filt.L_TAX)

    gb = filt.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)
    agg = gb.agg(
        SUM_QTY=pd.NamedAgg(column="L_QUANTITY", aggfunc="sum"),
        SUM_BASE_PRICE=pd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="sum"),
        SUM_DISC_PRICE=pd.NamedAgg(column="DISC_PRICE", aggfunc="sum"),
        SUM_CHARGE=pd.NamedAgg(column="CHARGE", aggfunc="sum"),
        AVG_QTY=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean"),
        AVG_PRICE=pd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="mean"),
        AVG_DISC=pd.NamedAgg(column="L_DISCOUNT", aggfunc="mean"),
        COUNT_ORDER=pd.NamedAgg(column="L_ORDERKEY", aggfunc="size"),
    )

    result_df = agg.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    return result_df


if __name__ == "__main__":
    utils.run_query(Q_NUM, q)
