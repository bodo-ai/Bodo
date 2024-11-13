import pandas as pd
import bodo
import os

SNOWSQL_CRED = os.getenv("SNOWSQL_CRED")


@bodo.jit(cache=True)
def q3():
    date = pd.to_datetime("1995-03-04").date()
    lineitem = pd.read_sql(
        "SELECT * FROM LINEITEM",
        "snowflake://"
        + SNOWSQL_CRED
        + "@bodopartner.us-east-1/SNOWFLAKE_SAMPLE_DATA/TPCH_SF1?warehouse=DEMO_WH",
    )
    orders = pd.read_sql(
        "SELECT * FROM ORDERS",
        "snowflake://"
        + SNOWSQL_CRED
        + "@bodopartner.us-east-1/SNOWFLAKE_SAMPLE_DATA/TPCH_SF1?warehouse=DEMO_WH",
    )
    customer = pd.read_sql(
        "SELECT * FROM CUSTOMER",
        "snowflake://"
        + SNOWSQL_CRED
        + "@bodopartner.us-east-1/SNOWFLAKE_SAMPLE_DATA/TPCH_SF1?warehouse=DEMO_WH",
    )
    flineitem = lineitem[lineitem.l_shipdate > date]
    forders = orders[orders.o_orderdate < date]
    fcustomer = customer[customer.c_mktsegment == "household"]
    jn1 = fcustomer.merge(forders, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(flineitem, left_on="o_orderkey", right_on="l_orderkey")
    jn2["tmp"] = jn2.l_extendedprice * (1 - jn2.l_discount)
    col_list = ["l_orderkey", "o_orderdate", "o_shippriority"]
    total = jn2.groupby(col_list, as_index=False)["tmp"].sum()
    total = total.sort_values(["tmp"], ascending=False)
    res = total[["l_orderkey", "tmp", "o_orderdate", "o_shippriority"]]
    print(res.head(10))


if __name__ == "__main__":
    q3()
