from queries.pyspark import utils

Q_NUM = 3


def q() -> None:
    query_str = """
    select
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        date(o_orderdate),
        o_shippriority
    from
        customer,
        orders,
        lineitem
    where
        c_mktsegment = 'BUILDING'
        and c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15'
        and l_shipdate > date '1995-03-15'
    group by
        l_orderkey,
        o_orderdate,
        o_shippriority
    order by
        revenue desc,
        o_orderdate
    limit 10
    """

    utils.get_customer_ds()
    utils.get_orders_ds()
    utils.get_line_item_ds()

    q_final = utils.get_or_create_spark().sql(query_str)

    utils.run_query(Q_NUM, q_final)


if __name__ == "__main__":
    q()
