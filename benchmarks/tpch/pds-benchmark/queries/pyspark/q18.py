from queries.pyspark import utils

Q_NUM = 18


def q() -> None:
    query_str = """
    select
        c_name,
        c_custkey,
        o_orderkey,
        to_date(o_orderdate) as o_orderdat,
        o_totalprice,
        DOUBLE(sum(l_quantity)) as col6
    from
        customer,
        orders,
        lineitem
    where
        o_orderkey in (
            select
                l_orderkey
            from
                lineitem
            group by
                l_orderkey having
                    sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
    group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
    order by
        o_totalprice desc,
        o_orderdate
    limit 100
	"""

    utils.get_line_item_ds()
    utils.get_customer_ds()
    utils.get_orders_ds()

    q_final = utils.get_or_create_spark().sql(query_str)

    utils.run_query(Q_NUM, q_final)


if __name__ == "__main__":
    q()
