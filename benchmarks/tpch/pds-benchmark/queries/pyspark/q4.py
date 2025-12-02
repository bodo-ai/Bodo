from queries.pyspark import utils

Q_NUM = 4


def q() -> None:
    query_str = """
    select
        o_orderpriority,
        count(*) as order_count
    from
        orders
    where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-07-01' + interval '3' month
        and exists (
            select
                *
            from
                lineitem
            where
                l_orderkey = o_orderkey
                and l_commitdate < l_receiptdate
        )
    group by
        o_orderpriority
    order by
        o_orderpriority
    """

    utils.get_orders_ds()
    utils.get_line_item_ds()

    q_final = utils.get_or_create_spark().sql(query_str)

    utils.run_query(Q_NUM, q_final)


if __name__ == "__main__":
    q()
