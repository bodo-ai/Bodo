from queries.pyspark import utils

Q_NUM = 6


def q() -> None:
    query_str = """
    select
        sum(l_extendedprice * l_discount) as revenue
    from
        lineitem
    where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between .06 - 0.01 and .06 + 0.01
        and l_quantity < 24
    """

    utils.get_line_item_ds()

    q_final = utils.get_or_create_spark().sql(query_str)

    utils.run_query(Q_NUM, q_final)


if __name__ == "__main__":
    q()
