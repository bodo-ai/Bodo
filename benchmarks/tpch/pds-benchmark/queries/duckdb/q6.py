from queries.duckdb import utils

Q_NUM = 6


def q() -> None:
    line_item_ds = utils.get_line_item_ds()

    query_str = f"""
    select
        sum(l_extendedprice * l_discount) as revenue
    from
        {line_item_ds}
    where
        l_shipdate >= timestamp '1994-01-01'
        and l_shipdate < timestamp '1994-01-01' + interval '1' year
        and l_discount between .06 - 0.01 and .06 + 0.01
        and l_quantity < 24
    """

    utils.run_query(Q_NUM, query_str)


if __name__ == "__main__":
    q()
