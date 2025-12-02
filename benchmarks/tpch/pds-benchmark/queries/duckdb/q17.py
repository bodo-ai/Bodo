from queries.duckdb import utils

Q_NUM = 17


def q() -> None:
    part_ds = utils.get_part_ds()
    line_item_ds = utils.get_line_item_ds()

    query_str = f"""
    select
        round(sum(l_extendedprice) / 7.0, 2) as avg_yearly
    from
        {line_item_ds},
        {part_ds}
    where
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container = 'MED BOX'
        and l_quantity < (
            select
                0.2 * avg(l_quantity)
            from
                {line_item_ds}
            where
                l_partkey = p_partkey
        )
	"""

    utils.run_query(Q_NUM, query_str)


if __name__ == "__main__":
    q()
