from queries.duckdb import utils

Q_NUM = 15


def q() -> None:
    line_item_ds = utils.get_line_item_ds()
    supplier_ds = utils.get_supplier_ds()

    query_str = f"""
    with revenue (supplier_no, total_revenue) as (
        select
            l_suppkey,
            sum(l_extendedprice * (1 - l_discount))
        from
            {line_item_ds}
        where
            l_shipdate >= date '1996-01-01'
            and l_shipdate < date '1996-01-01' + interval '3' month
        group by
            l_suppkey
    )
    select
        s_suppkey,
        s_name,
        s_address,
        s_phone,
        total_revenue
    from
        {supplier_ds},
        revenue
    where
        s_suppkey = supplier_no
        and total_revenue = (
            select
                max(total_revenue)
            from
                revenue
        )
    order by
        s_suppkey
	"""

    utils.run_query(Q_NUM, query_str)


if __name__ == "__main__":
    q()
