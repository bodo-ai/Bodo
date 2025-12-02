from queries.pyspark import utils

Q_NUM = 15


def q() -> None:
    spark = utils.get_or_create_spark()

    ddl = """
    create temp view revenue (supplier_no, total_revenue) as
        select
            l_suppkey,
            sum(l_extendedprice * (1 - l_discount))
        from
            lineitem
        where
            l_shipdate >= date '1996-01-01'
            and l_shipdate < date '1996-01-01' + interval '3' month
        group by
            l_suppkey
    """

    query_str = """
    select
        s_suppkey,
        s_name,
        s_address,
        s_phone,
        total_revenue
    from
        supplier,
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

    utils.get_line_item_ds()
    utils.get_supplier_ds()

    spark.sql(ddl)
    q_final = spark.sql(query_str)

    utils.run_query(Q_NUM, q_final)
    spark.sql("drop view revenue")


if __name__ == "__main__":
    q()
