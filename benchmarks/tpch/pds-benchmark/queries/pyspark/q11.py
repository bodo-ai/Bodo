from queries.pyspark import utils
from settings import Settings

settings = Settings()

Q_NUM = 11


def q() -> None:
    scale_factor = settings.scale_factor
    fraction = 0.0001 / scale_factor

    query_str = f"""
    select
        ps_partkey,
        round(sum(ps_supplycost * ps_availqty), 2) as value
    from
        partsupp,
        supplier,
        nation
    where
        ps_suppkey = s_suppkey
        and s_nationkey = n_nationkey
        and n_name = 'GERMANY'
    group by
        ps_partkey having
                sum(ps_supplycost * ps_availqty) > (
            select
                sum(ps_supplycost * ps_availqty) * {fraction}
            from
                partsupp,
                supplier,
                nation
            where
                ps_suppkey = s_suppkey
                and s_nationkey = n_nationkey
                and n_name = 'GERMANY'
            )
        order by
            value desc
	"""

    utils.get_supplier_ds()
    utils.get_part_supp_ds()
    utils.get_nation_ds()

    q_final = utils.get_or_create_spark().sql(query_str)

    utils.run_query(Q_NUM, q_final)


if __name__ == "__main__":
    q()
