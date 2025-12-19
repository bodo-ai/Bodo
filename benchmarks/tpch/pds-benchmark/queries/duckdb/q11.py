from queries.duckdb import utils
from settings import Settings

settings = Settings()

Q_NUM = 11


def q() -> None:
    supplier_ds = utils.get_supplier_ds()
    part_supp_ds = utils.get_part_supp_ds()
    nation_ds = utils.get_nation_ds()
    scale_factor = settings.scale_factor
    fraction = 0.0001 / scale_factor

    query_str = f"""
    select
        ps_partkey,
        round(sum(ps_supplycost * ps_availqty), 2) as value
    from
        {part_supp_ds},
        {supplier_ds},
        {nation_ds}
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
                {part_supp_ds},
                {supplier_ds},
                {nation_ds}
            where
                ps_suppkey = s_suppkey
                and s_nationkey = n_nationkey
                and n_name = 'GERMANY'
            )
        order by
            value desc
	"""

    utils.run_query(Q_NUM, query_str)


if __name__ == "__main__":
    q()
