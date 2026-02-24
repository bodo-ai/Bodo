from queries.duckdb import utils

Q_NUM = 2


def q() -> None:
    region_ds = utils.get_region_ds()
    nation_ds = utils.get_nation_ds()
    supplier_ds = utils.get_supplier_ds()
    part_ds = utils.get_part_ds()
    part_supp_ds = utils.get_part_supp_ds()

    query_str = f"""
    select
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
    from
        {part_ds},
        {supplier_ds},
        {part_supp_ds},
        {nation_ds},
        {region_ds}
    where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and p_size = 15
        and p_type like '%BRASS'
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'EUROPE'
        and ps_supplycost = (
            select
                min(ps_supplycost)
            from
                {part_supp_ds},
                {supplier_ds},
                {nation_ds},
                {region_ds}
            where
                p_partkey = ps_partkey
                and s_suppkey = ps_suppkey
                and s_nationkey = n_nationkey
                and n_regionkey = r_regionkey
                and r_name = 'EUROPE'
        )
    order by
        s_acctbal desc,
        n_name,
        s_name,
        p_partkey
    limit 100
    """

    utils.run_query(Q_NUM, query_str)


if __name__ == "__main__":
    q()
