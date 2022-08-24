# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of TPCH Benchmark on BodoSQL

Some of these queries should be set with variables. These variables and their values can
be seen in the TPC-H document,
http://tpc.org/tpc_documents_current_versions/pdf/tpc-h_v2.18.0.pdf. For now we set most
of these variables according to the reference query.
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_tpch_q1(tpch_data, spark_info, memory_leak_check):
    tpch_query = """select
                      l_returnflag,
                      l_linestatus,
                      sum(l_quantity) as sum_qty,
                      sum(l_extendedprice) as sum_base_price,
                      sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
                      sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
                      avg(l_quantity) as avg_qty,
                      avg(l_extendedprice) as avg_price,
                      avg(l_discount) as avg_disc,
                      count(*) as count_order
                    from
                      lineitem
                    where
                      l_shipdate <= date '1998-12-01'
                    group by
                      l_returnflag,
                      l_linestatus
                    order by
                      l_returnflag,
                      l_linestatus
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q2(tpch_data, spark_info, memory_leak_check):
    SIZE = 15
    TYPE = "BRASS"
    REGION = "EUROPE"
    tpch_query = f"""select
                       s_acctbal,
                       s_name,
                       n_name,
                       p_partkey,
                       p_mfgr,
                       s_address,
                       s_phone,
                       s_comment
                     from
                       part,
                       supplier,
                       partsupp,
                       nation,
                       region
                     where
                       p_partkey = ps_partkey
                       and s_suppkey = ps_suppkey
                       and p_size = {SIZE}
                       and p_type like '%{TYPE}'
                       and s_nationkey = n_nationkey
                       and n_regionkey = r_regionkey
                       and r_name = '{REGION}'
                       and ps_supplycost = (
                         select
                           min(ps_supplycost)
                         from
                           partsupp, supplier,
                           nation, region
                         where
                           p_partkey = ps_partkey
                           and s_suppkey = ps_suppkey
                           and s_nationkey = n_nationkey
                           and n_regionkey = r_regionkey
                           and r_name = '{REGION}'
                         )
                     order by
                       s_acctbal desc,
                       n_name,
                       s_name,
                       p_partkey
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q3(tpch_data, spark_info, memory_leak_check):
    tpch_query = """select
                      l_orderkey,
                      sum(l_extendedprice * (1 - l_discount)) as revenue,
                      o_orderdate,
                      o_shippriority
                    from
                      customer,
                      orders,
                      lineitem
                    where
                      c_mktsegment = 'BUILDING'
                      and c_custkey = o_custkey
                      and l_orderkey = o_orderkey
                      and o_orderdate < '1995-03-15'
                      and l_shipdate > '1995-03-15'
                    group by
                      l_orderkey,
                      o_orderdate,
                      o_shippriority
                    order by
                      revenue desc,
                      o_orderdate,
                      l_orderkey
                    limit 10
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q4(tpch_data, spark_info, memory_leak_check):
    DATE = "1993-07-01"
    tpch_query = f"""select
                       o_orderpriority,
                       count(*) as order_count
                     from
                       orders
                     where
                       o_orderdate >= date '{DATE}'
                       and o_orderdate < date '{DATE}' + interval '3' month
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
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q5(tpch_data, spark_info, memory_leak_check):
    tpch_query = """select
                      n_name,
                      sum(l_extendedprice * (1 - l_discount)) as revenue
                    from
                      customer,
                      orders,
                      lineitem,
                      supplier,
                      nation,
                      region
                    where
                      c_custkey = o_custkey
                      and l_orderkey = o_orderkey
                      and l_suppkey = s_suppkey
                      and c_nationkey = s_nationkey
                      and s_nationkey = n_nationkey
                      and n_regionkey = r_regionkey
                      and r_name = 'ASIA'
                      and o_orderdate >= '1994-01-01'
                      and o_orderdate < '1995-01-01'
                    group by
                      n_name
                    order by
                      revenue desc
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q6(tpch_data, spark_info, memory_leak_check):
    tpch_query = """select
                      sum(l_extendedprice * l_discount) as revenue
                    from
                      lineitem
                    where
                      l_shipdate >= '1994-01-01'
                      and l_shipdate < '1995-01-01'
                      and l_discount between 0.05 and 0.07
                      and l_quantity < 24
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        spark_dataframe_dict=spark_df_dict,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_tpch_q7(tpch_data, spark_info, memory_leak_check):
    NATION1 = "FRANCE"
    NATION2 = "GERMANY"
    tpch_query = f"""select
                       supp_nation,
                       cust_nation,
                       l_year, sum(volume) as revenue
                     from (
                       select
                         n1.n_name as supp_nation,
                         n2.n_name as cust_nation,
                         extract(year from l_shipdate) as l_year,
                         l_extendedprice * (1 - l_discount) as volume
                       from
                         supplier,
                         lineitem,
                         orders,
                         customer,
                         nation n1,
                         nation n2
                       where
                         s_suppkey = l_suppkey
                         and o_orderkey = l_orderkey
                         and c_custkey = o_custkey
                         and s_nationkey = n1.n_nationkey
                         and c_nationkey = n2.n_nationkey
                         and (
                           (n1.n_name = '{NATION1}' and n2.n_name = '{NATION2}')
                           or (n1.n_name = '{NATION2}' and n2.n_name = '{NATION1}')
                         )
                         and l_shipdate between date '1995-01-01' and date '1996-12-31'
                       ) as shipping
                     group by
                       supp_nation,
                       cust_nation,
                       l_year
                     order by
                       supp_nation,
                       cust_nation,
                       l_year
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q8(tpch_data, spark_info, memory_leak_check):
    NATION = "BRAZIL"
    REGION = "AMERICA"
    TYPE = "ECONOMY ANODIZED STEEL"
    tpch_query = f"""select
                       o_year,
                       sum(case
                         when nation = '{NATION}'
                         then volume
                         else 0
                       end) / sum(volume) as mkt_share
                     from (
                       select
                         extract(year from o_orderdate) as o_year,
                         l_extendedprice * (1-l_discount) as volume,
                         n2.n_name as nation
                       from
                         part,
                         supplier,
                         lineitem,
                         orders,
                         customer,
                         nation n1,
                         nation n2,
                         region
                       where
                         p_partkey = l_partkey
                         and s_suppkey = l_suppkey
                         and l_orderkey = o_orderkey
                         and o_custkey = c_custkey
                         and c_nationkey = n1.n_nationkey
                         and n1.n_regionkey = r_regionkey
                         and r_name = '{REGION}'
                         and s_nationkey = n2.n_nationkey
                         and o_orderdate between date '1995-01-01' and date '1996-12-31'
                         and p_type = '{TYPE}'
                       ) as all_nations
                     group by
                       o_year
                     order by
                       o_year
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q9(tpch_data, spark_info, memory_leak_check):
    COLOR = "green"
    tpch_query = f"""select
                       nation,
                       o_year,
                       sum(amount) as sum_profit
                     from (
                       select
                         n_name as nation,
                         extract(year from o_orderdate) as o_year,
                         l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                       from
                         part,
                         supplier,
                         lineitem,
                         partsupp,
                         orders,
                         nation
                       where
                         s_suppkey = l_suppkey
                         and ps_suppkey = l_suppkey
                         and ps_partkey = l_partkey
                         and p_partkey = l_partkey
                         and o_orderkey = l_orderkey
                         and s_nationkey = n_nationkey
                         and p_name like '%{COLOR}%'
                       ) as profit
                     group by
                       nation,
                       o_year
                     order by
                       nation,
                       o_year desc
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q10(tpch_data, spark_info, memory_leak_check):
    tpch_query = """select
                      c_custkey,
                      c_name,
                      sum(l_extendedprice * (1 - l_discount)) as revenue,
                      c_acctbal,
                      n_name,
                      c_address,
                      c_phone,
                      c_comment
                    from
                      customer,
                      orders,
                      lineitem,
                      nation
                    where
                      c_custkey = o_custkey
                      and l_orderkey = o_orderkey
                      and o_orderdate >= '1993-10-01'
                      and o_orderdate < '1994-01-01'
                      and l_returnflag = 'R'
                      and c_nationkey = n_nationkey
                    group by
                      c_custkey,
                      c_name,
                      c_acctbal,
                      c_phone,
                      n_name,
                      c_address,
                      c_comment
                    order by
                      revenue desc,
                      c_custkey
                    limit 20
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )


@pytest.mark.slow
def test_tpch_q11(tpch_data, spark_info, memory_leak_check):
    NATION = "GERMANY"
    FRACTION = 0.0001
    tpch_query = f"""
                  select
                    ps_partkey,
                    sum(ps_supplycost * ps_availqty) as val
                  from
                    partsupp,
                    supplier,
                    nation
                  where
                    ps_suppkey = s_suppkey
                    and s_nationkey = n_nationkey
                    and n_name = '{NATION}'
                  group by
                    ps_partkey having
                      sum(ps_supplycost * ps_availqty) > (
                        select
                          sum(ps_supplycost * ps_availqty) * {FRACTION}
                        from
                          partsupp,
                          supplier,
                          nation
                        where
                          ps_suppkey = s_suppkey
                          and s_nationkey = n_nationkey
                          and n_name = '{NATION}'
                      )
                  order by
                    val desc
    """
    bodosql_df_dict, spark_df_dict = tpch_data
    check_query(
        tpch_query,
        bodosql_df_dict,
        spark_info,
        check_dtype=False,
        spark_dataframe_dict=spark_df_dict,
    )
