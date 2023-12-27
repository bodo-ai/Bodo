# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of TPCH Benchmark on BodoSQL

Some of these queries should be set with variables. These variables and their values can
be seen in the TPC-H document,
http://tpc.org/tpc_documents_current_versions/pdf/tpc-h_v2.18.0.pdf. For now we set most
of these variables according to the reference query.
"""

import datetime

import pandas as pd
import pytest

from bodosql.tests.utils import check_query, shrink_data


@pytest.mark.slow
def test_tpch_q12(tpch_data, memory_leak_check):
    SHIPMODE1 = "MAIL"
    SHIPMODE2 = "SHIP"
    DATE = "1994-01-01"
    tpch_query = f"""select
                       l_shipmode,
                       sum(case
                         when o_orderpriority ='1-URGENT'
                           or o_orderpriority ='2-HIGH'
                         then 1
                         else 0
                       end) as high_line_count,
                       sum(case
                         when o_orderpriority <> '1-URGENT'
                           and o_orderpriority <> '2-HIGH'
                         then 1
                         else 0
                       end) as low_line_count
                     from
                       orders,
                       lineitem
                     where
                       o_orderkey = l_orderkey
                       and l_shipmode in ('{SHIPMODE1}', '{SHIPMODE2}')
                       and l_commitdate < l_receiptdate
                       and l_shipdate < l_commitdate
                       and l_receiptdate >= date '{DATE}'
                       and l_receiptdate < date '{DATE}' + interval '1' year
                     group by
                       l_shipmode
                     order by
                       l_shipmode
    """
    py_output = pd.DataFrame(
        {
            "L_SHIPMODE": ["MAIL", "SHIP"],
            "HIGH_LINE_COUNT": [124, 146],
            "LOW_LINE_COUNT": [174, 193],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_tpch_q13(tpch_data, memory_leak_check):
    WORD1 = "special"
    WORD2 = "requests"
    tpch_query = f"""select
                       c_count, count(*) as custdist
                     from (
                       select
                         c_custkey,
                         count(o_orderkey)
                       from
                         customer left outer join orders on
                           c_custkey = o_custkey
                           and o_comment not like '%{WORD1}%{WORD2}%'
                       group by
                         c_custkey
                       )as c_orders (c_custkey, c_count)
                     group by
                       c_count
                     order by
                       custdist desc,
                       c_count desc
    """
    py_output = pd.DataFrame(
        {
            "C_COUNT": [
                0,
                10,
                9,
                11,
                18,
                12,
                14,
                15,
                19,
                8,
                17,
                7,
                13,
                20,
                23,
                22,
                16,
                6,
                21,
                24,
                5,
                25,
                26,
                27,
                4,
                28,
                3,
                30,
                29,
                2,
                32,
                31,
                34,
                33,
                38,
                36,
                35,
            ],
            "CUSTDIST": [
                1000,
                151,
                122,
                119,
                106,
                101,
                100,
                99,
                98,
                98,
                95,
                92,
                89,
                84,
                83,
                81,
                80,
                73,
                70,
                58,
                44,
                39,
                25,
                23,
                19,
                10,
                9,
                7,
                6,
                5,
                4,
                3,
                2,
                2,
                1,
                1,
                1,
            ],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_tpch_q14(tpch_data, memory_leak_check):
    DATE = "1995-09-01"
    tpch_query = f"""select
                      100.00 * sum(case
                        when p_type like 'PROMO%'
                        then l_extendedprice*(1-l_discount)
                        else 0
                      end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
                    from
                      lineitem,
                      part
                    where
                      l_partkey = p_partkey
                      and l_shipdate >= date '{DATE}'
                      and l_shipdate < date '{DATE}' +  interval '1' month
    """
    py_output = pd.DataFrame({"PROMO_REVENUE": [15.814852]})
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        is_out_distributed=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_tpch_q15_blazingsql(tpch_data, memory_leak_check):
    DATE = "1996-01-01"
    # This query is modified because we don't support DDL properly.
    # The changes match the blazingsql test suite.
    # TODO: Match Q15 exactly with DDL
    tpch_query = f"""
                    with revenue (supplier_no, total_revenue) as (
                      select
                        l_suppkey,
                        sum(l_extendedprice * (1 - l_discount))
                      from
                        lineitem
                      where
                        l_shipdate >= date '{DATE}'
                        and l_shipdate < date '{DATE}' + interval '3' month
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
    py_output = pd.DataFrame(
        {
            "S_SUPPKEY": [49],
            "S_NAME": ["Supplier#000000049"],
            "S_ADDRESS": ["Nvq 6macF4GtJvz"],
            "S_PHONE": ["34-211-567-6800"],
            "TOTAL_REVENUE": [1244477.1629],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


# test_tpch_q16 has timed out at 350 seconds with
# multiple processes
@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q16(tpch_data, spark_info, memory_leak_check):
    BRAND = "BRAND#45"
    TYPE = "MEDIUM POLISHED"
    SIZE1 = 49
    SIZE2 = 14
    SIZE3 = 23
    SIZE4 = 45
    SIZE5 = 19
    SIZE6 = 3
    SIZE7 = 36
    SIZE8 = 9
    tpch_query = f"""select
                      P_BRAND,
                       P_TYPE,
                       P_SIZE,
                       count(distinct ps_suppkey) as SUPPLIER_CNT
                     from
                       partsupp,
                       part
                     where
                       p_partkey = ps_partkey
                       and p_brand <> '{BRAND}'
                       and p_type not like '{TYPE}%'
                       and p_size in ({SIZE1}, {SIZE2}, {SIZE3}, {SIZE4}, {SIZE5}, {SIZE6}, {SIZE7}, {SIZE8})
                       and ps_suppkey not in (
                         select
                           s_suppkey
                         from
                           supplier
                         where
                           s_comment like '%Customer%Complaints%'
                       )
                     group by
                       p_brand,
                       p_type,
                       p_size
                     order by
                       supplier_cnt desc,
                       p_brand,
                       p_type,
                       p_size
    """
    # We seem to be encountering memory errors on nightly, shrinking the input DataFrames
    # to see if this fixes the issue.
    tpch_data = shrink_data(tpch_data, 1000)
    check_query(
        tpch_query,
        tpch_data,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q17(tpch_data, memory_leak_check):
    BRAND = "Brand#23"
    CONTAINER = "MED BOX"
    tpch_query = f"""select
                       sum(l_extendedprice) / 7.0 as avg_yearly
                     from
                       lineitem,
                       part
                     where
                       p_partkey = l_partkey
                       and p_brand = '{BRAND}'
                       and p_container = '{CONTAINER}'
                       and l_quantity < (
                         select
                           0.2 * avg(l_quantity)
                         from
                           lineitem
                         where
                           l_partkey = p_partkey
                       )
    """
    py_output = pd.DataFrame({"AVG_YEARLY": [3008.928571]})
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        is_out_distributed=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_tpch_q18(tpch_data, memory_leak_check):
    QUANTITY = 300
    tpch_query = f"""select
                       c_name,
                       c_custkey,
                       o_orderkey,
                       o_orderdate,
                       o_totalprice,
                       sum(L_QUANTITY)
                     from
                       customer,
                       orders,
                       lineitem
                     where
                       o_orderkey in (
                         select
                           l_orderkey
                         from
                           lineitem
                         group by
                           l_orderkey having
                             sum(l_quantity) > {QUANTITY}
                       )
                       and c_custkey = o_custkey
                       and o_orderkey = l_orderkey
                     group by
                       c_name,
                       c_custkey,
                       o_orderkey,
                       o_orderdate,
                       o_totalprice
                     order by
                       o_totalprice desc,
                       o_orderdate
    """
    py_output = pd.DataFrame(
        {
            "C_NAME": ["Customer#000000355", "Customer#000001331"],
            "C_CUSTKEY": [355, 1331],
            "O_ORDERKEY": [6882, 29158],
            "O_ORDERDATE": [datetime.date(1997, 4, 9), datetime.date(1995, 10, 21)],
            "O_TOTALPRICE": [451578.1, 387687.84],
            "SUM(L_QUANTITY)": [303.0, 305.0],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q19(tpch_data, memory_leak_check):
    QUANTITY1 = 1
    QUANTITY2 = 10
    QUANTITY3 = 20
    BRAND1 = "Brand#12"
    BRAND2 = "Brand#23"
    BRAND3 = "Brand#34"
    tpch_query = f"""select
                       sum(l_extendedprice * (1 - l_discount) ) as revenue
                     from
                       lineitem,
                       part
                     where
                       (
                         p_partkey = l_partkey
                         and p_brand = '{BRAND1}'
                         and p_container in ( 'SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                         and l_quantity >= {QUANTITY1} and l_quantity <= {QUANTITY1} + 10
                         and p_size between 1 and 5
                         and l_shipmode in ('AIR', 'AIR REG')
                         and l_shipinstruct = 'DELIVER IN PERSON'
                       )
                       or
                       (
                         p_partkey = l_partkey
                         and p_brand = '{BRAND2}'
                         and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                         and l_quantity >= {QUANTITY2} and l_quantity <= {QUANTITY2} + 10
                         and p_size between 1 and 10
                         and l_shipmode in ('AIR', 'AIR REG')
                         and l_shipinstruct = 'DELIVER IN PERSON'
                       )
                       or
                         (
                         p_partkey = l_partkey
                         and p_brand = '{BRAND3}'
                         and p_container in ( 'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                         and l_quantity >= {QUANTITY3} and l_quantity <= {QUANTITY3} + 10
                         and p_size between 1 and 15
                         and l_shipmode in ('AIR', 'AIR REG')
                         and l_shipinstruct = 'DELIVER IN PERSON'
                       )
    """
    py_output = pd.DataFrame({"REVENUE": [81238.2247]})
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        is_out_distributed=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
# NOTE (allai5): Arbitrary high timeout number due to inability to replicate
# timeout locally
def test_tpch_q20(tpch_data, memory_leak_check):
    COLOR = "forest"
    DATE = "1994-01-01"
    NATION = "CANADA"
    tpch_query = f"""select
                       s_name,
                       s_address
                     from
                       supplier, nation
                     where
                       s_suppkey in (
                         select
                           ps_suppkey
                         from
                           partsupp
                         where
                           ps_partkey in (
                             select
                               p_partkey
                             from
                               part
                             where
                               p_name like '{COLOR}%'
                           )
                         and ps_availqty > (
                           select
                             0.5 * sum(l_quantity)
                           from
                             lineitem
                           where
                             l_partkey = ps_partkey
                             and l_suppkey = ps_suppkey
                             and l_shipdate >= date '{DATE}'
                             and l_shipdate < date '{DATE}' + interval '1' year
                         )
                       )
                       and s_nationkey = n_nationkey
                       and n_name = '{NATION}'
                     order by
                       s_name
    """
    py_output = pd.DataFrame(
        {
            "S_NAME": ["Supplier#000000091"],
            "S_ADDRESS": ["YV45D7TkfdQanOOZ7q9QxkyGUapU1oOWU6q3"],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.timeout(900)
@pytest.mark.slow
def test_tpch_q21(tpch_data, memory_leak_check):
    NATION = "SAUDI ARABIA"
    tpch_query = f"""select
                       s_name,
                       count(*) as numwait
                     from
                       supplier,
                       lineitem l1,
                       orders,
                       nation
                     where
                       s_suppkey = l1.l_suppkey
                       and o_orderkey = l1.l_orderkey
                       and o_orderstatus = 'F'
                       and l1.l_receiptdate > l1.l_commitdate
                       and exists (
                         select
                           *
                         from
                           lineitem l2
                         where
                           l2.l_orderkey = l1.l_orderkey
                           and l2.l_suppkey <> l1.l_suppkey
                       )
                       and not exists (
                         select
                           *
                         from
                           lineitem l3
                         where
                           l3.l_orderkey = l1.l_orderkey
                           and l3.l_suppkey <> l1.l_suppkey
                           and l3.l_receiptdate > l3.l_commitdate
                       )
                       and s_nationkey = n_nationkey
                       and n_name = '{NATION}'
                     group by
                       s_name
                     order by
                       numwait desc,
                       s_name
    """
    py_output = pd.DataFrame(
        {
            "S_NAME": [
                "Supplier#000000114",
                "Supplier#000000167",
                "Supplier#000000144",
                "Supplier#000000188",
                "Supplier#000000074",
            ],
            "NUMWAIT": [14, 10, 9, 9, 7],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q22(tpch_data, memory_leak_check):
    I1 = 13
    I2 = 31
    I3 = 23
    I4 = 29
    I5 = 30
    I6 = 18
    I7 = 17
    tpch_query = f"""select
                      cntrycode,
                      count(*) as numcust,
                      sum(c_acctbal) as totacctbal
                    from (
                      select
                        substring(c_phone from 1 for 2) as cntrycode,
                        c_acctbal
                      from
                        customer
                      where
                        substring(c_phone from 1 for 2) in
                          ('{I1}','{I2}','{I3}','{I4}','{I5}','{I6}','{I7}')
                        and c_acctbal > (
                          select
                            avg(c_acctbal)
                          from
                            customer
                          where
                            c_acctbal > 0.00
                            and substring (c_phone from 1 for 2) in
                              ('{I1}','{I2}','{I3}','{I4}','{I5}','{I6}','{I7}')
                        )
                        and not exists (
                          select
                            *
                          from
                            orders
                          where
                            o_custkey = c_custkey
                        )
                      ) as custsale
                    group by
                      cntrycode
                    order by
                      cntrycode
    """
    py_output = pd.DataFrame(
        {
            "CNTRYCODE": ["13", "17", "18", "23", "29", "30", "31"],
            "NUMCUST": [16, 21, 22, 12, 20, 24, 16],
            "TOTACCTBAL": [
                122554.49,
                153821.14000000004,
                171225.89,
                93112.65999999999,
                150627.41000000003,
                181736.16999999998,
                120672.20999999999,
            ],
        }
    )
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )
