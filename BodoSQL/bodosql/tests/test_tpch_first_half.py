"""
Test correctness of TPCH Benchmark on BodoSQL

Some of these queries should be set with variables. These variables and their values can
be seen in the TPC-H document,
http://tpc.org/tpc_documents_current_versions/pdf/tpc-h_v2.18.0.pdf. For now we set most
of these variables according to the reference query.
"""

import datetime
import io

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_tpch_q1(tpch_data, memory_leak_check):
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
    check_query(
        tpch_query,
        tpch_data,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=pd.DataFrame(
            {
                "L_RETURNFLAG": ["A", "N", "N", "R"],
                "L_LINESTATUS": ["F", "F", "O", "F"],
                "SUM_QTY": [754903.0, 18528.0, 1547191.0, 756206.0],
                "SUM_BASE_PRICE": [
                    1057903302.66,
                    25562558.470000003,
                    2168783476.210001,
                    1059849272.5500004,
                ],
                "SUM_DISC_PRICE": [
                    1004953720.6289998,
                    24328272.68109999,
                    2060497575.354499,
                    1006781563.809,
                ],
                "SUM_CHARGE": [
                    1045349326.7462609,
                    25293278.719270002,
                    2143008061.0654063,
                    1047214071.436704,
                ],
                "AVG_QTY": [
                    25.559607245640766,
                    26.16949152542373,
                    25.48872341477076,
                    25.572554191606642,
                ],
                "AVG_PRICE": [
                    35818.6322214322,
                    36105.30857344633,
                    35728.95794484442,
                    35840.83299685504,
                ],
                "AVG_DISC": [
                    0.05011647198239391,
                    0.048644067796610166,
                    0.049992916096934154,
                    0.050270197152615874,
                ],
                "COUNT_ORDER": [29535, 708, 60701, 29571],
            }
        ),
    )


# test_tpch_q2 has timed out at 350 seconds with
# multiple processes
@pytest.mark.timeout(900)
@pytest.mark.slow
def test_tpch_q2(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "S_ACCTBAL": [8561.72, 8271.39, 4186.95, 2972.26, 1687.81, 1596.44],
            "S_NAME": [
                "Supplier#000000151",
                "Supplier#000000146",
                "Supplier#000000077",
                "Supplier#000000016",
                "Supplier#000000017",
                "Supplier#000000158",
            ],
            "N_NAME": ["RUSSIA", "RUSSIA", "GERMANY", "RUSSIA", "ROMANIA", "GERMANY"],
            "P_PARTKEY": [1634, 3080, 323, 1015, 2156, 2037],
            "P_MFGR": [
                "Manufacturer#2",
                "Manufacturer#2",
                "Manufacturer#4",
                "Manufacturer#4",
                "Manufacturer#5",
                "Manufacturer#1",
            ],
            "S_ADDRESS": [
                "2hd,3OAKPb39IY7 XuptY",
                "rBDNgCr04x0sfdzD5,gFOutCiG2",
                "wVtcr0uH3CyrSiWMLsqnB09Syo,UuZxPMeBghlY",
                "YjP5C55zHDXL7LalK27zfQnwejdpin4AMpvh",
                "c2d,ESHRSkK3WYnxpgw6aOqN0q",
                " fkjbx7,DYi",
            ],
            "S_PHONE": [
                "32-960-568-5148",
                "32-792-619-3155",
                "17-281-345-4863",
                "32-822-502-4215",
                "29-601-884-9219",
                "17-873-902-6175",
            ],
            "S_COMMENT": [
                "hely final packages. ironic pinto beans haggle qu",
                "s cajole quickly special requests. quickly enticing theodolites h",
                "the slyly final asymptotes. blithely pending theodoli",
                "ously express ideas haggle quickly dugouts? fu",
                "eep against the furiously bold ideas. fluffily bold packa",
                "cuses sleep after the pending, final ",
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
def test_tpch_q3(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "L_ORDERKEY": [
                22276,
                93382,
                29095,
                47525,
                23270,
                94432,
                74016,
                86566,
                20641,
                109188,
            ],
            "REVENUE": [
                275673.9548,
                264970.7683,
                248090.49339999998,
                238091.1183,
                235409.15840000001,
                232498.77,
                215325.5636,
                210468.1248,
                206330.73539999998,
                200755.77349999998,
            ],
            "O_ORDERDATE": [
                datetime.date(1995, 1, 29),
                datetime.date(1995, 1, 21),
                datetime.date(1995, 3, 9),
                datetime.date(1995, 3, 4),
                datetime.date(1995, 3, 14),
                datetime.date(1995, 1, 19),
                datetime.date(1995, 3, 11),
                datetime.date(1995, 3, 9),
                datetime.date(1995, 2, 20),
                datetime.date(1995, 3, 14),
            ],
            "O_SHIPPRIORITY": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
def test_tpch_q3_logging_info(tpch_data, memory_leak_check):
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
    # Adding an secondary test to tpch3 to
    # double check logging information is correct for the
    # rel node timing information,
    # since this TPCH uses most of the rel nodes we would need to support
    # (Join, Filter, Project, Groupby/Order by)

    bc = bodosql.BodoSQLContext(tpch_data)

    def impl(bc):
        return bc.sql(tpch_query)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl)(bc)
        plan = bc.generate_plan(tpch_query)
        # Check the columns were pruned
        for relNodeStr in plan.split("\n"):
            relNodeStr = relNodeStr.strip()
            if not (
                relNodeStr.startswith("PandasTableScan")
                or relNodeStr.startswith("CombineStreamsExchange")
                or relNodeStr.startswith("SeparateStreamExchange")
                or relNodeStr.startswith("PandasToBodoPhysicalConverter")
            ):
                check_logger_msg(stream, relNodeStr, check_case=False)


@pytest.mark.slow
def test_tpch_q4(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "O_ORDERPRIORITY": [
                "1-URGENT",
                "2-HIGH",
                "3-MEDIUM",
                "4-NOT SPECIFIED",
                "5-LOW",
            ],
            "ORDER_COUNT": [172, 216, 204, 202, 223],
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
@pytest.mark.bodosql_cpp
def test_tpch_q5(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "N_NAME": ["JAPAN", "INDONESIA", "INDIA", "VIETNAM", "CHINA"],
            "REVENUE": [
                1646669.3695,
                1448212.9797999999,
                1096228.0677999996,
                507624.05040000007,
                456508.70560000004,
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
def test_tpch_q6(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame({"REVENUE": [2317732.5497]})
    check_query(
        tpch_query,
        tpch_data,
        None,
        is_out_distributed=False,
        check_dtype=False,
        sort_output=False,
        expected_output=py_output,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q7(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "SUPP_NATION": ["FRANCE", "FRANCE", "GERMANY", "GERMANY"],
            "CUST_NATION": ["GERMANY", "GERMANY", "FRANCE", "FRANCE"],
            "L_YEAR": [1995, 1996, 1995, 1996],
            "REVENUE": [
                823153.0747000001,
                1322563.2609999997,
                946232.9485000003,
                1289011.5261,
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


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_tpch_q8(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "O_YEAR": [1995, 1996],
            "MKT_SHARE": [0.129271, 0.066993],
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
def test_tpch_q9(tpch_data, spark_info, memory_leak_check):
    COLOR = "green"
    tpch_query = f"""select
                       NATION,
                       O_YEAR,
                       sum(amount) as SUM_PROFIT
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
    # Note: There are 175 rows so its hard to hardcode this answer without a file.
    check_query(
        tpch_query,
        tpch_data,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.timeout(700)
@pytest.mark.slow
def test_tpch_q10(tpch_data, memory_leak_check):
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
    py_output = pd.DataFrame(
        {
            "C_CUSTKEY": [
                394,
                529,
                2236,
                1744,
                955,
                871,
                844,
                2321,
                226,
                1297,
                1706,
                2956,
                1474,
                2060,
                1984,
                2143,
                1543,
                349,
                1999,
                421,
            ],
            "C_NAME": [
                "Customer#000000394",
                "Customer#000000529",
                "Customer#000002236",
                "Customer#000001744",
                "Customer#000000955",
                "Customer#000000871",
                "Customer#000000844",
                "Customer#000002321",
                "Customer#000000226",
                "Customer#000001297",
                "Customer#000001706",
                "Customer#000002956",
                "Customer#000001474",
                "Customer#000002060",
                "Customer#000001984",
                "Customer#000002143",
                "Customer#000001543",
                "Customer#000000349",
                "Customer#000001999",
                "Customer#000000421",
            ],
            "REVENUE": [
                481779.4893,
                459834.41670000006,
                431506.03510000004,
                425876.6368,
                417901.13420000003,
                414347.1318,
                381894.047,
                375562.0412999999,
                372249.6906,
                371166.3517,
                370668.8651,
                367425.16359999997,
                345538.07719999994,
                344127.32920000004,
                343200.4994,
                337188.2567,
                333879.1647,
                333827.6899,
                323406.6697,
                317531.0798,
            ],
            "C_ACCTBAL": [
                5200.96,
                9647.58,
                -968.87,
                1436.96,
                138.31,
                -395.89,
                2954.9,
                -721.89,
                9008.61,
                6074.01,
                455.15,
                7048.48,
                2961.79,
                3995.46,
                8661.08,
                5373.42,
                5653.73,
                -565.35,
                -117.85,
                7073.17,
            ],
            "N_NAME": [
                "UNITED KINGDOM",
                "MOROCCO",
                "ROMANIA",
                "PERU",
                "ALGERIA",
                "SAUDI ARABIA",
                "IRAQ",
                "KENYA",
                "CANADA",
                "VIETNAM",
                "BRAZIL",
                "MOROCCO",
                "MOZAMBIQUE",
                "MOROCCO",
                "JORDAN",
                "INDONESIA",
                "CHINA",
                "UNITED KINGDOM",
                "BRAZIL",
                "JORDAN",
            ],
            "C_ADDRESS": [
                "nxW1jt,MQvImdr z72gAt1bslnfEipCh,bKZN",
                "oGKgweC odpyORKPJ9oxTqzzdlYyFOwXm2F97C",
                "x8 7D8xSxqIGoVOlqVCEBflsLXwKewpGMv,V",
                "cUBf1 YMJEgbt2XDeQWD4WinTu4iFIF",
                "FIis0dJhR5DwVCLy",
                "KcLmBKitbx7NvU7bpu9clIyccxWG",
                "1nUzjsH9HS1sPAGLwDIom9IESivLeEh1BvyynjU",
                "2nRj1CdoD3W0sJww2OZ21xh",
                "ToEmqB90fM TkLqyEgX8MJ8T8NkK",
                "4QnYEe0KXOP3yridKldXROs7jQdMu9tE",
                "FBx04exFTAFFRA3G,UR9Q2XSM1c8Uopaal2rEFv",
                "cTaJAqxDF,SJqSk2l6dAY5VTilR",
                "KB83CaaM8DRjvAmEMg1fw",
                " kY6weQvh3EKcZCH6N4,WaV11Ma8js",
                "MAqwYLxOBbMoyAWwvjEZK9QYgRMbhtFkdHbiR",
                "iSIlVncg,sc sQBHnywt4",
                "IKgaPQRsONfY1vAsPP",
                "vjJBjxjW9uoRZP02nS p6XY5wU6Ic,6xHpxUKA",
                "y8mRnn6pJ0V",
                "it3mUlkZAe9J8gmy",
            ],
            "C_PHONE": [
                "33-422-600-6936",
                "25-383-240-7326",
                "29-962-402-1321",
                "27-864-312-2867",
                "10-918-863-8880",
                "30-933-714-8982",
                "21-285-410-4046",
                "24-470-393-1146",
                "13-452-318-7709",
                "31-579-682-9907",
                "12-442-364-1024",
                "25-286-818-7068",
                "26-609-226-4269",
                "25-635-470-2363",
                "23-768-636-1831",
                "19-861-895-7214",
                "28-327-662-8527",
                "33-818-229-3473",
                "12-967-439-5391",
                "23-918-228-2560",
            ],
            "C_COMMENT": [
                " instructions. carefully special ideas after the fluffily unusual r",
                " deposits after the fluffily special foxes integrate carefully blithely dogged dolphins. enticingly bold d",
                "totes. quickly even instructions boost blithely regular packages? carefully final foxes haggle slyly against the s",
                "egularly bold, ironic packages. even theodolites nag. unusual theodolites would sleep furiously express",
                "ts cajole quickly according to the pending, unusual dolphins. special, ironic c",
                "ts. blithely silent courts doze. regular atta",
                "ymptotes. ironic, unusual notornis wake after the ironic, special deposits. blithely fina",
                "ow furiously fluffily daring deposits. regular, regular accounts haggle blithely acro",
                "ic packages. ideas cajole furiously slyly special theodolites: carefully express pinto beans acco",
                " pinto beans! furiously regular courts ea",
                " beans after the ironically pending accounts affix furiously quick pla",
                "s about the theodolites sleep furiously quickly regular instructions. f",
                "kages above the requests sleep furiously packages-- deposits detect fluffily. pending th",
                "ely. ironic, ironic asymptotes alongside of the even packages are furiously regular asymptotes; regular e",
                "y unusual requests. furiously ironic deposits haggle quickly a",
                "ver the unusual accounts. sile",
                "ckages haggle. idly even deposits according to the regularly even ideas haggle blithely re",
                "y. bold, ironic instructions after the theodolites sleep blithely ironic packages. ideas c",
                "heodolites. furiously ironic excuses boost along the carefully f",
                "lithely final deposits haggle furiously above the",
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
def test_tpch_q11(tpch_data, spark_info, memory_leak_check):
    NATION = "GERMANY"
    FRACTION = 0.0001
    tpch_query = f"""
                  select
                    PS_PARTKEY,
                    sum(ps_supplycost * ps_availqty) as VAL
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
    # Note: There are > 700 rows so its hard to hardcode this answer without a file.
    check_query(
        tpch_query,
        tpch_data,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )
