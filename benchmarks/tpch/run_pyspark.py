"""
Example that runs all TPCH queries all together.

"""

import argparse
import time

from pyspark import StorageLevel
from pyspark.sql import SparkSession


def _register_tables(spark, root, names, ext=".parquet", skip=False):
    if skip:
        return
    for name in names:
        table_path = f"{root}/{name}{ext}/"
        df = spark.read.parquet(table_path)
        df.createOrReplaceTempView(name)


def run_queries(data_folder, spark, queries, scale_factor, exclude_io=False):
    # Load the data
    t1 = time.time()
    if exclude_io:
        load_lineitem(data_folder, spark)
        load_orders(data_folder, spark)
        load_customer(data_folder, spark)
        load_nation(data_folder, spark)
        load_region(data_folder, spark)
        load_supplier(data_folder, spark)
        load_part(data_folder, spark)
        load_partsupp(data_folder, spark)

    # Force the reads to execute here so we can time IO
    print("Reading time (s): ", time.time() - t1)
    # Run the Queries:
    for query in queries:
        globals()[f"q{query:02}"](
            spark, data_folder, scale_factor, exclude_io=exclude_io
        )

    # print("Total Query time (s): ", time.time() - t2)
    # print("Total time (s): ", time.time() - t1)
    print(f"Total query execution time (s): {time.time() - t1}")


def load_lineitem(data_folder, spark, ext=".parquet"):
    lineitem = spark.read.parquet(data_folder + f"/lineitem{ext}/")
    # Persist to get accurate read times
    lineitem.persist(StorageLevel.MEMORY_ONLY)
    print(lineitem.count())
    # Add for Spark SQL
    lineitem.createOrReplaceTempView("lineitem")


def load_part(data_folder, spark, ext=".parquet"):
    part = spark.read.parquet(data_folder + f"/part{ext}/")
    # Persist to get accurate read times
    part.persist(StorageLevel.MEMORY_ONLY)
    print(part.count())
    # Add for Spark SQL
    part.createOrReplaceTempView("part")


def load_orders(data_folder, spark, ext=".parquet"):
    orders = spark.read.parquet(data_folder + f"/orders{ext}/")
    # Persist to get accurate read times
    orders.persist(StorageLevel.MEMORY_ONLY)
    print(orders.count())
    # Add for Spark SQL
    orders.createOrReplaceTempView("orders")


def load_customer(data_folder, spark, ext=".parquet"):
    customer = spark.read.parquet(data_folder + f"/customer{ext}/")
    # Persist to get accurate read times
    customer.persist(StorageLevel.MEMORY_ONLY)
    print(customer.count())
    # Add for Spark SQL
    customer.createOrReplaceTempView("customer")


def load_nation(data_folder, spark, ext=".parquet"):
    nation = spark.read.parquet(data_folder + f"/nation{ext}/")
    # Persist to get accurate read times
    nation.persist(StorageLevel.MEMORY_ONLY)
    print(nation.count())
    # Add for Spark SQL
    nation.createOrReplaceTempView("nation")


def load_region(data_folder, spark, ext=".parquet"):
    region = spark.read.parquet(data_folder + f"/region{ext}/")
    # Persist to get accurate read times
    region.persist(StorageLevel.MEMORY_ONLY)
    print(region.count())
    # Add for Spark SQL
    region.createOrReplaceTempView("region")


def load_supplier(data_folder, spark, ext=".parquet"):
    supplier = spark.read.parquet(data_folder + f"/supplier{ext}/")
    # Persist to get accurate read times
    supplier.persist(StorageLevel.MEMORY_ONLY)
    print(supplier.count())
    # Add for Spark SQL
    supplier.createOrReplaceTempView("supplier")


def load_partsupp(data_folder, spark, ext=".parquet"):
    partsupp = spark.read.parquet(data_folder + f"/partsupp{ext}/")
    # Persist to get accurate read times
    partsupp.persist(StorageLevel.MEMORY_ONLY)
    print(partsupp.count())
    # Add for Spark SQL
    partsupp.createOrReplaceTempView("partsupp")


def q01(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem"], skip=exclude_io)

    sql_lineitem = spark.sql(
        """select
                l_returnflag,
                l_linestatus,
                sum(l_quantity) as sum_qty,
                sum(l_extendedprice) as sum_base_price,
                sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
                sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
                avg(l_quantity) as avg_qty,
                avg(l_extendedprice) as avg_price,
                avg(l_discount) as avg_disc,
                count(*) as count_order
            from
                lineitem
            where
                l_shipdate <= date '1998-12-01' - interval '90' day
            group by
                l_returnflag,
                l_linestatus
            order by
                l_returnflag,
                l_linestatus"""
    )

    sql_lineitem.show()
    print("Q01 Execution time (s): ", time.time() - t1)


def q02(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    SIZE = 15
    TYPE = "BRASS"
    REGION = "EUROPE"

    _register_tables(
        spark,
        data_folder,
        ["part", "supplier", "partsupp", "nation", "region"],
        skip=exclude_io,
    )

    total = spark.sql(
        f"""select
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
                p_partkey"""
    )

    total.show()
    print("Q02 Execution time (s): ", time.time() - t1)


def q03(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark, data_folder, ["customer", "orders", "lineitem"], skip=exclude_io
    )

    total = spark.sql(
        """select
            l_orderkey,
            sum(l_extendedprice * (1 - l_discount)) as revenue,
            o_orderdate,
            o_shippriority
        from
            customer,
            orders,
            lineitem
        where
            c_mktsegment = 'HOUSEHOLD'
            and c_custkey = o_custkey
            and l_orderkey = o_orderkey
            and o_orderdate < date '1995-03-04'
            and l_shipdate > date '1995-03-04'
        group by
            l_orderkey,
            o_orderdate,
            o_shippriority
        order by
            revenue desc,
            o_orderdate
        limit 10"""
    )

    total.show()
    print("Q03 Execution time (s): ", time.time() - t1)


def q04(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["orders", "lineitem"], skip=exclude_io)

    total = spark.sql(
        """select
                o_orderpriority,
                count(*) as order_count
            from
                orders
            where
                o_orderdate >= date '1993-08-01'
                and o_orderdate < date '1993-08-01' + interval '3' month
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
                o_orderpriority"""
    )

    total.show()
    print("Q04 Execution time (s): ", time.time() - t1)


def q05(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark,
        data_folder,
        ["customer", "orders", "lineitem", "supplier", "nation", "region"],
        skip=exclude_io,
    )

    total = spark.sql(
        """select
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
                and o_orderdate >= date '1996-01-01'
                and o_orderdate < date '1996-01-01' + interval '1' year
            group by
                n_name
            order by
                revenue desc"""
    )

    total.show()
    print("Q05 Execution time (s): ", time.time() - t1)


def q06(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem"], skip=exclude_io)

    sql_lineitem = spark.sql(
        """select
                sum(l_extendedprice * l_discount) as revenue
            from
                lineitem
            where
                l_shipdate >= date '1996-01-01'
                and l_shipdate < date '1996-01-01' + interval '1' year
                and l_discount between .08 and .1
                and l_quantity < 24"""
    )
    sql_lineitem.show()
    print("Q06 Execution time (s): ", time.time() - t1)


def q07(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    NATION1 = "FRANCE"
    NATION2 = "GERMANY"

    _register_tables(
        spark,
        data_folder,
        ["supplier", "lineitem", "orders", "customer", "nation"],
        skip=exclude_io,
    )

    total = spark.sql(
        f"""select
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
                l_year"""
    )
    total.show()
    print("Q07 Execution time (s): ", time.time() - t1)


def q08(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    NATION = "BRAZIL"
    REGION = "AMERICA"
    TYPE = "ECONOMY ANODIZED STEEL"

    _register_tables(
        spark,
        data_folder,
        ["part", "supplier", "lineitem", "orders", "customer", "nation", "region"],
        skip=exclude_io,
    )

    total = spark.sql(
        f"""select
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
                o_year"""
    )
    total.show()
    print("Q08 Execution time (s): ", time.time() - t1)


def q09(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark,
        data_folder,
        ["part", "supplier", "lineitem", "partsupp", "orders", "nation"],
        skip=exclude_io,
    )

    total = spark.sql(
        """select
                nation,
                o_year,
                sum(amount) as sum_profit
            from
                (
                    select
                        n_name as nation,
                        year(o_orderdate) as o_year,
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
                        and p_name like '%ghost%'
                ) as profit
            group by
                nation,
                o_year
            order by
                nation,
                o_year desc"""
    )
    total.show()
    print("Q09 Execution time (s): ", time.time() - t1)


def q10(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark,
        data_folder,
        ["customer", "orders", "lineitem", "nation"],
        skip=exclude_io,
    )

    total = spark.sql(
        """select
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
                and o_orderdate >= date '1994-11-01'
                and o_orderdate < date '1994-11-01' + interval '3' month
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
                revenue desc
            limit 20"""
    )

    total.show()
    print("Q10 Execution time (s): ", time.time() - t1)


def q11(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    NATION = "GERMANY"
    FRACTION = 0.0001 / scale

    _register_tables(
        spark, data_folder, ["partsupp", "supplier", "nation"], skip=exclude_io
    )

    total = spark.sql(
        f"""select
                ps_partkey,
                sum(ps_supplycost * ps_availqty) as value
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
                    value desc"""
    )

    total.show()
    print("Q11 Execution time (s): ", time.time() - t1)


def q12(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["orders", "lineitem"], skip=exclude_io)

    total = spark.sql(
        """select
                l_shipmode,
                sum(case
                    when o_orderpriority = '1-URGENT'
                        or o_orderpriority = '2-HIGH'
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
                and l_shipmode in ('MAIL', 'SHIP')
                and l_commitdate < l_receiptdate
                and l_shipdate < l_commitdate
                and l_receiptdate >= date '1994-01-01'
                and l_receiptdate < date '1994-01-01' + interval '1' year
            group by
                l_shipmode
            order by
                l_shipmode"""
    )

    total.show()
    print("Q12 Execution time (s): ", time.time() - t1)


def q13(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    WORD1 = "special"
    WORD2 = "requests"

    _register_tables(spark, data_folder, ["customer", "orders"], skip=exclude_io)

    total = spark.sql(
        f"""select
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
                c_count desc"""
    )
    total.show()
    print("Q13 Execution time (s): ", time.time() - t1)


def q14(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem", "part"], skip=exclude_io)

    total = spark.sql(
        """select
                100.00 * sum(case
                    when p_type like 'PROMO%'
                        then l_extendedprice * (1 - l_discount)
                    else 0
                end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
            from
                lineitem,
                part
            where
                l_partkey = p_partkey
                and l_shipdate >= date '1994-03-01'
                and l_shipdate < date '1994-03-01' + interval '1' month"""
    )

    total.show()
    print("Q14 Execution time (s): ", time.time() - t1)


def q15(spark, data_folder, scale, exclude_io=False):
    # From spec the l_extendedprice and l_discount are defined as decimals.
    # It defines decimal as having 12 digits and 2 digits after the point.
    # See p14 in
    # http://www.tpc.org/tpc_documents_current_versions/pdf/tpc-h_v2.17.1.pdf
    # That's why `CAST` is used here
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem", "supplier"], skip=exclude_io)

    spark.sql(
        """create temp view revenue (supplier_no, total_revenue) as
                select
                    l_suppkey,
                    CAST(sum(l_extendedprice * (1 - l_discount)) as DECIMAL(12,2))
                from
                    lineitem
                where
                    l_shipdate >= date '1996-01-01'
                    and l_shipdate < date '1996-01-01' + interval '3' month
                group by
                    l_suppkey"""
    )
    total = spark.sql(
        """
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
                s_suppkey"""
    )
    spark.sql("drop view revenue")
    total.show()
    print("Q15 Execution time (s): ", time.time() - t1)


def q16(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    BRAND = "Brand#45"
    TYPE = "MEDIUM POLISHED"
    SIZE1 = 49
    SIZE2 = 14
    SIZE3 = 23
    SIZE4 = 45
    SIZE5 = 19
    SIZE6 = 3
    SIZE7 = 36
    SIZE8 = 9

    _register_tables(
        spark, data_folder, ["partsupp", "part", "supplier"], skip=exclude_io
    )

    total = spark.sql(
        f"""select
                p_brand,
                p_type,
                p_size,
                count(distinct ps_suppkey) as supplier_cnt
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
                p_size"""
    )

    total.show()
    print("Q16 Execution time (s): ", time.time() - t1)


def q17(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem", "part"], skip=exclude_io)

    total = spark.sql(
        """select
                sum(l_extendedprice) / 7.0 as avg_yearly
            from
                lineitem,
                part
            where
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container = 'MED BOX'
                and l_quantity < (
                    select
                        0.2 * avg(l_quantity)
                    from
                        lineitem
                    where
                        l_partkey = p_partkey
                )"""
    )

    total.show()
    print("Q17 Execution time (s): ", time.time() - t1)


def q18(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark, data_folder, ["customer", "orders", "lineitem"], skip=exclude_io
    )

    total = spark.sql(
        """select
                c_name,
                c_custkey,
                o_orderkey,
                o_orderdate,
                o_totalprice,
                sum(l_quantity)
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
                            sum(l_quantity) > 300
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
            limit 100"""
    )

    total.show()
    print("Q18 Execution time (s): ", time.time() - t1)


def q19(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(spark, data_folder, ["lineitem", "part"], skip=exclude_io)

    total = spark.sql(
        """select
                sum(l_extendedprice* (1 - l_discount)) as revenue
            from
                lineitem,
                part
            where
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#31'
                    and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                    and l_quantity >= 4 and l_quantity <= 4 + 10
                    and p_size between 1 and 5
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )
                or
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#43'
                    and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                    and l_quantity >= 15 and l_quantity <= 25
                    and p_size between 1 and 10
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )
                or
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#43'
                    and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                    and l_quantity >= 26 and l_quantity <= 36
                    and p_size between 1 and 15
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )"""
    )

    total.show()
    print("Q19 Execution time (s): ", time.time() - t1)


def q20(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()

    _register_tables(
        spark,
        data_folder,
        ["supplier", "nation", "partsupp", "part", "lineitem"],
        skip=exclude_io,
    )

    total = spark.sql(
        """select
                s_name,
                s_address
            from
                supplier,
                nation
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
                                p_name like 'azure%'
                        )
                        and ps_availqty > (
                            select
                                0.5 * sum(l_quantity)
                            from
                                lineitem
                            where
                                l_partkey = ps_partkey
                                and l_suppkey = ps_suppkey
                                and l_shipdate >= date '1996-01-01'
                                and l_shipdate < date '1996-01-01' + interval '1' year
                        )
                )
                and s_nationkey = n_nationkey
                and n_name = 'JORDAN'
            order by
                s_name"""
    )

    total.show()
    print("Q20 Execution time (s): ", time.time() - t1)


def q21(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    NATION = "SAUDI ARABIA"

    _register_tables(
        spark,
        data_folder,
        ["supplier", "lineitem", "orders", "nation"],
        skip=exclude_io,
    )

    total = spark.sql(
        f"""select
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
                s_name"""
    )
    total.show()
    print("Q21 Execution time (s): ", time.time() - t1)


def q22(spark, data_folder, scale, exclude_io=False):
    t1 = time.time()
    I1 = 13
    I2 = 31
    I3 = 23
    I4 = 29
    I5 = 30
    I6 = 18
    I7 = 17

    _register_tables(spark, data_folder, ["customer", "orders"], skip=exclude_io)

    total = spark.sql(
        f"""select
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
                cntrycode"""
    )

    total.show()
    print("Q22 Execution time (s): ", time.time() - t1)


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        default="data/tpch-datagen/data",
        help="The folder containing TPCH data",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Space separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        required=False,
        default=1.0,
        help="Scale factor (used in query 11).",
    )
    parser.add_argument(
        "--exclude_io",
        action="store_true",
        required=False,
        help="Separate IO operations from timing.",
    )
    args = parser.parse_args()
    folder = args.folder
    scale_factor = args.scale_factor
    exclude_io = args.exclude_io
    spark = (
        SparkSession.builder.appName("SQL Queries with Spark")
        .config("spark.executor.memory", "16g")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.cores", "8")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.maxResultSize", "8g")
        .getOrCreate()
    )

    queries = args.queries or list(range(1, 23))
    run_queries(folder, spark, queries, scale_factor, exclude_io)


if __name__ == "__main__":
    main()
