import pandas as pd
import numpy as np
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DateType, IntegerType, DoubleType


def spark_sql_q3(data_folder, spark):
    t0 = time.time()
    lineitem_struct = (
    StructType()
    .add("L_ORDERKEY", IntegerType(), True, None)
    .add("L_PARTKEY", IntegerType(), True, None)
    .add("L_SUPPKEY", IntegerType(), True, None)
    .add("L_LINENUMBER", IntegerType(), True, None)
    .add("L_QUANTITY", DoubleType(), True, None)
    .add("L_EXTENDEDPRICE", DoubleType(), True, None)
    .add("L_DISCOUNT", DoubleType(), True, None)
    .add("L_TAX", DoubleType(), True, None)
    .add("L_RETURNFLAG", StringType(), True, None)
    .add("L_LINESTATUS", StringType(), True, None)
    .add("L_SHIPDATE", DateType(), True, None)
    .add("L_COMMITDATE", DateType(), True, None)
    .add("L_RECEIPTDATE", DateType(), True, None)
    .add("L_SHIPINSTRUCT", StringType(), True, None)
    .add("L_SHIPMODE", StringType(), True, None)
    .add("L_COMMENT", StringType(), True, None)
    )


    orders_struct = (
        StructType()
        .add("O_ORDERKEY", IntegerType(), True, None)
        .add("O_CUSTKEY", IntegerType(), True, None)
        .add("O_ORDERSTATUS", StringType(), True, None)
        .add("O_TOTALPRICE", DoubleType(), True, None)
        .add("O_ORDERDATE", DateType(), True, None)
        .add("O_ORDERPRIORITY", StringType(), True, None)
        .add("O_CLERK", StringType(), True, None)
        .add("O_SHIPPRIORITY", IntegerType(), True, None)
        .add("O_COMMENT", StringType(), True, None)
    )


    customer_struct = (
        StructType()
        .add("C_CUSTKEY", IntegerType(), True, None)
        .add("C_NAME", StringType(), True, None)
        .add("C_ADDRESS", StringType(), True, None)
        .add("C_NATIONKEY", IntegerType(), True, None)
        .add("C_PHONE", StringType(), True, None)
        .add("C_ACCTBAL", DoubleType(), True, None)
        .add("C_MKTSEGMENT", StringType(), True, None)
        .add("C_COMMENT", StringType(), True, None)
    )
    lineitem = spark.read.csv(
        data_folder + "/lineitem.tbl", sep="|", schema=lineitem_struct
    )
    orders = spark.read.csv(data_folder + "/orders.tbl", sep="|", schema=orders_struct)
    customer = spark.read.csv(
        data_folder + "/customer.tbl", sep="|", schema=customer_struct
    )

    print("Execution time for the reading: ", ((time.time() - t0) * 1000), " (ms)")

    lineitem.createOrReplaceTempView("lineitem")
    orders.createOrReplaceTempView("orders")
    customer.createOrReplaceTempView("customer")

    t1 = time.time()
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
    print("Execution time for the query: ", ((time.time() - t1) * 1000), " (ms)")
