import argparse
import gc
import os
import time
import warnings

from pyspark.sql import SparkSession


def load_tables(spark, base):
    lineitem = spark.read.parquet(f"{base}/lineitem.pq")
    orders = spark.read.parquet(f"{base}/orders.pq")
    customer = spark.read.parquet(f"{base}/customer.pq")
    part = spark.read.parquet(f"{base}/part.pq")
    partsupp = spark.read.parquet(f"{base}/partsupp.pq")
    supplier = spark.read.parquet(f"{base}/supplier.pq")
    nation = spark.read.parquet(f"{base}/nation.pq")
    region = spark.read.parquet(f"{base}/region.pq")

    tables = {
        "lineitem": lineitem,
        "orders": orders,
        "customer": customer,
        "part": part,
        "partsupp": partsupp,
        "supplier": supplier,
        "nation": nation,
        "region": region,
    }

    # Make table names recognizable from spark.sql queries.
    for name, df in tables.items():
        df.createOrReplaceTempView(name)

    return tables


def create_queries(queries, sql_dir="../sql"):
    for q in queries:
        nn = f"{q:02d}"  # zero-padded two-digit string
        sql_path = os.path.join(sql_dir, f"q{nn}.sql")

        # read SQL file
        with open(sql_path, encoding="utf-8") as f:
            sql_text = f.read()

        func_name = f"tpch_q{nn}"

        # Build the function source string
        func_src = (
            f"""
def {func_name}(spark):
    tpch_query = """
            + "'''\\\n"
            + sql_text
            + "\\\n'''\n"
            + """
    df = spark.sql(tpch_query)
    df.collect()
"""
        )

        exec(func_src, globals())


def run_queries(spark, data_folder: str, queries: list[int], scale_factor: float = 1.0):
    load_tables(spark, data_folder)
    create_queries(queries)

    t1 = time.time()

    for query in queries:
        query_func = globals().get(f"tpch_q{query:02}")

        if query_func is None:
            print(f"Query {query:02} not implemented yet.")
            continue

        t2 = time.time()
        query_func(spark)  # run the query
        print(f"Query {query:02} took {time.time() - t2:.2f} seconds")
        spark.catalog.clearCache()
        gc.collect()

    print(f"Total time: {time.time() - t1:.2f} seconds")


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
    args = parser.parse_args()
    folder = args.folder
    scale_factor = args.scale_factor

    spark = (
        SparkSession.builder.appName("SQL Queries with Spark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1,")
        .config("spark.driver.memory", "12g")  # driver JVM heap
        .config("spark.executor.memory", "8g")  # executor JVM heap (cluster mode)
        .config(
            "spark.executor.memoryOverhead", "2g"
        )  # off-heap overhead for executors
        .config("spark.sql.shuffle.partitions", "200")  # reduce per-task pressure
        .getOrCreate()
    )

    queries = args.queries or list(range(1, 23))

    warnings.filterwarnings("ignore")

    run_queries(spark, folder, queries, scale_factor)


if __name__ == "__main__":
    main()
