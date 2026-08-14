import argparse
import gc
import os
import time
import warnings

from pyspark.sql import SparkSession


def load_tables(spark, base, use_parquet: bool):
    """
    Load TPCH tables either from Parquet files (use_parquet=True)
    or from Iceberg tables (use_parquet=False).

    For Iceberg:
      - If base contains a dot (e.g., "iceberg_catalog.tpch"), tables are loaded as
        spark.read.format("iceberg").load(f"{base}.{table_name}")
      - Otherwise base is treated as a filesystem path and tables are loaded as
        spark.read.format("iceberg").load(f"{base}/{table_name}")
    """
    table_names = [
        "lineitem",
        "orders",
        "customer",
        "part",
        "partsupp",
        "supplier",
        "nation",
        "region",
    ]

    tables = {}

    if use_parquet:
        # Existing parquet behavior
        for name in table_names:
            df = spark.read.parquet(f"{base}/{name}.pq")
            tables[name] = df
    else:
        # Iceberg behavior
        # Decide whether base looks like a catalog.namespace (contains a dot)
        is_catalog_style = "." in base
        for name in table_names:
            if is_catalog_style:
                iceberg_identifier = f"{base}.{name.upper()}"
            else:
                # treat base as a path prefix
                iceberg_identifier = f"{base}/{name.upper()}"
            df = spark.read.format("iceberg").load(iceberg_identifier)
            tables[name.upper()] = df

    # Make table names recognizable from spark.sql queries.
    for name, df in tables.items():
        df.createOrReplaceTempView(name)

    return tables


def load_query(spark, nn: str, sql_dir="../sql") -> str:
    filename = f"q{nn}.sql"

    if sql_dir.startswith("s3://"):
        path = f"{sql_dir}/q{nn}.sql"

        return "\n".join(spark.sparkContext.textFile(path).collect())

    path = os.path.join(sql_dir, filename)
    with open(path) as f:
        sql_text = f.read()

    return sql_text


def create_queries(spark, queries, scale_factor, sql_dir="../sql"):
    for q in queries:
        nn = f"{q:02d}"  # zero-padded two-digit string

        sql_text = load_query(spark, nn, sql_dir)

        # Allow queries to have f-string expressions in them using scale_factor.
        sql_text = f'f"""{sql_text}"""'
        # Calculate those f-string expressions if present.
        sql_text = eval(sql_text)

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
    return df
"""
        )

        exec(func_src, globals())


def run_queries(
    spark,
    data_folder: str,
    queries: list[int],
    scale_factor: float = 1.0,
    sql_dir: str = "../sql",
    use_parquet: bool = False,
    store_output: bool = False,
):
    load_tables(spark, data_folder, use_parquet)
    create_queries(spark, queries, scale_factor, sql_dir)

    t1 = time.time()

    for query in queries:
        print("Running query", query)
        query_func = globals().get(f"tpch_q{query:02}")

        if query_func is None:
            print(f"Query {query:02} not implemented yet.")
            continue

        t2 = time.time()
        output_df = query_func(spark)  # run the query
        print(f"Query {query:02} took {time.time() - t2:.2f} seconds")
        if store_output:
            output_df.coalesce(1).write.parquet(f"q{query:02}_output")
        spark.catalog.clearCache()
        gc.collect()

    print(f"Total time: {time.time() - t1:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        default="data/tpch-datagen/data",
        help="The folder containing TPCH data or the Iceberg catalog.namespace",
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
        "--sql_dir",
        type=str,
        default="../sql",
        help="Directory containing SQL query files.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run queries on GPU (default: CPU).",
    )
    parser.add_argument(
        "--use_parquet",
        action="store_true",
        help="Read data from Parquet files instead of Iceberg (default: False).",
    )
    parser.add_argument(
        "--store_output",
        action="store_true",
        help="Write the output for each query to a file (default: False).",
    )
    args = parser.parse_args()
    folder = args.folder
    scale_factor = args.scale_factor
    run_on_gpu = args.gpu
    use_parquet = args.use_parquet
    store_output = args.store_output

    iceberg_version = "1.11.0"  # or your preferred Iceberg version
    spark_version = "4.0"  # match your Spark major.minor version
    scala_version = "2.13"
    catalog_name = "local"  # arbitrary catalog identifier

    if run_on_gpu:
        spark = (
            SparkSession.builder.appName("SQL Queries with Spark on GPU")
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1,")
            .config("spark.driver.memory", "12g")  # driver JVM heap
            .config("spark.executor.memory", "8g")  # executor JVM heap (cluster mode)
            .config(
                "spark.executor.memoryOverhead", "2g"
            )  # off-heap overhead for executors
            .config("spark.sql.shuffle.partitions", "200")  # reduce per-task pressure
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config("spark.executor.resource.gpu.amount", "2")
            .config("spark.task.resource.gpu.amount", "0.125")
            .config("spark.rapids.memory.pinnedPool.size", "2G")
            .getOrCreate()
        )
    else:
        packages = f"org.apache.iceberg:iceberg-spark-runtime-{spark_version}_{scala_version}:{iceberg_version},org.apache.hadoop:hadoop-aws:3.4.1,software.amazon.awssdk:bundle:2.24.6"
        spark = (
            SparkSession.builder.appName("SQL Queries with Spark")
            .appName("IcebergTPCH")
            .config("spark.jars.packages", packages)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # Enable Iceberg Spark extensions
            .config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            # Register a Spark catalog backed by Iceberg (Hadoop catalog)
            .config(
                f"spark.sql.catalog.{catalog_name}",
                "org.apache.iceberg.spark.SparkCatalog",
            )
            .config(f"spark.sql.catalog.{catalog_name}.type", "hadoop")
            .config(f"spark.sql.catalog.{catalog_name}.warehouse", folder)
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

    run_queries(
        spark, folder, queries, scale_factor, args.sql_dir, use_parquet, store_output
    )


if __name__ == "__main__":
    main()
