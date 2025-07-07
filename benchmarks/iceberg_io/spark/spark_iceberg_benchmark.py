import time

from pyspark.sql import SparkSession


def run_iceberg_read_write():
    spark = (
        SparkSession.builder.appName("IcebergBenchmark")
        .config("spark.sql.catalog.tpch", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.tpch.catalog-impl", "org.apache.iceberg.rest.RESTCatalog"
        )
        .config(
            "spark.sql.catalog.tpch.uri",
            "https://s3tables.us-east-2.amazonaws.com/iceberg",
        )
        .config("spark.sql.catalog.tpch.warehouse", "s3://tpch")
        .config("spark.sql.catalog.tpch.rest.signing-name", "s3tables")
        .config("spark.sql.catalog.tpch.rest.signing-region", "us-east-2")
        .config("spark.sql.catalog.tpch.rest.sigv4-enabled", "true")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .getOrCreate()
    )

    start = time.time()

    src_table = "tpch.sf1000.orders"
    dst_table = "tpch.sf1000.orders_copy_spark"

    # print(spark.catalog.listTables("tpch"))

    # Drop destination table if it exists
    if spark.catalog._jcatalog.tableExists(dst_table):
        spark.sql(f"DROP TABLE {dst_table} PURGE")

    # Read from Iceberg
    orders_df = spark.read.format("iceberg").load(src_table)
    orders_df = orders_df.limit(10000)
    # Write to new Iceberg table
    orders_df.writeTo(dst_table).using("iceberg").create()

    print("Read and write completed in", time.time() - start, "seconds")


if __name__ == "__main__":
    run_iceberg_read_write()
