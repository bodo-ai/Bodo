from pyspark.sql import SparkSession


def main():
    spark = (
        SparkSession.builder.appName("S3TablesCatalog Example")
        .config("spark.sql.catalog.s3tbl", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.s3tbl.catalog-impl",
            "software.amazon.s3tables.iceberg.S3TablesCatalog",
        )
        .config(
            "spark.sql.catalog.s3tbl.warehouse",
            "arn:aws:s3tables:us-east-2:427443013497:bucket/tpch",
        )  # change this
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .getOrCreate()
    )

    # Drop destination table if it exists
    spark.sql("SHOW NAMESPACES IN s3tbl").show()
    spark.sql("SHOW TABLES IN s3tbl.sf100").show()

    """Below is full benchmark code, does not run yet."""
    # start = time.time()

    # src_table = "tpch.sf1000.orders"
    # dst_table = "tpch.sf1000.orders_copy_spark"

    # # print(spark.catalog.listTables("tpch"))

    # # Drop destination table if it exists
    # if spark.catalog._jcatalog.tableExists(dst_table):
    #     spark.sql(f"DROP TABLE {dst_table} PURGE")

    # # Read from Iceberg
    # orders_df = spark.read.format("iceberg").load(src_table)
    # orders_df = orders_df.limit(10000)
    # # Write to new Iceberg table
    # orders_df.writeTo(dst_table).using("iceberg").create()

    # print("Read and write completed in", time.time() - start, "seconds")


if __name__ == "__main__":
    main()
