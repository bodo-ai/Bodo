from pyspark.sql import SparkSession

# spark-shell \
# --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,software.amazon.s3tables:s3-tables-catalog-for-iceberg-runtime:0.1.4 \
# --conf spark.sql.catalog.s3tablesbucket=org.apache.iceberg.spark.SparkCatalog \
# --conf spark.sql.catalog.s3tablesbucket.catalog-impl=software.amazon.s3tables.iceberg.S3TablesCatalog \
# --conf spark.sql.catalog.s3tablesbucket.warehouse=arn:aws:s3tables:us-east-1:111122223333:bucket/amzn-s3-demo-table-bucket \
# --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions

SPARK_JAR_PACKAGES = [
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1",
    "software.amazon.s3tables:s3-tables-catalog-for-iceberg-runtime:0.1.7",
    "software.amazon.awssdk:s3:2.25.25",
    "software.amazon.awssdk:sts:2.25.25",
    "software.amazon.awssdk:kms:2.25.25",
    "software.amazon.awssdk:dynamodb:2.25.25",
    "software.amazon.awssdk:glue:2.25.25",
]

table_bucket = "tpch"

builder = SparkSession.builder.appName("spark")
builder.config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))

builder.config(
    f"spark.sql.catalog.{table_bucket}", "org.apache.iceberg.spark.SparkCatalog"
)
builder.config(
    f"spark.sql.catalog.{table_bucket}.catalog-impl",
    "software.amazon.s3tables.iceberg.S3TablesCatalog",
)
builder.config(
    f"spark.sql.catalog.{table_bucket}.warehouse",
    "arn:aws:s3tables:us-east-2:427443013497:bucket/my-bucket1121",
)
builder.config(
    "spark.sql.extensions",
    "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
)

spark = builder.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print(spark.sql(" SELECT * FROM `my-bucket1121`.my_namespace.`my_table_1` ").show())

import time

start = time.time()

# src_table = "tpch.sf1000.orders"
# dst_table = "tpch.sf1000.orders_copy_spark"
src_table = "`my-bucket1121`.my_namespace.`my_table_1`"
dst_table = "`my-bucket1121`.my_namespace.`my_table_copy`"

# if spark.catalog._jcatalog.tableExists("`my-bucket1121`.my_namespace.`my_table`"):
# spark.sql("DROP TABLE IF EXISTS `my-bucket1121`.my_namespace.`my_table` PURGE")

spark.sql(
    " CREATE TABLE IF NOT EXISTS `my-bucket1121`.my_namespace.`my_table` ( id INT, name STRING, value INT ) USING iceberg "
)


spark.sql(
    """
    INSERT INTO `my-bucket1121`.my_namespace.`my_table`
    VALUES 
        (1, 'ABC', 100), 
        (2, 'XYZ', 200)
"""
)

spark.sql("SELECT * FROM `my-bucket1121`.my_namespace.`my_table`").show()

# # Drop destination table if it exists
# if spark.catalog._jcatalog.tableExists(dst_table):
#     spark.sql(f"DROP TABLE {dst_table} PURGE")

# df = spark.sql(" SELECT * FROM `my-bucket1121`.my_namespace.`my_table_1` ")

# df.writeTo(dst_table).using("Iceberg").create()

# if spark.catalog._jcatalog.tableExists(dst_table):
#     spark.sql(" SELECT * FROM `my-bucket1121`.my_namespace.`my_table_copy` ").show()
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


# if __name__ == "__main__":
#     main()
