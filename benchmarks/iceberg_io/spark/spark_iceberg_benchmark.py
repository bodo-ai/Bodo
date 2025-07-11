""" Copy an S3Table using Spark. This script is intended to be run on an EMR cluster
using the provided terraform script:

Example Usage:
    terraform apply \
    -var="input_table_bucket_arn=arn:aws:s3tables:us-east-2:012345678910:bucket/my-bucket" \
    -var="input_namespace=my_namespace" \
    -var="input_table_name=my_table_"
"""

import argparse
import time

from pyspark.sql import SparkSession


def get_spark(warehouse_loc, table_bucket):
    """Get and configure spark session."""
    SPARK_JAR_PACKAGES = [
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1",
        "software.amazon.s3tables:s3-tables-catalog-for-iceberg-runtime:0.1.7",
    ]

    builder = SparkSession.builder.appName("spark")
    builder.config("spark.jars.packages", ",".join(SPARK_JAR_PACKAGES))

    builder.config(
        f"spark.sql.catalog.{table_bucket}", "org.apache.iceberg.spark.SparkCatalog"
    )
    builder.config(
        f"spark.sql.catalog.{table_bucket}.catalog-impl",
        "software.amazon.s3tables.iceberg.S3TablesCatalog",
    )
    builder.config(f"spark.sql.catalog.{table_bucket}.warehouse", warehouse_loc)
    builder.config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def copy_s3table(spark, source_table, destination_table):
    """Read from source table and copy to destination (replace if exists)."""
    df = spark.sql(f" SELECT * FROM {source_table} ")

    df.writeTo(destination_table).using("Iceberg").tableProperty(
        "format-version", "2"
    ).createOrReplace()


def main():
    parser = argparse.ArgumentParser(description="Run Spark S3Tables job")
    parser.add_argument(
        "--warehouse-loc",
        type=str,
        default="arn:aws:s3tables:us-east-2:012345678910:bucket/my-bucket1121",
        help="The ARN of the S3Tables warehouse location",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="my_namespace",
        help="The namespace for the table",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="my_table_1",
        help="The name of the table to copy",
    )
    args = parser.parse_args()

    warehouse_loc = args.warehouse_loc
    namespace = args.namespace
    table_name = args.table_name

    table_bucket = warehouse_loc.split("/")[-1]

    source_table = f"`{table_bucket}`.{namespace}.`{table_name}`"
    destination_table = f"`{table_bucket}`.{namespace}.`{table_name}_copy`"

    spark = get_spark(warehouse_loc, table_bucket)

    start_copy = time.time()
    copy_s3table(spark, source_table, destination_table)
    print(f"total read/write time: {(time.time() - start_copy):.2f}")


if __name__ == "__main__":
    main()
