import pandas as pd
from pyspark.sql import SparkSession

DATABASE_NAME = "iceberg_db"


def get_spark():
    spark = (
        SparkSession.builder.appName("Iceberg with Spark")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.2_2.12:0.13.0",
        )
        .config(
            "spark.sql.catalog.hadoop_prod", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config("spark.sql.catalog.hadoop_prod.type", "hadoop")
        .config("spark.sql.catalog.hadoop_prod.warehouse", ".")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .getOrCreate()
    )
    return spark


def create_iceberg_table(df: pd.DataFrame, schema, table_name: str, spark=None):
    if spark is None:
        spark = get_spark()
    df = spark.createDataFrame(df, schema=schema)
    ## To write:
    df.writeTo(f"hadoop_prod.{DATABASE_NAME}.{table_name}").tableProperty(
        "format-version", "2"
    ).tableProperty("write.delete.mode", "merge-on-read").createOrReplace()
