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
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    # Spark throws a WARNING with a very long stacktrace whenever creating am
    # Iceberg table with Hadoop because it is initially unable to determine that
    # it wrote a `version-hint.text` file, even though it does.
    # Setting the Log Level to "ERROR" hides it
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def create_iceberg_table(
    df: pd.DataFrame, sql_schema, spark_schema, table_name: str, spark=None
):
    if spark is None:
        spark = get_spark()

    col_defs = ",\n".join([" ".join(x) for x in sql_schema])
    # Create the table and then add the data to it.
    # We create table using SQL syntax, because DataFrame API
    # doesn't write the nullability in Iceberg metadata correctly.
    spark.sql(
        f""" 
        CREATE TABLE hadoop_prod.{DATABASE_NAME}.{table_name} (
            {col_defs})
        USING iceberg
        TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
    """
    )
    df = spark.createDataFrame(df, schema=spark_schema)
    df.writeTo(f"hadoop_prod.{DATABASE_NAME}.{table_name}").append()
