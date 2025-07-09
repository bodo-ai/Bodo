from pyspark.sql import SparkSession

SPARK_JAR_PACKAGES = [
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1",
    "software.amazon.s3tables:s3-tables-catalog-for-iceberg-runtime:0.1.7",
]

table_bucket = "my-bucket"
namespace = "my_namespace"
source_table = f"`{table_bucket}`.{namespace}.`my_table_1`"
destination_table = f"`{table_bucket}`.{namespace}.`my_table_1_copy`"

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

print(spark.sql(f"SHOW TABLES IN `{table_bucket}`.my_namespace").show())

df = spark.sql(f" SELECT * FROM {source_table} ")

df.writeTo(destination_table).using("Iceberg").tableProperty(
    "format-version", "2"
).createOrReplace()

print("DONE!")
