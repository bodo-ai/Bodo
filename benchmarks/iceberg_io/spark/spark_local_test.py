from pyspark.sql import SparkSession


def main():
    spark = (
        SparkSession.builder.appName("Local Iceberg Test")
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config(
            "spark.sql.catalog.local.warehouse",
            "file:///Users/chrisoh/chris/Bodo/benchmarks/iceberg_io/spark/warehouse",
        )
        .getOrCreate()
    )

    data = spark.range(10).withColumnRenamed("id", "value")
    data.writeTo("local.db.my_table").using("iceberg").createOrReplace()

    df = spark.read.format("iceberg").table("local.db.my_table")
    df.show()


if __name__ == "__main__":
    main()
