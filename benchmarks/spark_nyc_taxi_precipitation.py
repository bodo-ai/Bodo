import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    dayofweek,
    hour,
    mean,
    month,
    to_date,
    when,
)


def get_monthly_travels_weather():
    spark = (
        SparkSession.builder.appName("MonthlyTravelsWeather")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    start = time.time()

    central_park_weather_observations = (
        spark.read.csv(
            "s3a://bodo-example-data/nyc-taxi/central_park_weather.csv",
            header=True,
            inferSchema=True,
        )
        .withColumnRenamed("DATE", "date")
        .withColumnRenamed("PRCP", "precipitation")
    )

    fhvhv_tripdata = spark.read.parquet(
        "s3a://bodo-example-data/nyc-taxi/fhvhv_tripdata/"
    )

    fhvhv_tripdata = (
        fhvhv_tripdata.withColumn("date", to_date(col("pickup_datetime")))
        .withColumn("month", month(col("pickup_datetime")))
        .withColumn("hour", hour(col("pickup_datetime")))
        .withColumn("weekday", dayofweek(col("pickup_datetime")).isin([2, 3, 4, 5, 6]))
    )

    monthly_trips_weather = fhvhv_tripdata.join(
        central_park_weather_observations, on="date", how="inner"
    ).withColumn("date_with_precipitation", col("precipitation") > 0.1)

    # Define time buckets
    time_bucket_expr = (
        when(col("hour").isin(8, 9, 10), "morning")
        .when(col("hour").isin(11, 12, 13, 14, 15), "midday")
        .when(col("hour").isin(16, 17, 18), "afternoon")
        .when(col("hour").isin(19, 20, 21), "evening")
        .otherwise("other")
    )

    monthly_trips_weather = monthly_trips_weather.withColumn(
        "time_bucket", time_bucket_expr
    )

    aggregated_data = monthly_trips_weather.groupBy(
        "PULocationID",
        "DOLocationID",
        "month",
        "weekday",
        "date_with_precipitation",
        "time_bucket",
    ).agg(
        count("hvfhs_license_num").alias("trips"),
        mean("trip_miles").alias("avg_distance"),
    )

    sorted_data = aggregated_data.orderBy(
        "PULocationID",
        "DOLocationID",
        "month",
        "weekday",
        "date_with_precipitation",
        "time_bucket",
    )
    sorted_data.write.parquet("out.parquet")
    print("Execution time:", time.time() - start)


get_monthly_travels_weather()
