import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    dayofweek,
    hour,
    month,
    to_date,
)


def get_monthly_travels_weather():
    spark = (
        SparkSession.builder.appName("MonthlyTravelsWeather")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    import pyspark.pandas as ps

    start = time.time()

    # Read in weather data using pandas-on-Spark
    central_park_weather_observations = ps.read_csv(
        "s3a://bodo-example-data/nyc-taxi/central_park_weather.csv",
    ).rename(columns={"DATE": "date", "PRCP": "precipitation"})

    central_park_weather_observations["date"] = ps.to_datetime(
        central_park_weather_observations["date"]
    )

    # Read in trip data using spark
    fhvhv_tripdata = spark.read.parquet(
        "s3a://bodo-example-data/nyc-taxi/fhvhv_tripdata_rewrite/"
    ).drop("__index_level_0__")

    # Convert datetime columns and create necessary features
    fhvhv_tripdata = (
        (
            fhvhv_tripdata.withColumn("date", to_date(col("pickup_datetime")))
            .withColumn("month", month(col("pickup_datetime")))
            .withColumn("hour", hour(col("pickup_datetime")))
            .withColumn(
                "weekday", dayofweek(col("pickup_datetime")).isin([2, 3, 4, 5, 6])
            )
            # pandas-on-Spark doesn't like these datetime columns which is why we use spark apis for the read and this conversion
        )
        .drop("pickup_datetime")
        .drop("dropoff_datetime")
        .drop("on_scene_datetime")
        .drop("request_datetime")
    )
    # Convert trip data to pandas-on-Spark
    fhvhv_tripdata = ps.DataFrame(fhvhv_tripdata)

    # Join trip data with weather observations on 'date'
    monthly_trips_weather = fhvhv_tripdata.merge(
        central_park_weather_observations, on="date", how="inner"
    )

    ## Create a new column for precipitation indicator
    monthly_trips_weather["date_with_precipitation"] = (
        monthly_trips_weather["precipitation"] > 0.1
    )

    ## Define time bucket based on hour of the day
    def get_time_bucket(t):
        bucket = "other"
        if t in (8, 9, 10):
            bucket = "morning"
        elif t in (11, 12, 13, 14, 15):
            bucket = "midday"
        elif t in (16, 17, 18):
            bucket = "afternoon"
        elif t in (19, 20, 21):
            bucket = "evening"
        return bucket

    monthly_trips_weather["time_bucket"] = monthly_trips_weather.hour.map(
        get_time_bucket
    )
    monthly_trips_weather.groupby(
        [
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ],
        as_index=False,
    ).agg({"hvfhs_license_num": "count", "trip_miles": "mean"})

    sorted_data = monthly_trips_weather.sort_values(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    )

    ## Write the results to a parquet file
    sorted_data.to_parquet("out.parquet", mode="overwrite")
    print("Execution time:", time.time() - start)


get_monthly_travels_weather()
