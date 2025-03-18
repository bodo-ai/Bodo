import time

import polars as pl


def get_monthly_travels_weather(weather_dataset_path, hvfhv_dataset_path):
    start = time.time()

    # read datasets lazily
    hvfhv_dataset = pl.scan_parquet(hvfhv_dataset_path)
    weather_dataset = pl.scan_csv(weather_dataset_path, try_parse_dates=True)

    weather_dataset = weather_dataset.select(
        pl.col("DATE").alias("date"),
        (pl.col("PRCP") > 0.1).alias("date_with_precipitation"),
    )

    hvfhv_dataset = hvfhv_dataset.with_columns(
        pl.col("pickup_datetime").dt.date().alias("date"),
        pl.col("pickup_datetime").dt.month().alias("month"),
        pl.col("pickup_datetime").dt.hour().alias("hour"),
        pl.col("pickup_datetime").dt.weekday().is_in([1, 2, 3, 4, 5]).alias("weekday"),
    )

    # merge with weather observations
    monthly_trips_weather = hvfhv_dataset.join(weather_dataset, on="date")

    # place rides in bucket determined by hour of the day
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

    monthly_trips_weather = monthly_trips_weather.with_columns(
        pl.col("hour")
        .map_elements(get_time_bucket, return_dtype=pl.String)
        .alias("time_bucket")
    )

    # get total trips and average distance for all trips
    groupby_columns = [
        "PULocationID",
        "DOLocationID",
        "month",
        "weekday",
        "date_with_precipitation",
        "time_bucket",
    ]

    monthly_trips_weather = monthly_trips_weather.group_by(groupby_columns).agg(
        pl.col("hvfhs_license_num").count().alias("count"),
        pl.col("trip_miles").mean().alias("avg_distance"),
    )

    monthly_trips_weather = monthly_trips_weather.sort(groupby_columns)

    # evaluate the LazyDataframe and store the output
    monthly_trips_weather.sink_parquet("monthly_trips_weather.pq")

    end = time.time()
    print("Monthly Taxi Travel Times Computation Time: ", end - start)

    return monthly_trips_weather


if __name__ == "__main__":
    weather_dataset_path = "s3://bodo-example-data/nyc-taxi/central_park_weather.csv"
    # NOTE: Using rewrite here for consistent schema across parquet files.
    hvfhv_dataset_path = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata_rewrite/"

    get_monthly_travels_weather(weather_dataset_path, hvfhv_dataset_path)
