""" """

import time

import daft
import ray
from daft import col


def get_monthly_travels_weather(weather_dataset_path, hvfhv_dataset_path):
    start = time.time()

    # read data, rename some columns
    central_park_weather_observations = daft.read_csv(weather_dataset_path)
    central_park_weather_observations = (
        central_park_weather_observations.with_columns_renamed(
            {"DATE": "date", "PRCP": "precipitation"}
        )
    )
    hvfhv_dataset = daft.read_parquet(hvfhv_dataset_path)

    hvfhv_dataset.explain(show_all=True)

    # parse dates
    central_park_weather_observations = central_park_weather_observations.with_column(
        "date",
        central_park_weather_observations["date"].dt.date(),
    )
    hvfhv_dataset = hvfhv_dataset.with_columns(
        {
            "date": col("pickup_datetime").dt.date(),
            "month": col("pickup_datetime").dt.month(),
            "hour": col("pickup_datetime").dt.hour(),
            "weekday": col("pickup_datetime").dt.day_of_week().is_in([0, 1, 2, 3, 4]),
        }
    )

    # merge with weather observations
    monthly_trips_weather = hvfhv_dataset.join(
        central_park_weather_observations, on="date", how="inner"
    )
    monthly_trips_weather = monthly_trips_weather.with_column(
        "date_with_precipitation", col("precipitation") > 0.1
    )

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

    monthly_trips_weather = monthly_trips_weather.with_column(
        "time_bucket",
        col("hour").apply(get_time_bucket, return_dtype=daft.DataType.string()),
    )

    # get total trips and average distance for all trips
    monthly_trips_weather = monthly_trips_weather.groupby(
        [
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    ).agg(col("hvfhs_license_num").count(), col("trip_miles").mean())

    monthly_trips_weather = monthly_trips_weather.sort(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    )

    monthly_trips_weather = monthly_trips_weather.with_columns_renamed(
        {
            "hvfhs_license_num": "trips",
            "trip_miles": "avg_distance",
        },
    )

    # write output to S3
    monthly_trips_weather.write_parquet(
        "s3://test-daft/full_result.pq", write_mode="overwrite"
    )

    end = time.time()
    print("Total E2E time:", (end - start))

    return monthly_trips_weather


def main():
    ray.init(address="auto")
    daft.context.set_runner_ray()

    weather_dataset = "s3://bodo-example-data/nyc-taxi/central_park_weather.csv"
    hvfhv_dataset = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/**"
    monthly_trips_weather = get_monthly_travels_weather(weather_dataset, hvfhv_dataset)

    monthly_trips_weather = monthly_trips_weather.collect()
    print(monthly_trips_weather.show())
    print(monthly_trips_weather.to_pandas())


if __name__ == "__main__":
    main()
