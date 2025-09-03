"""
NYC Taxi Monthly Trips with Precipitation

Similar to:
https://github.com/toddwschneider/nyc-taxi-data/blob/c65ad8332a44f49770644b11576c0529b40bbc76/citibike_comparison/analysis/analysis_queries.sql#L1
"""

import argparse
import time

import pandas as pd

import bodo
import bodo.pandas


def get_monthly_travels_weather(weather_dataset, hvfhv_dataset):
    t0 = time.time()
    central_park_weather_observations = pd.read_csv(
        weather_dataset,
        parse_dates=["DATE"],
    )
    central_park_weather_observations = central_park_weather_observations.rename(
        columns={"DATE": "date", "PRCP": "precipitation"}, copy=False
    )
    fhvhv_tripdata = pd.read_parquet(hvfhv_dataset)
    print("Total read time: ", time.time() - t0)

    central_park_weather_observations["date"] = central_park_weather_observations[
        "date"
    ].dt.date
    fhvhv_tripdata["date"] = fhvhv_tripdata["pickup_datetime"].dt.date
    fhvhv_tripdata["month"] = fhvhv_tripdata["pickup_datetime"].dt.month
    fhvhv_tripdata["hour"] = fhvhv_tripdata["pickup_datetime"].dt.hour
    fhvhv_tripdata["weekday"] = fhvhv_tripdata["pickup_datetime"].dt.dayofweek.isin(
        [0, 1, 2, 3, 4]
    )

    monthly_trips_weather = fhvhv_tripdata.merge(
        central_park_weather_observations, on="date", how="inner"
    )
    monthly_trips_weather["date_with_precipitation"] = (
        monthly_trips_weather["precipitation"] > 0.1
    )

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
    monthly_trips_weather = monthly_trips_weather.groupby(
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
    monthly_trips_weather = monthly_trips_weather.sort_values(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    )
    monthly_trips_weather = monthly_trips_weather.rename(
        columns={
            "hvfhs_license_num": "trips",
            "trip_miles": "avg_distance",
        },
        copy=False,
    )

    monthly_trips_weather.to_parquet("monthly_trips_weather.pq")
    print("Total time spent in get_monthly_travels_weather time:", (time.time() - t0))
    return monthly_trips_weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_jit", action="store_true")
    args = parser.parse_args()

    weather_dataset = "s3://bodo-example-data/nyc-taxi/central_park_weather.csv"
    hvfhv_dataset = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/"

    if args.use_jit:
        start = time.time()
        bodo.jit(cache=True)(get_monthly_travels_weather)(
            weather_dataset, hvfhv_dataset
        )
        end = time.time()
        print("Total E2E time:", (end - start))
    else:
        get_monthly_travels_weather.__globals__.update({"pd": bodo.pandas})
        get_monthly_travels_weather(weather_dataset, hvfhv_dataset)


if __name__ == "__main__":
    main()
