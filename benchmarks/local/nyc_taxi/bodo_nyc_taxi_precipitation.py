"""
Bodo implementation of the High Volume For Hire Vehicle benchmark workload
that can be run locally.

usage:
    python bodo_nyc_taxi_precipitation.py
"""

import time

import pandas as pd

import bodo


@bodo.jit(cache=True)
def get_monthly_travels_weather(fhvhv_dataset):
    start = time.time()
    central_park_weather_observations = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/central_park_weather.csv",
        parse_dates=["DATE"],
        storage_options={"anon": True},
    )
    central_park_weather_observations = central_park_weather_observations.rename(
        columns={"DATE": "date", "PRCP": "precipitation"}, copy=False
    )
    fhvhv_tripdata = pd.read_parquet(fhvhv_dataset, storage_options={"anon": True})
    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()

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
    end = time.time()
    print("Monthly Taxi Travel Times Computation Time: ", end - start)

    start = time.time()
    monthly_trips_weather.to_parquet("bodo_out.pq")
    end = time.time()
    print("Writing time:", (end - start))
    return monthly_trips_weather


if __name__ == "__main__":
    fhvhv_dataset = "s3://bodo-example-data/nyc_taxi/fhvhv_5M_rows.pq"
    get_monthly_travels_weather(fhvhv_dataset)
