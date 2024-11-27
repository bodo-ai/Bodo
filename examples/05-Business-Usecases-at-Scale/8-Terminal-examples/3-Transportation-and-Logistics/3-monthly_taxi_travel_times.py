"""
NYC Green Taxi Monthly Trips with Preciptation in 2019

Source: https://github.com/toddwschneider/nyc-taxi-data/blob/master/analysis/2017_update/queries_2017.sql

Usage:
    python monthly_taxi_travel_times.py

Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.

Data source: Green Taxi 2019 s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv
             Central Park Weather s3://bodo-example-data/nyc-taxi/central_park_weather.csv
Full dataset: https://github.com/toddwschneider/nyc-taxi-data/blob/master/setup_files/raw_data_urls.txt
            https://github.com/toddwschneider/nyc-taxi-data/blob/master/data/central_park_weather.csv
"""

import bodo
import pandas as pd
import time


@bodo.jit(cache=True)
def get_monthly_travels_weather():
    start = time.time()
    central_park_weather_observations = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/central_park_weather.csv", parse_dates=["date"]
    )
    central_park_weather_observations["date"] = central_park_weather_observations[
        "date"
    ].dt.date

    green_taxi = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv",
        usecols=[0, 1, 5, 6, 8],
        parse_dates=["lpep_pickup_datetime"],
    )
    green_taxi["date"] = green_taxi["lpep_pickup_datetime"].dt.date
    green_taxi["month"] = green_taxi["lpep_pickup_datetime"].dt.month
    green_taxi["hour"] = green_taxi["lpep_pickup_datetime"].dt.hour
    green_taxi["weekday"] = green_taxi["lpep_pickup_datetime"].dt.dayofweek

    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()
    monthly_trips_weather = green_taxi.merge(
        central_park_weather_observations, on="date", how="inner"
    )
    monthly_trips_weather = monthly_trips_weather[
        (monthly_trips_weather["weekday"].isin([1, 2, 3, 4, 5]))
        & (monthly_trips_weather["precipitation"] > 0.1)
    ]
    monthly_trips_weather["time_bucket"] = monthly_trips_weather.hour.replace(
        {
            8: 0,
            9: 0,
            10: 0,
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 2,
            17: 2,
            18: 2,
            18: 2,
            19: 3,
            20: 3,
            21: 3,
            22: 4,
            23: 4,
            0: 4,
            1: 4,
            2: 4,
            3: 4,
            4: 4,
            5: 4,
            6: 4,
            7: 4,
        }
    )
    monthly_trips_weather = monthly_trips_weather.groupby(
        [
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "precipitation",
            "time_bucket",
        ],
        as_index=False,
    ).agg({"VendorID": "count", "trip_distance": "mean"})
    monthly_trips_weather = monthly_trips_weather.sort_values(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "precipitation",
            "time_bucket",
        ]
    )
    monthly_trips_weather = monthly_trips_weather.rename(
        columns={
            "VendorID": "trips",
            "trip_distance": "avg_distance",
            "precipitation": "date_with_precipitation",
        },
        copy=False,
    )
    end = time.time()
    print("Monthly Taxi Travel Times Computation Time: ", end - start)
    print(monthly_trips_weather.head())


get_monthly_travels_weather()
