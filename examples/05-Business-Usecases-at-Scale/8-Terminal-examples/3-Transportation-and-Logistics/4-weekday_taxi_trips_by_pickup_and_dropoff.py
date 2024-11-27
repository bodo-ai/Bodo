"""
NYC Green Taxi weekday daily pickup and dropoff in 2019

Source: https://github.com/toddwschneider/nyc-taxi-data/blob/master/citibike_comparison/analysis/analysis_queries.sql

Usage:
    python weekday_taxi_trips_by_pickup_and_dropoff.py

Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.

Data source: Green Taxi 2019 s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv
Full dataset: https://github.com/toddwschneider/nyc-taxi-data/blob/master/setup_files/raw_data_urls.txt
"""

import pandas as pd
import time
import bodo


@bodo.jit(cache=True)
def get_weekday_trips():
    start = time.time()
    green_taxi = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv",
        usecols=[1, 5, 6],
        parse_dates=["lpep_pickup_datetime"],
        dtype={
            "PULocationID": "int64",
            "PULocationID": "int64",
        },
    )
    green_taxi["pickup_date"] = green_taxi["lpep_pickup_datetime"].dt.date
    green_taxi["pickup_dow"] = green_taxi["lpep_pickup_datetime"].dt.dayofweek
    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()
    trips_weekdays = green_taxi[
        (green_taxi["lpep_pickup_datetime"] >= pd.to_datetime("2019-01-01"))
        & (green_taxi["lpep_pickup_datetime"] < pd.to_datetime("2020-01-01"))
        & (green_taxi["pickup_dow"].isin([1, 2, 3, 4, 5]))
    ]
    trips_weekdays = trips_weekdays.groupby(
        ["PULocationID", "DOLocationID"], as_index=False
    ).count()
    trips_weekdays = trips_weekdays[
        ["PULocationID", "DOLocationID", "lpep_pickup_datetime"]
    ]
    trips_weekdays = trips_weekdays.sort_values(by=["PULocationID", "DOLocationID"])
    trips_weekdays = trips_weekdays.rename(
        columns={
            "PULocationID": "pickup_location_id",
            "DOLocationID": "dropoff_location_id",
            "lpep_pickup_datetime": "trips",
        }
    )
    end = time.time()
    print("Weekday Pickup and Dropoff Computation Time: ", end - start)
    print(trips_weekdays.head())


get_weekday_trips()
