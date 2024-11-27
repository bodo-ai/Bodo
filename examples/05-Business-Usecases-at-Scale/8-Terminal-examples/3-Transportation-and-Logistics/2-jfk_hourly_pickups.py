"""
NYC Green Taxi JFK daily pickups in 2019

Source: https://github.com/toddwschneider/nyc-taxi-data/blob/master/analysis/2017_update/queries_2017.sql

Usage:
    python jfk_hourly_pickups.py

Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.

Data source: Green Taxi 2019 s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv
Full dataset: https://github.com/toddwschneider/nyc-taxi-data/blob/master/setup_files/raw_data_urls.txt

"""

import bodo
import pandas as pd
import time


@bodo.jit(cache=True)
def get_jfk_hourly_pickups():
    start = time.time()
    green_taxi = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv",
        usecols=[1, 5],
        parse_dates=["lpep_pickup_datetime"],
        dtype={"PULocationID": "int64"},
    )
    green_taxi["pickup_hour"] = green_taxi["lpep_pickup_datetime"].dt.hour
    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()
    trips = green_taxi.loc[green_taxi["PULocationID"] == 132]
    jfk_hourly = trips.groupby(["pickup_hour", "PULocationID"], as_index=False)[
        "lpep_pickup_datetime"
    ].count()
    jfk_hourly = jfk_hourly.rename(
        columns={
            "lpep_pickup_datetime": "trips",
            "PULocationID": "pickup_location_id",
        },
        copy=False,
    )
    jfk_hourly = jfk_hourly.sort_values(
        by=["pickup_hour", "pickup_location_id"],
    )
    end = time.time()
    print("JFK Hourly Pickups Computation Time: ", end - start)
    print(jfk_hourly.head())


get_jfk_hourly_pickups()
