# Copied from https://github.com/Bodo-inc/Bodo-examples/blob/master/05-Business-Usecases-at-Scale/7-Terminal-mpiexec-examples/3-Transportation-and-Logistics/1-get_daily_pickups.py
"""
NYC Green Taxi daily pickups in 2019
Source: https://github.com/toddwschneider/nyc-taxi-data/blob/master/analysis/2017_update/queries_2017.sql
Usage:
    mpiexec -n [cores] python get_daily_pickups.py
Data source: Green Taxi 2019 s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv
Full dataset: https://github.com/toddwschneider/nyc-taxi-data/blob/master/setup_files/raw_data_urls.txt
"""

import pandas as pd
import time
import bodo


@bodo.jit(cache=True)
def get_daily_pickups():
    start = time.time()
    green_taxi = pd.read_csv(
        "s3://bodo-example-data/nyc-taxi/green_tripdata_2019.csv",
        usecols=[1, 5],
        parse_dates=["lpep_pickup_datetime"],
        dtype={"lpep_pickup_datetime": "str", "PULocationID": "int64"},
    )
    green_taxi["pickup_date"] = green_taxi["lpep_pickup_datetime"].dt.date

    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()
    daily_pickups_taxi = green_taxi.groupby(
        ["PULocationID", "pickup_date"], as_index=False
    )["lpep_pickup_datetime"].count()
    daily_pickups_taxi = daily_pickups_taxi.rename(
        columns={
            "PULocationID": "pickup_location_id",
            "pickup_date": "date",
            "lpep_pickup_datetime": "trips",
        },
        copy=False,
    )
    daily_pickups_taxi = daily_pickups_taxi.sort_values(by="trips", ascending=False)
    end = time.time()
    print("Daily pickups Computation Time: ", (end - start))
    print(daily_pickups_taxi.head())


if __name__ == "__main__":
    get_daily_pickups()
