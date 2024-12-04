from dask.distributed import Client
import numpy as np
import pandas as pd
import dask.dataframe as dd
import time


def run_dask():
    # create cluster on EC2
    # cluster = EC2Cluster(
    #     # NOTE: Setting security = False to avoid large config size
    #     # https://github.com/dask/dask-cloudprovider/issues/249
    #     # TODO: Need to set up network-level security manually or using aws cli
    #     # to avoid exposing the cluster to the internet
    #     security=False,
    #     n_workers=1,
    # )
    # with Client(cluster):
    # create local cluster on all cores
    with Client() as client:
        start = time.time()

        central_park_weather_observations = dd.read_csv(
        "s3://bodo-example-data/nyc-taxi/central_park_weather.csv", parse_dates=["DATE"]
        )
        central_park_weather_observations = central_park_weather_observations.rename(
            columns={"DATE": "date", "PRCP": "precipitation"}
        )
        
        fhvhv_tripdata = dd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/")

        central_park_weather_observations["date"] = central_park_weather_observations[
            "date"
        ].dt.date
        fhvhv_tripdata["date"] = fhvhv_tripdata["pickup_datetime"].dt.date
        fhvhv_tripdata["month"] = fhvhv_tripdata["pickup_datetime"].dt.month
        fhvhv_tripdata["hour"] = fhvhv_tripdata["pickup_datetime"].dt.hour
        fhvhv_tripdata["weekday"] = fhvhv_tripdata["pickup_datetime"].dt.dayofweek.isin(
            [1, 2, 3, 4, 5]
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
            # as_index=False,
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
        )
        monthly_trips_weather = monthly_trips_weather.compute()
        
        end = time.time()
        print("Total read and compute time: ", end - start)
    # cluster.close()
    print(monthly_trips_weather.head())
    return monthly_trips_weather



def run_dask_compute_after_io():
    # create cluster on EC2
    # cluster = EC2Cluster(
    #     # NOTE: Setting security = False to avoid large config size
    #     # https://github.com/dask/dask-cloudprovider/issues/249
    #     # TODO: Need to set up network-level security manually or using aws cli
    #     # to avoid exposing the cluster to the internet
    #     security=False,
    #     n_workers=1,
    # )
    # with Client(cluster):
    # create local cluster on all cores
    with Client() as client:
        start_io = time.time()

        # use .compute() to isolate IO time
        central_park_weather_observations = dd.read_csv(
        "s3://bodo-example-data/nyc-taxi/central_park_weather.csv", parse_dates=["DATE"]
        ).compute()
        fhvhv_tripdata = dd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/").compute()
        
        end_io = time.time()
        print("Read time: ", end_io - start_io)

        # convert back to dask dataframes for computation
        fhvhv_tripdata = dd.from_pandas(fhvhv_tripdata)
        central_park_weather_observations = dd.from_pandas(central_park_weather_observations)
        
        start_compute = time.time()

        central_park_weather_observations = central_park_weather_observations.rename(
            columns={"DATE": "date", "PRCP": "precipitation"}
        )

        central_park_weather_observations["date"] = central_park_weather_observations[
            "date"
        ].dt.date
        fhvhv_tripdata["date"] = fhvhv_tripdata["pickup_datetime"].dt.date
        fhvhv_tripdata["month"] = fhvhv_tripdata["pickup_datetime"].dt.month
        fhvhv_tripdata["hour"] = fhvhv_tripdata["pickup_datetime"].dt.hour
        fhvhv_tripdata["weekday"] = fhvhv_tripdata["pickup_datetime"].dt.dayofweek.isin(
            [1, 2, 3, 4, 5]
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
            # as_index=False,
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
        )
        monthly_trips_weather = monthly_trips_weather.compute()
        
        end_compute = time.time()
        print("Total compute time: ", end_compute - start_compute)
    # cluster.close()
    print(monthly_trips_weather.head())
    return monthly_trips_weather


if __name__ == '__main__':
    # NOTE: need to run dask code here
    result1 = run_dask()
    result2 = run_dask_compute_after_io()
    