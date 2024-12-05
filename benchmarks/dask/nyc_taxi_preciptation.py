from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import time
import pandas as pd

def run_dask_local():
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            start = time.time()
            central_park_weather_observations = dd.read_csv(
                "weather.csv", 
                parse_dates=["DATE"], 
            )
            central_park_weather_observations = central_park_weather_observations.rename(
                columns={"DATE": "date", "PRCP": "precipitation"}
            )
            
            fhvhv_tripdata = dd.read_parquet(
                "taxi_1k.pq", 
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
            get_time_bucket, meta=('hour', 'object')
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
                dropna=True
            ).agg({"hvfhs_license_num": "count", "trip_miles": "mean"}).reset_index()
            monthly_trips_weather = monthly_trips_weather.sort_values(
                by=[
                    "PULocationID",
                    "DOLocationID",
                    "month",
                    "weekday",
                    "date_with_precipitation",
                    "time_bucket",
                ],
                ascending=True
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

    print(monthly_trips_weather.dtypes)
    print(monthly_trips_weather.shape)
    print(monthly_trips_weather.head())
    return monthly_trips_weather


def run_dask_ec2():
    env_vars = {"EXTRA_CONDA_PACKAGES": "s3fs==2024.10.0"}

    with EC2Cluster(
        # NOTE: Setting security = False to avoid large config size
        # https://github.com/dask/dask-cloudprovider/issues/249
        security=False,
        n_workers=4,
        instance_type="c6i.8xlarge",
        # for accessing bodo-example-data
        region="us-east-2",
        debug=True,
        env_vars=env_vars,
    ) as cluster:
        with Client(cluster) as client:
            start = time.time()
            central_park_weather_observations = dd.read_csv(
                "s3://bodo-example-data/nyc-taxi/central_park_weather.csv", 
                parse_dates=["DATE"], 
                storage_options={"anon": True}
            )
            central_park_weather_observations = central_park_weather_observations.rename(
                columns={"DATE": "date", "PRCP": "precipitation"}
            )
            
            fhvhv_tripdata = dd.read_parquet(
                "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata", 
                storage_options={"anon": True}
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
            get_time_bucket, meta=('hour', 'object')
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
            ).agg({"hvfhs_license_num": "count", "trip_miles": "mean"}).reset_index()
            monthly_trips_weather = monthly_trips_weather.sort_values(
                by=[
                    "PULocationID",
                    "DOLocationID",
                    "month",
                    "weekday",
                    "date_with_precipitation",
                    "time_bucket",
                ],
                ascending=True
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

    print(monthly_trips_weather.head())
    return monthly_trips_weather


if __name__ == "__main__":
    run_dask_ec2()
    
    # for debugging, run on a subset of the data.
    # run_dask_local()

