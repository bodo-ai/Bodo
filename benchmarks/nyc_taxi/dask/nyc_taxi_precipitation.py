"""This is the script version of the Dask benchmark. Before running, ensure
that your local environment matches the environment on the cloud:

   cd benchmarks/dask
   conda env create -f env.yml
   conda activate benchmark_dask

usage:
   python nyc_taxi_preciptation.py
"""

import time

import dask.dataframe as dd
from dask.distributed import Client


def get_monthly_travels_weather(hvfhv_dataset):
    start = time.time()
    central_park_weather_observations = dd.read_csv(
        "s3://bodo-example-data/nyc-taxi/central_park_weather.csv",
        parse_dates=["DATE"],
        storage_options={"anon": True},
    )
    central_park_weather_observations = central_park_weather_observations.rename(
        columns={"DATE": "date", "PRCP": "precipitation"}
    )

    fhvhv_tripdata = dd.read_parquet(
        hvfhv_dataset,
        storage_options={"anon": True},
    )

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
        get_time_bucket, meta=("hour", "object")
    )
    monthly_trips_weather = (
        monthly_trips_weather.groupby(
            [
                "PULocationID",
                "DOLocationID",
                "month",
                "weekday",
                "date_with_precipitation",
                "time_bucket",
            ],
        )
        .agg({"hvfhs_license_num": "count", "trip_miles": "mean"})
        .reset_index()
    )
    monthly_trips_weather = monthly_trips_weather.sort_values(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ],
        ascending=True,
    )
    monthly_trips_weather = monthly_trips_weather.rename(
        columns={
            "hvfhs_license_num": "trips",
            "trip_miles": "avg_distance",
        },
    )

    # TODO: Write output to S3 once permissions issue is resolved.
    monthly_trips_weather = monthly_trips_weather.compute()

    end = time.time()

    return end - start


def local_get_monthly_travels_weather(hvfhv_dataset):
    """Run Dask on local cluster."""
    with Client():
        total_time = get_monthly_travels_weather(hvfhv_dataset)
        print("Total time for IO and compute:", total_time)


def ec2_get_monthly_travels_weather(hvfhv_dataset):
    """Run Dask on EC2 cluster."""
    from dask_cloudprovider.aws import EC2Cluster

    env_vars = {"EXTRA_CONDA_PACKAGES": "s3fs==2024.10.0"}
    with EC2Cluster(
        # NOTE: Setting security = False to avoid large config size
        # https://github.com/dask/dask-cloudprovider/issues/249
        security=False,
        n_workers=4,
        instance_type="r6i.16xlarge",
        # Region for accessing bodo-example-data
        region="us-east-2",
        env_vars=env_vars,
    ) as cluster:
        with Client(cluster) as client:
            for _ in range(3):
                future = client.submit(get_monthly_travels_weather, hvfhv_dataset)
                total_time = future.result()
                client.restart()
                print("Total time for IO and compute:", total_time)


if __name__ == "__main__":
    hvfhv_dataset = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/"
    ec2_get_monthly_travels_weather(hvfhv_dataset)
